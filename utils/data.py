import os
import torch

from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import DataLoader, SequentialSampler
from .thirdparty.qlora.oasst1 import DataCollatorForCausalLM
from .thirdparty.LoRA.e2enlg import FT_Dataset


def get_dataloaders(args, tokenizer, train=True, valid=True, eval=True, shuffle=False):
    if "openassistant" or "oasst" in args.data_path:
        return get_oasst1_dataloaders(args, tokenizer)
    elif "e2enlg" in args.data_path:
        return get_e2enlg_dataloaders(
            args, train=train, valid=valid, eval=eval, shuffle=shuffle
        )
    else:
        raise NotImplementedError(f"Dataset {args.data_path} is not supported now!")


def get_e2enlg_dataloaders(args, train, valid, shuffle=False):
    train_dataloader = None
    if train:
        train_dataset = FT_Dataset(
            args.data_path + "/train.jsonl",
            args.per_device_train_batch_size,
            args.max_seq_len,
            joint_lm=False,
            label_smooth=args.label_smooth,
        )
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=args.global_rank
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.per_device_train_batch_size,
            num_workers=0,
            shuffle=shuffle,
            pin_memory=False,
            drop_last=True,
            sampler=train_sampler,
        )

    valid_dataloader = None
    if valid:
        valid_dataset = FT_Dataset(
            args.data_path + "/valid.jsonl",
            args.per_device_eval_batch_size,
            max_seq_length=args.max_seq_len,
            joint_lm=False,
            label_smooth=args.label_smooth,
        )
        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=args.per_device_eval_batch_size,
            num_workers=0,
            shuffle=shuffle,
            pin_memory=False,
            drop_last=True,
            sampler=None,
        )

    return train_dataloader, valid_dataloader


def get_oasst1_dataloaders(args, tokenizer):
    """
    A ~9k sample OASST1 dataset as in QLoRA: Efficient Finetuning of Quantized LLMs (https://arxiv.org/pdf/2305.14314).
    """
    train_dataset = load_dataset("timdettmers/openassistant-guanaco")
    # Map raw columns to input and output
    train_dataset = train_dataset.map(
        lambda x: {
            "input": "",
            "output": x["text"],
        }
    )
    # Remove unused columns.
    train_dataset = train_dataset.remove_columns(
        [
            col
            for col in train_dataset.column_names["train"]
            if col not in ["input", "output"]
        ]
    )
    # Group by length.
    train_dataset = train_dataset.map(
        lambda x: {"length": len(x["input"]) + len(x["output"])}
    )

    source_max_len = 16
    target_max_len = 512
    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=source_max_len,
        target_max_len=target_max_len,
        train_on_source=False,
        predict_with_generate=False,
    )
    train_dataset = split_dataset_by_node(
        train_dataset["train"],
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
    )
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=args.per_device_train_batch_size,
        collate_fn=data_collator,
    )
    valid_dataloader = get_mmlu_dataloader(args, tokenizer)

    return train_dataloader, valid_dataloader


def get_mmlu_dataloader(args, tokenizer):
    mmlu_split = "eval"
    mmlu_dataset = "mmlu-fs"
    mmlu_source_max_len = 2048
    target_max_len = 512
    mmlu_dataset = load_dataset(
        "json",
        data_files={
            "eval": "data/mmlu/five_shot_mmlu_val.json",
            "test": "data/mmlu/five_shot_mmlu_test.json",
        },
    )
    mmlu_dataset = mmlu_dataset[mmlu_split]

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=mmlu_source_max_len,
        target_max_len=target_max_len,
        train_on_source=False,
        predict_with_generate=False,
    )

    dataloader_params = {
        "batch_size": args.per_device_eval_batch_size,
        "collate_fn": data_collator,
        "num_workers": 4,
        "pin_memory": False,
        "shuffle": False,
        "persistent_workers": False,
    }

    if not isinstance(mmlu_dataset, torch.utils.data.IterableDataset):
        dataloader_params["sampler"] = SequentialSampler(mmlu_dataset)
        dataloader_params["drop_last"] = False
        dataloader_params["prefetch_factor"] = None

    mmlu_dataloader = DataLoader(mmlu_dataset, **dataloader_params)
    return mmlu_dataloader
