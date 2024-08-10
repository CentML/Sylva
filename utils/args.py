import argparse
import json
import os
import errno

from transformers import SchedulerType


def parse_args():
    parser = argparse.ArgumentParser(description="LLM Fine-tuning")
    # model/data arguments
    parser.add_argument(
        "--data_path", type=str, help="Path to the local or Hugging Face dataset id."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to the local model checkpoints or Hugging Face model id.",
    )
    # training arguments
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size on each GPU device for training.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size on each GPU device for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate before an optimization step",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate (after warming up and before decay)",
    )
    parser.add_argument(
        "--scheduler",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps to warmup the learning rate.",
    )
    parser.add_argument(
        "--max_steps", type=int, default=60000, help="Maximum number of training steps."
    )
    parser.add_argument(
        "--target-score",
        type=float,
        help="The target fine-tuning score. If reached, exit early.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay (in optimizers)."
    )
    parser.add_argument(
        "--no_decay_bias",
        action="store_true",
        help="No weight decay for bias and layer norm weight.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce peak GPU memory usage.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Total number of training epochs.",
    )
    # checkpoints and reproducibility
    parser.add_argument(
        "--log_interval",
        type=int,
        default=1,
        help="Log every ? number of gradient accumulation steps",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=1,
        help="Evaluate every ? number of gradient accumulation steps",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Path to the directory in which the model will be stored.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="A random seed.")
    # Sylva configuration arguments
    parser.add_argument(
        "--preprocess_num_samples",
        type=int,
        default=128,
        help="Number of samples used in the preprocessing step.",
    )
    parser.add_argument(
        "--num_partition",
        type=int,
        default=4,
        help="Number of partition in the hierarchical approximation.",
    )
    parser.add_argument(
        "--sparsity", type=float, default=0, help="Sparsity (%) in the adapters."
    )
    parser.add_argument(
        "--block_size", type=int, default=64, help="Block size in the sparse adapters."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        choices=["fp32", "bf16"],
        help="Fine-tuning precision.",
    )
    parser.add_argument(
        "--scope", type=str, default="decoder.layers.", help="The scope of adapters."
    )

    args = parser.parse_args()

    # create output directory if there isn't one
    if os.path.isdir(args.output_dir) and len(os.listdir(args.output_dir)):
        print(
            f"WARNING: output directory ({args.output_dir}) already exists and non-empty!"
        )
    else:
        try:
            os.makedirs(args.output_dir, exist_ok=True)
        except OSError as e:
            assert e.errno != errno.EEXIST
            pass
        print(f"Created output direcotry ({args.output_dir}).")

    # dump arguments in the output directory
    with open(args.output_dir + "/args.json", "w") as f:
        json.dump(args.__dict__, f, indent=2)

    return args
