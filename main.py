# Copyright 2024 CentML Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import json
import sys
import torch
import time


from sylva.preprocess import preprocess
from utils.args import parse_args
from utils.data import get_dataloaders
from utils.dist import init_process_group
from utils.misc import print_rank_0, prepare, set_random_seed, to_device
from utils.model import (
    load_model_and_tokenizer,
    save_model,
)

from utils.thirdparty.LoRA.adam import (
    create_optimizer_scheduler,
    create_adam_optimizer_from_args,
)

from utils.thirdparty.qlora.mmlu import mmlu_eval


def main():
    # 0. parse arguments and set up
    args = parse_args()
    set_random_seed(args.seed)
    args.device = init_process_group(args)

    # 1. load tokenizer and model
    model, tokenizer = load_model_and_tokenizer(args)

    # 2. prepare data
    train_dataloader, _ = get_dataloaders(
        args=args, tokenizer=tokenizer, train=True, valid=False
    )

    # 3. config adapters
    # use training data for pre-processing if there is no pre-computed masks
    model, m_masks = preprocess(model=model, dataloader=train_dataloader, args=args)
    train_dataloader, valid_dataloader = get_dataloaders(
        args=args, tokenizer=tokenizer, train=True, valid=True
    )

    # 4. create optimizer, learning rate scheduler
    optimizer = create_adam_optimizer_from_args(model, args)
    lr_scheduler = create_optimizer_scheduler(optimizer, args)

    # 5. initialize torch DDP
    model = torch.nn.parallel.DistributedDataParallel(model)

    # 6. define closure to retrieve masks in optimizer
    def closure():
        return model, args.scope, args.block_size, m_masks

    # Train!
    model = to_device(model, args.device)
    model.train()
    best_score = None
    """
    best_score = validation(
        model,
        valid_dataloader,
        args,
        best_score,
        tokenizer,
        lr_scheduler,
        optimizer,
    )
    """

    print_rank_0("\n********** Training **********", args.global_rank)
    print(datetime.datetime.now())

    for epoch in range(args.num_train_epochs):
        model.train()

        iter_time, iter_loss = 0, 0
        for step, batch in enumerate(train_dataloader):
            start = time.perf_counter()
            batch = prepare(batch, args.model_name_or_path, args.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
                if isinstance(outputs, tuple):
                    loss = outputs[1]
                elif hasattr(outputs, "loss"):
                    loss = outputs.loss
                else:
                    raise NotImplementedError
                if torch.isnan(loss):
                    print(f"WARNING: loss is NAN on rank {args.global_rank}, exit!")
                    return
                loss.backward()
            lr_scheduler.step()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step(closure)
                optimizer.zero_grad()
            end = time.perf_counter()

            iter_loss += loss.item()
            iter_time += end - start

            if (
                torch.distributed.get_rank() == 0
                and (step + 1) % args.log_interval == 0
            ):
                log(
                    epoch,
                    step,
                    iter_loss,
                    iter_time,
                    args.log_interval,
                    optimizer,
                    args.per_device_train_batch_size,
                    len(train_dataloader),
                )
                iter_time, iter_loss = 0, 0

            total_steps = epoch * len(train_dataloader) + step + 1
            if total_steps == args.max_steps or total_steps % args.eval_interval == 0:
                best_score = validation(
                    model,
                    valid_dataloader,
                    args,
                    best_score,
                    tokenizer,
                    lr_scheduler,
                    optimizer,
                )
                if best_score >= args.target_score:
                    print_rank_0(
                        f"Reached target score at epoch {epoch} step {step}, exit!"
                    )
                    print(datetime.datetime.now())
                    return

            if total_steps == args.max_steps:
                print_rank_0(
                    "Reached target maximum training steps {args.max_steps}, exit!"
                )
                print(datetime.datetime.now())
                return


def log(
    epoch,
    step,
    iter_loss,
    iter_time,
    log_interval,
    optimizer,
    batch_size,
    num_total_batch,
):
    avg_loss = iter_loss / log_interval
    for group in optimizer.param_groups:
        cur_lr = group["lr"]
        break
    time_per_step = iter_time / log_interval
    samples_per_second = batch_size / time_per_step
    sys.stdout.write(
        f"Epoch {epoch} | Step {step + 1}/{num_total_batch} | Loss {avg_loss:.3f} | LR {cur_lr:.1e} | Time/Batch {time_per_step:.2f}s | Seq/s {samples_per_second:.2f} \r"
    )
    sys.stdout.flush()


def validation(
    model, valid_dataloader, args, best_score, tokenizer, lr_scheduler, optimizer
):
    """
    Evaluate the model. Only save the checkpoint of the best model.
    Return the best score obtained so far.
    """
    results_to_dump = None
    print_rank_0("\n********** Evaluating **********", args.global_rank)
    model.eval()

    accuracy, eval_loss, results_to_dump = mmlu_eval(
        model=model,
        tokenizer=tokenizer,
        valid_dataloader=valid_dataloader,
        device=args.device,
    )
    print_rank_0(
        f"accuracy: {accuracy * 100:.2f}, loss: {eval_loss:.2f}", args.global_rank
    )
    if args.global_rank == 0 and (
        best_score is None or best_score < accuracy
    ):  # the higher the better
        best_score = accuracy
        print_rank_0(f" * best score: {accuracy * 100:.2f}")
        with open(args.output_dir + "/results.json", "w") as f:
            json.dump(results_to_dump, f)
        save_model(
            args,
            model=model,
            tokenizer=tokenizer,
            lr_scheduler=lr_scheduler,
            optimizer=optimizer,
        )

    model.train()
    return best_score


if __name__ == "__main__":
    main()
