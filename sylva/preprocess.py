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


import numpy as np
import os
import time
import torch

from .hierarchy import Hierarchy
from .sparse import convert_linear_to_block_sparse_linear

KNOWN_CLASSES = ["MergedLinear", "Conv1D", "Linear"]


def to_device(obj, device=None):
    if hasattr(obj, "items"):
        output = {}
        for k, v in obj.items():
            if device:
                output[k] = v.to(device)
            else:
                output[k] = v.cuda()
    else:
        if device:
            output = obj.to(device)
        else:
            output = obj.cuda()
    return output


def prepare(batch, model_name, device):
    batch = to_device(batch, device)
    if "gpt" in model_name:
        batch["label_smooth"] = 0.1
    return batch


def print_rank_0(msg, rank=None):
    if rank is not None and rank <= 0:
        print(msg)
    elif torch.distributed.get_rank() == 0:
        print(msg)


def zero_grad(model):
    for p in model.parameters():
        p.grad = None
    torch.cuda.empty_cache()


def save_mean_grad(model, scope, divisor, transpose=False):
    m_g = {}
    for n, m in model.named_modules():
        if scope in n and m.__class__.__name__ in KNOWN_CLASSES:
            m_g[m] = (m.weight.grad / divisor).cpu()
            if transpose:
                m_g[m] = m_g[m].t()
    return m_g


def compute_mask(
    model, scope, sparsity, mask_dir, hierarchy, m_g, transpose=False, block_size=64
):
    m_mask = {}
    for n, m in model.named_modules():
        if scope in n and m.__class__.__name__ in KNOWN_CLASSES:
            input_importance = []
            input_dim = m_g[m].size(1)
            for i in range(input_dim):
                input_importance += [
                    m_g[m][:, i].t().to(torch.float32)
                    @ hierarchy.hierarchical_matvec(
                        blocks=hierarchy.m_G[m], vec=m_g[m][:, i].to(torch.float32)
                    )
                ]
            input_importance = torch.tensor(input_importance) / hierarchy.m_Ainv[m]

            output_importance = []
            output_dim = m_g[m].size(0)
            input_dim = m_g[m].size(1)
            for i in range(output_dim):
                output_importance += [
                    m_g[m][i, :].to(torch.float32)
                    @ hierarchy.hierarchical_matvec(
                        blocks=hierarchy.m_A[m], vec=m_g[m][i, :].t().to(torch.float32)
                    )
                ]
            output_importance = torch.tensor(output_importance) / hierarchy.m_Ginv[m]
            importance = torch.einsum("i,o->oi", input_importance, output_importance)

            block_importance = torch.Tensor(
                [
                    [0 for _ in range(input_dim // block_size)]
                    for _ in range(output_dim // block_size)
                ]
            )
            for i in range(output_dim // block_size):
                for j in range(input_dim // block_size):
                    block_importance[i][j] = torch.sum(
                        importance[
                            i * block_size : (i + 1) * block_size,
                            j * block_size : (j + 1) * block_size,
                        ]
                    )

            threshold = np.quantile(block_importance.numpy(), sparsity)
            mask = block_importance > threshold
            m_mask[m] = mask.t() if transpose else mask
            torch.save(m_mask[m], mask_dir + n + "_mask.pt")
    return m_mask


def get_total_numel(model):
    total_numel = 0
    for n, p in model.named_parameters():
        total_numel += p.numel()
    return total_numel


def get_original_trainable_numel(model, class_name, args):
    original_trainable_numel = 0
    for n, m in model.named_modules():
        if args.scope in n and m.__class__.__name__ == class_name:
            original_trainable_numel += m.weight.numel()
    return original_trainable_numel


def report_sparsity(model, total_numel, original_trainable_numel, args):
    trainable_numel = 0
    for p in model.parameters():
        if p.requires_grad:
            trainable_numel += p.numel()
    print_rank_0(
        f"Total sparsity: {trainable_numel/total_numel*100:.3f}% Scope ({args.scope}) sparsity: {trainable_numel/original_trainable_numel*100:.3f}%"
    )


def preprocess(model, dataloader, args, debug=False):
    print_rank_0("\n********** Pre-Processing **********", args.global_rank)
    model = to_device(model, args.device)
    m_mask = None
    mask_dir = args.output_dir + "/masks/"
    original_trainable_numel = get_original_trainable_numel(model, "Linear", args)
    total_numel = get_total_numel(model)
    if os.path.isdir(mask_dir) and len(os.listdir(mask_dir)):
        if "llama" in args.model_name_or_path:
            model = llama_preprocess(
                model, gradient_checkpointing=args.gradient_checkpointing
            )
        # 0. load precomputed masks
        print_rank_0("* load precomputed masks...")
        tic = time.perf_counter()
        n_mask = {}
        m_mask = {}
        for n, m in model.named_modules():
            if args.scope in n and m.__class__.__name__ in KNOWN_CLASSES:
                mask = torch.load(mask_dir + n + "_mask.pt")
                n_mask[n] = mask
                m_mask[m] = mask
        print_rank_0(f"done ({time.perf_counter() - tic:.3f}) sec")

        # 1. convert to block sparse layers
        tic = time.perf_counter()
        print_rank_0("* convert to block sparse layers...")
        model = convert_linear_to_block_sparse_linear(
            model,
            scope=args.scope,
            known_classes=KNOWN_CLASSES,
            m_mask=m_mask,
            device=args.device,
            block_size=args.block_size,
            transpose=False,
            dtype=args.dtype,
        )
        m_mask = {}
        for n, m in model.named_modules():
            if m.__class__.__name__ in ["BlockSparseLinear"]:
                m_mask[m] = n_mask[n]
        del n_mask
        print_rank_0(f"done ({time.perf_counter() - tic:.3f}) sec")

        # 2. set requires_grad
        tic = time.perf_counter()
        for n, m in model.named_modules():
            if args.scope in n and m.__class__.__name__ in ["BlockSparseLinear"]:
                m.w1.requires_grad = True
            else:
                if hasattr(m, "weight"):
                    if m.weight is not None:
                        m.weight.requires_grad = False
                if hasattr(m, "bias"):
                    if m.bias is not None:
                        m.bias.requires_grad = False

        torch.cuda.empty_cache()

        # 6. report sparsity
        report_sparsity(model, total_numel, original_trainable_numel, args)

        return model, m_mask

    os.makedirs(mask_dir, exist_ok=True)
    if "llama" in args.model_name_or_path:
        model = llama_preprocess(
            model, gradient_checkpointing=args.gradient_checkpointing
        )

    print_rank_0("Start preprocessing...")
    # 0. register modules and load data
    hierarchy = Hierarchy(
        scope=args.scope,
        num_partition=args.num_partition,
        known_classes=KNOWN_CLASSES,
        debug=debug,
    )
    hierarchy.register_hooks(model)
    # set parameters that requires gradient in preprocessing
    for n, p in model.named_parameters():
        if args.scope in n:
            p.requires_grad = True
        else:
            p.requires_grad = False

    # 1. compute hierarchical approximation during fw/bw passes
    tic = time.perf_counter()
    print_rank_0("* compute hierarchical approximation...")
    for step, batch in enumerate(dataloader):
        if (step + 1) % 10 == 0:
            print_rank_0(
                f"  sample {step * args.per_device_train_batch_size} out of {args.preprocess_num_samples}"
            )
            print_rank_0(f"  elapsed time: {time.perf_counter() - tic:.2f} sec")
            tic = time.perf_counter()
        batch = prepare(batch, args.model_name_or_path, args.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(**batch)
            if isinstance(outputs, tuple):
                loss = outputs[1]
            elif hasattr(outputs, "loss"):
                loss = outputs.loss
            else:
                raise NotImplementedError

            loss.backward()
        if (step + 1) * args.per_device_train_batch_size >= args.preprocess_num_samples:
            break

    print_rank_0(f"done ({time.perf_counter() - tic:.3f} ) sec")

    # 2. inverse hierarchical approximation, save accumulated grad
    tic = time.perf_counter()
    print_rank_0("* inverse and save grad...")
    hierarchy.inverse(model)
    m_g = save_mean_grad(
        model=model,
        scope=args.scope,
        divisor=args.preprocess_num_samples,
        transpose=False,
    )
    print_rank_0(f"done ({time.perf_counter() - tic:.3f}) sec")

    # 3. compute importance and mask
    tic = time.perf_counter()
    print_rank_0("* compute mask...")
    m_mask = compute_mask(
        model=model,
        scope=args.scope,
        sparsity=args.sparsity,
        mask_dir=mask_dir,
        hierarchy=hierarchy,
        m_g=m_g,
        block_size=args.block_size,
    )
    print_rank_0(f"done ({time.perf_counter() - tic:.3f}) sec")

    # 4. convert to block sparse layers
    n_mask = {}
    for n, m in model.named_modules():
        if args.scope in n and m.__class__.__name__ in KNOWN_CLASSES:
            n_mask[n] = m_mask[m]

    tic = time.perf_counter()
    print_rank_0("* convert to block sparse layers...")
    model = convert_linear_to_block_sparse_linear(
        model,
        scope=args.scope,
        known_classes=KNOWN_CLASSES,
        m_mask=m_mask,
        device=args.device,
        block_size=args.block_size,
        transpose=False,
        dtype=args.dtype,
    )
    m_mask = {}
    for n, m in model.named_modules():
        if m.__class__.__name__ in ["BlockSparseLinear"]:
            m_mask[m] = n_mask[n]
    del n_mask
    print_rank_0(f"done ({time.perf_counter() - tic:.3f}) sec")

    del hierarchy

    # 5. set requires_grad
    tic = time.perf_counter()
    for n, m in model.named_modules():
        if args.scope in n and m.__class__.__name__ in ["BlockSparseLinear"]:
            m.w1.requires_grad = True
        elif m.__class__.__name__ in KNOWN_CLASSES:
            m.weight.requires_grad = False
            if m.bias is not None:
                m.bias.requires_grad = False

    torch.cuda.empty_cache()

    # 6. report sparsity
    report_sparsity(model, total_numel, original_trainable_numel, args)

    return model, m_mask


def llama_preprocess(model, gradient_checkpointing=False, verbose=False):
    for n, m in model.named_modules():
        if m.__class__.__name__ in ["Linear", "BlockSparseLinear"]:
            m = m.to(torch.bfloat16)
        if "norm" in n:
            m = m.to(torch.float32)
        if "lm_head" in n or "embed_tokens" in n:
            if hasattr(m, "weight"):
                m = m.to(torch.bfloat16)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if verbose:
        print_rank_0(model)
        count = 0
        for n, p in model.named_parameters():
            print_rank_0(
                f"[{count}] n={n}, p={p.shape}, requires_grad={p.requires_grad}, dtype={p.dtype}"
            )
            count += 1
    return model
