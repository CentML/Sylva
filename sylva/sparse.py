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


import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.ops


class BlockSparseLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w0, w1, b, x, op1):
        ctx.save_for_backward(x, w0)
        ctx.op1 = op1

        with torch.no_grad():
            y = F.linear(x, w0, b)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        op1 = ctx.op1
        x, w0 = ctx.saved_tensors

        with torch.no_grad():
            if grad_output.size(-1) != 1:
                grad_output = grad_output.unsqueeze(-1)
            if x.size(-1) != 1:
                x = x.unsqueeze(-1)

            left = grad_output.permute((0, 3, 2, 1))
            right = x.permute((0, 3, 2, 1)).to(left.dtype)
            grad_w = op1(left, right)

            grad_input = grad_output.squeeze() @ w0

        # (ctx, w0, w1, b, x, op1)
        return None, grad_w, None, grad_input.unsqueeze(0), None


class BlockSparseLinear(nn.Module):
    def __init__(self, name, w, b, g_layout, block_size, device):
        super().__init__()
        self.name = name
        self.op1 = triton.ops.blocksparse.matmul(
            g_layout.unsqueeze(0),
            block_size,
            "sdd",
            device=device,
            trans_a=False,
            trans_b=True,
        )
        tmp = []
        for i in range(w.size(0) // block_size):
            for j in range(w.size(1) // block_size):
                if g_layout[i, j] == 1:
                    tmp.append(
                        w[
                            block_size * i : block_size * (i + 1),
                            block_size * j : block_size * (j + 1),
                        ].unsqueeze(0)
                    )

        self.w1 = nn.Parameter(
            torch.cat(tmp, dim=0).unsqueeze(0)
        )  # sparse trainable parameters
        self.w0 = w  # frozen pre-trained weights
        self.b = b

        del g_layout, tmp
        torch.cuda.empty_cache()

    def forward(self, x):
        y = BlockSparseLinearFunction.apply(self.w0, self.w1, self.b, x, self.op1)
        return y


def convert_linear_to_block_sparse_linear(
    model,
    scope,
    known_classes,
    m_mask,
    device,
    block_size=64,
    transpose=False,
    dtype="fp32",
):
    assert dtype in ["fp32", "fp16", "bf16"]
    repalce_name = []
    for name, module in model.named_modules():
        if scope in name and module.__class__.__name__ in known_classes:
            repalce_name.append(name)
    for name in repalce_name:
        module = recursive_getattr(model, name)
        w = module.weight.detach().t() if transpose else module.weight.detach()
        tmp = BlockSparseLinear(
            name,
            w,
            module.bias.detach() if module.bias is not None else None,
            g_layout=m_mask[module],
            block_size=block_size,
            device=device,
        )
        recursive_setattr(model, name, tmp)
    return model


def recursive_getattr(model, module_name):
    split_list = module_name.split(".")
    output = model
    for name in split_list:
        output = getattr(output, name)
    return output


def recursive_setattr(model, module_name, module):
    split_list = module_name.split(".")
    output = model
    for name in split_list[:-1]:
        output = getattr(output, name)
    output.__setattr__(split_list[-1], module)
