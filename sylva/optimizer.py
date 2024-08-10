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
import triton
import triton.language as tl


@torch.no_grad
def batch_sparse_optimize(self, model, adam_buf, scope, block_size, m_masks):
    if not hasattr(self, "sparse_stats"):
        weights = []
        weight_indices = []
        block_x = []
        block_y = []
        total_nnz_blocks = 0
        for n, m in model.named_modules():
            if scope in n and m.__class__.__name__ == "BlockSparseLinear":
                weights.append(m.w0)
                num_nnz_blocks = torch.sum(m_masks[m])
                total_nnz_blocks += num_nnz_blocks
                weight_indices += [len(weights) - 1 for _ in range(num_nnz_blocks)]

                for i in range(m_masks[m].size(0)):
                    for j in range(m_masks[m].size(1)):
                        if m_masks[m][i][j] == 1:
                            block_x.append(i * block_size)
                            block_y.append(j * block_size)
        self.sparse_stats = weights, weight_indices, block_x, block_y, total_nnz_blocks
    step_size = 0
    alpha = 0
    exp_avgs = []
    denoms = []
    for n, m in model.named_modules():
        if scope in n and m.__class__.__name__ == "BlockSparseLinear":
            p = m.w1
            step_size, exp_avg, denom, alpha = (
                adam_buf[p]["step_size"],
                adam_buf[p]["exp_avg"],
                adam_buf[p]["denom"],
                adam_buf[p]["alpha"],
            )
            block_num_bytes = block_size**2 * (16 // 8)  # float 16 / 1 byte
            start_data_ptr = exp_avg[:, 0, :, :].data_ptr()
            exp_avgs += [
                start_data_ptr + block_num_bytes * i for i in range(exp_avg.size(1))
            ]
            start_data_ptr = denom[:, 0, :, :].data_ptr()
            denoms += [
                start_data_ptr + block_num_bytes * i for i in range(denom.size(1))
            ]

    weights, weight_indices, block_x, block_y, total_nnz_blocks = self.sparse_stats
    add(
        weights,
        weight_indices,
        block_x,
        block_y,
        step_size,
        exp_avgs,
        denoms,
        alpha,
        size_x=4096,
        size_y=4096,
        block_size=64,
    )


@triton.jit
def _add(
    weights_ptr,
    weight_indices_ptr,
    step_size,
    exp_avgs_ptr,
    denoms_ptr,
    alpha,
    block_x_ptr,
    block_y_ptr,
    size_x,
    size_y,
    block_size: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    weight_idx = tl.load(weight_indices_ptr + pid)
    weight_ptr = tl.load(weights_ptr + weight_idx).to(tl.pointer_type(tl.bfloat16))
    blk_x = tl.load(block_x_ptr + pid)
    blk_y = tl.load(block_y_ptr + pid)
    exp_avg_ptr = tl.load(exp_avgs_ptr + pid).to(tl.pointer_type(tl.bfloat16))
    denom_ptr = tl.load(denoms_ptr + pid).to(tl.pointer_type(tl.bfloat16))

    for i in range(block_size):
        offsets = (blk_x + i) * size_y + blk_y + tl.arange(0, block_size)
        mask = offsets < size_x * size_y
        w = tl.load(weight_ptr + offsets, mask=mask)
        block_offsets = i * block_size + tl.arange(0, block_size)
        block_mask = block_offsets < block_size * block_size
        exp_avg = tl.load(exp_avg_ptr + block_offsets, mask=block_mask)
        denom = tl.load(denom_ptr + block_offsets, mask=block_mask)
        output = w + step_size * exp_avg / denom + w * alpha

        tl.store(weight_ptr + offsets, output, mask=mask)


def add(
    weights,
    weight_indices,
    block_x,
    block_y,
    step_size,
    exp_avgs,
    denoms,
    alpha,
    size_x,
    size_y,
    block_size,
):
    num_nnz_blocks = len(block_x)
    grid = (num_nnz_blocks,)
    weights_ptr = torch.tensor(
        [w.data_ptr() for w in weights], device="cuda"
    ).contiguous()
    weight_indices_ptr = torch.tensor(weight_indices, device="cuda").contiguous()
    exp_avgs_ptr = torch.tensor(exp_avgs, device="cuda").contiguous()
    denoms_ptr = torch.tensor(denoms, device="cuda").contiguous()
    block_x_ptr = torch.tensor(block_x, device="cuda").contiguous()
    block_y_ptr = torch.tensor(block_y, device="cuda").contiguous()

    _add[grid](
        weights_ptr,
        weight_indices_ptr,
        step_size,
        exp_avgs_ptr,
        denoms_ptr,
        alpha,
        block_x_ptr,
        block_y_ptr,
        size_x,
        size_y,
        block_size,
    )
    return weights
