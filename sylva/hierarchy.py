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


class Hierarchy:
    def __init__(self, scope, num_partition, known_classes, debug=False) -> None:
        self.scope = scope
        self.num_partition = num_partition
        self.known_classes = known_classes
        self.debug = debug
        self.m_A, self.m_G = {}, {}
        self.m_Ainv, self.m_Ginv = {}, {}

    def register_hooks(self, model):
        for n, m in model.named_modules():
            if self.scope in n and m.__class__.__name__ in self.known_classes:
                if self.debug:
                    print("   |_Hierarchy register:", n)
                m.register_forward_pre_hook(self.save_input)
                m.register_backward_hook(self.save_grad)

    def save_input(self, m, input):
        I = input[0].data.to(torch.float32).cpu()
        I = I.reshape(-1, I.size(-1)).mean(dim=0).unsqueeze(0)
        if m not in self.m_A:
            self.m_A[m] = self.hierarchy_init(I)
        else:
            self.m_A[m] = self.hierarchy_update(I, self.m_A[m], m)

        del I
        torch.cuda.empty_cache()

    def save_grad(self, m, grad_input, grad_output):
        G = grad_output[0].data.to(torch.float32).cpu()
        G = G.reshape(-1, G.size(-1)).mean(dim=0).unsqueeze(0)
        if m not in self.m_G:
            self.m_G[m] = self.hierarchy_init(G)
        else:
            self.m_G[m] = self.hierarchy_update(G, self.m_G[m], m)

        del G
        torch.cuda.empty_cache()

    def hierarchy_init(self, I):
        i = I.size(-1)
        blocks = [[] for _ in range(self.num_partition + 1)]

        # 1/2, 1 = 2^0, 1/4, 2 = 2^1, 1/8, 4 = 2^2, 1/16, 8 = 2^3
        for k in range(self.num_partition):
            size = i // (2 ** (k + 1))
            u_start = size
            v_start = 0
            for j in range(2**k):
                u = I.t()[u_start + size * j : u_start + size * (j + 1), :]
                u_norm = torch.norm(u, p=2)
                v = I[:, v_start + size * j : v_start + size * (j + 1)]
                v_norm = torch.norm(v, p=2)
                blocks[k].append([u / u_norm, u_norm * v_norm, v / v_norm])

        # diag, 2^4 = 16
        k = self.num_partition
        u = I.t()[: i // (2**k)]
        u_norm = torch.norm(u, p=2)
        for j in range(2**k):
            u = I.t()[i // (2**k) * j : i // (2**k) * (j + 1), :]
            blocks[k].append([u @ u.t()])

        return blocks

    def hierarchy_update(self, I, buf, m):
        i = I.size(-1)
        for k in range(self.num_partition):
            size = i // (2 ** (k + 1))
            u_start = size
            v_start = 0
            for j in range(2**k):
                u = I.t()[u_start + size * j : u_start + size * (j + 1), :]
                v = I[:, v_start + size * j : v_start + size * (j + 1)].t()
                buf[k][j] = self.rank_1_update(buf[k][j], [u, v], m)

        # diag, 2^4 = 16
        k = self.num_partition
        for j in range(2**k):
            u = I.t()[i // (2**k) * j : i // (2**k) * (j + 1), :]
            buf[k][j][0] += u @ u.t()

        torch.cuda.empty_cache()

        return buf

    def rank_1_update(self, buf, new, m):
        u, s, v = buf[0], buf[1], buf[2]
        a, b = new[0], new[1]
        r = u.size(1)

        m = u.t() @ a
        p = a - u @ m
        ra = torch.norm(p, p=2)
        p.div_(ra)

        n = v @ b
        q = b - v.t() @ n
        rb = torch.norm(q, p=2)
        q.div_(rb)

        k = torch.zeros(r + 1, r + 1).to(s.device).to(m.dtype)
        k[:r, :r] = s
        k1 = torch.cat([m, torch.tensor([[ra]]).to(m.device).to(m.dtype)], dim=0).t()
        k2 = torch.cat([n, torch.tensor([[rb]]).to(n.device).to(m.dtype)])
        k.add_(k1 @ k2)
        if torch.sum(torch.isnan(k)) > 0:
            return buf
        else:
            pass

        u_, s_, v_ = torch.svd(k.to(torch.float32))
        u_, s_, v_ = (
            u_[:, :1].to(torch.float32),
            s_[:1].to(torch.float32),
            v_[:, :1].to(torch.float32),
        )

        result = [
            torch.cat([u, p], dim=1) @ u_,
            s_,
            (torch.cat([v.t(), q], dim=1) @ v_).t(),
        ]

        del u, s, v, a, b, m, p, n, q, k, k1, k2, u_, s_, v_, buf, new
        torch.cuda.empty_cache()
        return result

    def inverse(self, model):
        for n, m in model.named_modules():
            if self.scope in n and m.__class__.__name__ in self.known_classes:
                self.m_Ainv[m] = self.hierarchy_inverse(self.m_A[m])
                self.m_Ginv[m] = self.hierarchy_inverse(self.m_G[m])

        return self.m_Ainv, self.m_Ginv

    def hierarchy_inverse(self, blocks, damping=1e-6):
        result = [[] for _ in range(self.num_partition + 1)]

        diag_blocks = blocks[-1]  # 2^(num_partition)
        for b in diag_blocks:
            result[-1].append(
                torch.inverse(b[0] + torch.eye(b[0].size(0)).to(b[0].device) * damping)
            )

        def helper(X11, X22, u, s, v):
            Y11 = v @ X11 @ u
            Y12 = s.unsqueeze(-1)
            Y21 = s.unsqueeze(-1)
            Y22 = v @ X22 @ u
            Y1 = torch.cat([Y11, Y12], dim=1)
            Y2 = torch.cat([Y21, Y22], dim=1)
            Y = torch.cat([Y1, Y2], dim=0)
            Y = torch.inverse(Y + torch.eye(Y.size(0)).to(Y.device) * damping)
            C1 = torch.block_diag(X11, X22)
            C2 = (
                torch.block_diag(X11 @ u, X22 @ u)
                @ Y
                @ torch.block_diag(v @ X11, v @ X22)
            )
            C = C1 - C2
            return C

        num_blocks = len(diag_blocks)
        for j in range(1, self.num_partition + 1):
            num_blocks = num_blocks // 2
            for i in range(num_blocks):
                offdiag = blocks[-(j + 1)][i]
                u, s, v = offdiag[0], offdiag[1], offdiag[2]
                X11 = result[-j][i * 2]
                X22 = result[-j][i * 2 + 1]
                C = helper(X11, X22, u, s, v)
                result[-(j + 1)].append(C)

        assert len(result[0]) == 1
        return torch.diag(result[0][0])

    def hierarchical_matvec(self, blocks, vec):
        diag_blocks = blocks[-1]
        result = [[] for _ in range(self.num_partition + 1)]

        num_blocks = len(diag_blocks)

        l0_size = diag_blocks[0][0].size(0)
        for i in range(len(diag_blocks)):
            result[0].append(diag_blocks[i][0] @ vec[l0_size * i : l0_size * (i + 1)])

        def helper(lower_sub_result, upper_sub_result, offdiag, lower_vec, upper_vec):
            u, s, v = offdiag[0], offdiag[1], offdiag[2]
            lower_result = lower_sub_result + u @ (s * (v @ lower_vec))
            upper_result = upper_sub_result + u @ (s * (v @ upper_vec))
            return torch.cat([upper_result, lower_result], dim=0)

        for j in range(1, self.num_partition + 1):
            num_blocks = num_blocks // 2
            cur_size = l0_size * (2**j)
            sub_size = l0_size * (2 ** (j - 1))
            for i in range(num_blocks):
                tmp = helper(
                    result[j - 1][i * 2 + 1],
                    result[j - 1][i * 2],
                    blocks[-(j + 1)][i],
                    vec[cur_size * i : cur_size * i + sub_size],
                    vec[cur_size * i + sub_size : cur_size * (i + 1)],
                )
                result[j].append(tmp)

        assert len(result[-1]) == 1
        return result[-1][0]
