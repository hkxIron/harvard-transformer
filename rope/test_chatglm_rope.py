import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import *

""" PyTorch ChatGLM model. """
class RotaryEmbedding(nn.Module):
    def __init__(self, dim:int, original_impl=False, device=None, dtype=None):
        super().__init__()
        # 原始公式： 1/10000^(-(2i-1)/d), i in [1,2,...,d/2]
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).to(dtype=dtype) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.dim = dim
        self.original_impl = original_impl

    def forward_impl(self, seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000):
        """Enhanced Transformer with Rotary Position Embedding.
        Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
        transformers/rope/__init__.py. MIT License:
        https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
        """
        # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        # theta: [dim=n_elem/2]
        theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=torch.float, device=device) / n_elem))

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        # seq_idx: [seq_len]
        seq_idx = torch.arange(seq_len, dtype=torch.float, device=device)

        # Calculate the product of position index and $\theta_i$
        # idx_theta: [seq_len, dim//2]
        idx_theta = torch.outer(seq_idx, theta).float()

        # cache: [seq_len, dim//2, 2], 最后一维dim=0为cos,最后一维dim=1为sin
        # Concatenates a sequence of tensors along a new dimension.注意与cat不同
        cache = torch.stack([torch.cos(idx_theta),
                                    torch.sin(idx_theta)], dim=-1)

        # this is to mimic the behaviour of complex32, else we will get different results
        if dtype in (torch.float16, torch.bfloat16, torch.int8):
            cache = cache.bfloat16() if dtype == torch.bfloat16 else cache.half()
        return cache

    def forward(self, max_seq_len, offset=0):
        return self.forward_impl(
            max_seq_len, self.dim, dtype=self.inv_freq.dtype, device=self.inv_freq.device
        )

# chatglm中使用的是multi query attention或者不使用
def apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    # x: [seq_len, batch_size, head_num=num_multi_query_groups_per_partition, head_size = hidden_size_per_attention_head]
    seq_len, batch, head_num, head_dim = x.size(0), x.size(1), x.size(2), x.size(3)
    # rope_cache: [seq_len, batch, dim//2, 2]
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    rope_cache = rope_cache[:seq_len]
    # x:[seq_len, batch_size, head_num, head_dim]
    # xshaped:[seq_len, batch_size, head_num, dim//2, 2]
    xshaped = x.reshape(seq_len, batch, head_num, rot_dim // 2, 2)
    # rope_cache:[seq_len, 1, head_num, dim//2, 2]
    rope_cache = rope_cache.view(seq_len, batch, 1, rot_dim//2, 2)
    # x_out2:[seq_len, batch_size, head_num, dim//2, 2]
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1], # x: x0,x1 angle:cos_t, sin_t => cos(x0)-sin(x1)
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1], # x: x1,x0 angle:cos_t, sin_t => cos(x1)+sin(x0)
        ],
        dim=-1,
    )
    # 将dim=3开始flatten成一维向量
    # x_out:[seq_len, batch_size, head_num dim]
    x_out = x_out2.flatten(3)
    return torch.cat((x_out, x_pass), dim=-1)

def test_rope():
    batch=10
    seq_len=6
    hidden_dim=16

    rotary_dim=hidden_dim # 两两分组之前
    rotary_pos_emb_layer = RotaryEmbedding(rotary_dim // 2, original_impl=True)

    # [batch, seq_len]
    position_ids = torch.tile(torch.arange(0, seq_len), dims=(batch, 1))

    head_num=2
    head_dim=hidden_dim//head_num
    # [batch, seq_len, head_num, head_dim]
    query_layer = torch.randn((batch, seq_len, head_num, head_dim))
    key_layer = torch.randn((batch, seq_len, head_num, head_dim))
    value_layer = torch.randn((batch, seq_len, hidden_dim))

    # [seq_len, dim//2, 2], 最后一维dim=0为cos,最后一维dim=1为sin
    rotary_pos_emb = rotary_pos_emb_layer.forward(seq_len)

    # [batch, seq_len, dim//2,2]
    rotary_pos_emb = rotary_pos_emb[position_ids]
    #rotary_pos_emb = rotary_pos_emb[None, (0,:seq_len]
    # [seq_len, batch, dim//2, 2]
    rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()

    # query_layer:[seq_len, batch_size, head_num, head_dim]
    query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
    # query_layer:[batch_size, seq_len, dim]
    query_layer=query_layer.transpose(0,1).reshape(batch, seq_len, hidden_dim)

    key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)
    # key_layer:[batch_size, seq_len, dim]
    key_layer = key_layer.transpose(0,1).reshape(batch, seq_len, hidden_dim)

    # scores.shape = (bs, seq_len, seq_len)
    scores = torch.matmul(query_layer, key_layer.transpose(1, 2)) / math.sqrt(hidden_dim)
    scores = F.softmax(scores.float(), dim=-1)
    output = torch.matmul(scores, value_layer)  # (batch_size, seq_len, dim)
    print(output.shape)

if __name__ == '__main__':
    test_rope()
    pass