import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import *

class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim:int, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        #freqs = torch.einsum("i,j->ij", t, self.inv_freq) # outer product
        # 等价于：torch.outer
        # freqs:[max_seq_len_cached, dim//2]
        freqs = torch.outer(t, self.inv_freq).float()
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # emb:[max_seq_len_cached, dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        # cos_cached:[1,1, max_seq_len_cached, dim]
        # sin_cached:[1,1, max_seq_len_cached, dim]
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False) # None代表新增一个维度，与unsqueeze有些相似
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x:torch.Tensor, seq_len=None):
        # x: [batch_size, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached: # 如果比最大长度大，则重新计算cache
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq) # 等价于torch.outer
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        # 否则直接取指定长度seq_len的向量, ...代表取余下所有的维度的数据
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    # x:[batch_size, num_heads, seq_len, head_dim]
    head_dim=x.shape[-1]
    #x1:[batch_size, num_heads, seq_len, head_dim//2]
    #x2:[batch_size, num_heads, seq_len, head_dim//2]
    x1 = x[..., : head_dim// 2]
    x2 = x[..., head_dim // 2 :]
    #return:[batch_size, num_heads, seq_len, head_dim]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # q:[batch_size, num_heads, seq_len, head_dim]
    # k:[batch_size, num_heads, seq_len, head_dim]
    # cos:[1, 1, seq_len, dim=head_dim]
    # sin:[1, 1, seq_len, dim=head_dim]
    # position_ids:[batch_size, seq_len]
    # gather_indices:[batch_size, 1, seq_len, 1]
    gather_indices = position_ids[:, None, :, None]
    head_dim=cos.shape[3]
    """
    torch.gather的理解
    index=[
    [x1,x2,x3], 
    [y1,y2,y3], 
    [z1,z2,z3]]

    index的index矩阵为：
    index_of_index=[
    [(0,0),(0,1),(0,2)], 
    [(1,0),(1,1),(1,2)], 
    [(2,0),(2,1),(2,2)]]

    如果dim=0,用index替换index_of_index第0维的值
    [
    [(x1,0),(x2,1),(x3,2)]
    [(y1,0),(y2,1),(y3,2)] 
    [(z1,0),(z2,1),(z3,2)]]

    如果dim=1,用index替换index_of_index第1维的值
    [
    [(0,x1),(0,x2),(0,x3)]
    [(1,y1),(1,y2),(1,y3)] 
    [(2,z1),(2,z2),(2,z3)]]
    
    gather输入index的shape等于输出value的shape
    """
    # gather_indices: [batch_size, 1, seq_len, head_dim], 最后一维复制dim份
    gather_indices = gather_indices.repeat(1, cos.shape[1], 1, head_dim)

    # cos: [batch_size, 1, seq_len, dim]
    batch = gather_indices.shape[0]
    # cos_batched:[batch, 1, seq_len, dim=head_dim]
    # gather_indices: [batch_size, 1, seq_len, head_dim], 最后一维复制dim份
    cos_batched=cos.repeat(batch, 1, 1, 1)
    # cos: [batch_size, 1, seq_len, head_dim], dim=2是seq_len维,即在dim=2维上使用position_id维度的值
    cos = torch.gather(input=cos_batched, dim=2, index=gather_indices)
    sin_batched=sin.repeat(batch, 1, 1, 1)
    sin = torch.gather(sin_batched, 2, gather_indices)
    # 整体上没有chatglm的好理解
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class LlamaAttention(nn.Module):
    def __init__(self, hidden_size:int=16, num_heads:int=2, max_position_embeddings=2048):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = max_position_embeddings

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)

        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

    def forward(self, x: torch.Tensor, position_ids:torch.Tensor):
        batch_size, seq_len, hidden_size = x.shape

        # query_states:[batch_size, seq_len, num_heads, head_dim] -> [batch_size, num_heads, seq_len, head_dim]
        query_states = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # key_states:[batch_size, seq_len, num_heads, head_dim] -> [batch_size, num_heads, seq_len, head_dim]
        key_states = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # value_states:[batch_size, seq_len, num_heads, head_dim] -> [batch_size, num_heads, seq_len, head_dim]
        value_states = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        # cos:[1,1, seq_len, dim]
        # sin:[1,1, seq_len, dim]
        cos, sin = self.rotary_emb.forward(value_states, seq_len=kv_seq_len)

        # query_states:[batch_size, num_heads, seq_len, head_dim]
        # key_states:[batch_size, num_heads, seq_len, head_dim]
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # attn_weights:[batch_size, num_heads, seq_len, seq_len]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        # 此处忽略了attention mask
        # upcast attention to fp32
        # attn_weights:[batch_size, num_heads, seq_len, seq_len]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # 各个batch_i, head_j之间均独立
        # value_states:[batch_size, num_heads, seq_len, head_dim]
        # attn_output:[batch_size, num_heads, seq_len, head_dim]
        attn_output = torch.matmul(attn_weights, value_states)

        # attn_output:[batch_size, seq_len, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2)
        # attn_output:[batch_size, seq_len, hidden_size]
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        # attn_output:[batch_size, seq_len, hidden_size]
        attn_output = self.o_proj(attn_output)
        return attn_output

def test_rope():
    batch=10
    seq_len=6
    hidden_dim=16
    head_num=2
    head_dim = hidden_dim//head_num
    max_position_embeddings = 2048

    # [batch, seq_len]
    position_ids = torch.tile(torch.arange(0, seq_len), dims=(batch, 1))
    # [batch, seq_len, head_num, head_dim]
    x = torch.randn((batch, seq_len, hidden_dim))

    atten_layer = LlamaAttention(hidden_size=hidden_dim, num_heads=head_num, max_position_embeddings=max_position_embeddings)
    attn_value = atten_layer.forward(x, position_ids)
    print(attn_value.shape)


if __name__ == '__main__':
    test_rope()
