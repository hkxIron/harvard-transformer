import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import *

# 获取偶数
def get_even(index: int):
    # 即index//2*2
    if index % 2 == 0:
        return index
    else:
        return index - 1

class AbsolutePositionEncoding(object):
    def __init__(self, hidden_dim=128):
        self.hidden_dim=hidden_dim

    # position 就对应 token 序列中的位置索引 i
    # hidden_dim 就对应词嵌入维度大小 d
    # seq_len 表示 token 序列长度
    def get_position_angle_vec(self, position:int):
        #等价于[position / np.power(10000, 2 * (hid_j // 2) / self.hidden_dim) for hid_j in range(self.hidden_dim)]
        return [position / np.power(10000, get_even(hid_j) / self.hidden_dim) for hid_j in range(self.hidden_dim)]

    def forward(self, seq_len:int):
        # position_angle_vecs.shape = [seq_len, hidden_dim]
        position_angle_vecs = np.array([self.get_position_angle_vec(pos_i) for pos_i in range(seq_len)])

        # 分别计算奇偶索引位置对应的 sin 和 cos 值
        position_angle_vecs[:, 0::2] = np.sin(position_angle_vecs[:, 0::2])  # dim 2t
        position_angle_vecs[:, 1::2] = np.cos(position_angle_vecs[:, 1::2])  # dim 2t+1

        # positional_embeddings.shape = [batch=1, seq_len, hidden_dim]
        positional_embeddings = torch.FloatTensor(position_angle_vecs).unsqueeze(0)
        return positional_embeddings
def test_polar():
    # out=abs⋅cos(angle)+abs⋅sin(angle)⋅j
    abs = torch.tensor([1, 2], dtype=torch.float64)
    #angle = torch.tensor([np.pi, np.pi], dtype=torch.float64)
    angle = torch.tensor([0.1, 0.2], dtype=torch.float64)
    z = torch.polar(abs, angle)
    print("z复数:", z)
    print("复数:",z[0], " 复数模长：",z[0].norm())
    print("复数:",z[1], " 复数模长：",z[1].norm())

if __name__ == '__main__':
    absEncoder = AbsolutePositionEncoding(hidden_dim=4)
    absEnc = absEncoder.forward(seq_len=6)
    print("绝对位置编码：", absEnc)
    test_polar()