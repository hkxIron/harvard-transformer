# -*- coding: utf-8 -*-
#
# * *v2022: Austin Huang, Suraj Subramanian, Jonathan Sum, Khalid Almubarak,
#    and Stella Biderman.*
# * *[Original](https://nlp.seas.harvard.edu/2018/04/03/attention.html):
#    [Sasha Rush](http://rush-nlp.com/).*

import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import altair as alt
from torch._tensor import Tensor
# alt.renderers.enable('notebook')
# alt.renderers.enable('mimetype')

from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
import GPUtil
import warnings
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# Set to False to skip notebook execution (e.g. for debugging)
warnings.filterwarnings("ignore")

class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None

class PositionWiseFeedForward(nn.Module):
    "Implements FFN equation."
    """
    feedforward并没有什么神秘之处，只不过是针对最后d_model维度进行两层MLP
    可以理解为对最后一维度进行加宽后压缩
    """

    # 注意：d_ff 一般比d_model大很多,如d_ff=2048, d_model=512
    def __init__(self, d_model:int, d_ff:int, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff, bias=True)
        self.w_2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    # x:[batch,seq_len, d_model]
    # out:[batch,seq_len, d_model]
    def forward(self, x:Tensor):
        # 注意：第二层后面没有激活函数
        return self.w_2(self.dropout(self.w_1(x).relu()))


class VocabEmbedding(nn.Module):
    def __init__(self, d_model:int, vocab:int):
        super(VocabEmbedding, self).__init__()
        self.embed = nn.Embedding(num_embeddings=vocab, embedding_dim=d_model)
        self.d_model = d_model

    # x: [batch, seq_len]
    # out:[batch, seq_len, d_model]
    def forward(self, x:Tensor):
        return self.embed(x) * math.sqrt(self.d_model)

#
# $$PE_{(pos,2i)} = \sin(pos / 10000^{2i/d_{\text{model}}})$$
#
# $$PE_{(pos,2i+1)} = \cos(pos / 10000^{2i/d_{\text{model}}})$$
#

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model:int, dropout:float, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        # pe:[max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        # position:[max_len, 1]
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe:[1, max_len, d_model]
        pe = pe.unsqueeze(0) # 这里不能加self.pe
        self.register_buffer("pe", pe)

    # x:[batch, seq_len, d_model]
    def forward(self, x:Tensor):
        # position_enc: [1, seq_len, d_model]
        position_enc = self.pe[:, : x.size(1)].requires_grad_(False)
        # out:[batch, seq_len, d_model]
        x = x + position_enc
        return self.dropout(x)

class EmbeddingPlusPositionEncoding(nn.Module):
    def __init__(self, vocab_embedding:VocabEmbedding, position_encoding:PositionalEncoding):
        super(EmbeddingPlusPositionEncoding, self).__init__()
        self.vocab_embedding = vocab_embedding
        self.position_encoding = position_encoding

    # x: [batch, seq_len]
    # out:[batch, seq_len, d_model]
    def forward(self, x:Tensor):
        # x: [batch, seq_len]
        # embed:[batch, seq_len, d_model]
        embed = self.vocab_embedding(x)
        # out:[batch, seq_len, d_model]
        out = self.position_encoding(embed)
        return out

class MultiHeadedAttention(nn.Module):
    def __init__(self, head_num:int, d_model:int, dropout:float=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % head_num == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // head_num
        self.h = head_num
        # 注意有4个linear + query + key+ value + attenion之后
        self.linears = clones(nn.Linear(in_features=d_model, out_features=d_model, bias=True), 4)
        self.attn = None # 这个只是用来debug显示attention_map图的
        self.dropout = nn.Dropout(p=dropout)

    # query:[batch, seq_len, d_model]
    # key:[batch, seq_len, d_model]
    # vlaue:[batch, seq_len, d_model]
    # mask:
    #   encoder时mask:[batch, 1, seq_len]，
    #   decoder时mask:[batch, seq_len-1, seq_len-1]
    #
    # return: [batch, seq_len, d_model]
    def forward(self, query:Tensor, key:Tensor, value:Tensor, mask:Tensor=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all head_num heads.
            # mask:[batch, 1, seq_len] => [batch, head=1, 1, seq_len], 添加一个head的维度
            #      [batch, seq_len-1, seq_len-1] => [batch, head=1, seq_len-1,seq_len-1], 添加一个head的维度
            mask = mask.unsqueeze(1)

        batch = query.size(0)
        seq_len = query.size(1)

        # 1) Do all the linear projections in batch from d_model => head_num x d_k
        # lin(x): [batch, seq_len, d_model]
        #   view => [batch, seq_len, head_num, d_k=dmodel/head_num]
        #   transpose => [batch, head_num, seq_len, d_k]
        # query, key, value => [batch, head_num, seq_len, d_k]
        query, key, value = [
            lin(x)
                .view(batch, -1, self.h, self.d_k) #  这里-1也是seq_len,但在cross_attn中，key,value的seq_len可能与query的seq_len不同,所以写-1
                .transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.

        # query, key, value:[batch, head_num, seq_len, d_k]
        # mask:
        #   encoder时mask:[batch, 1, 1, seq_len]，
        #   decoder时mask:[batch, 1, seq_len-1, seq_len-1]
        # x: [batch, head_num, seq_len, d_k]
        # attn: [batch, head_num, seq_len, d_k]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        del query
        del key
        del value

        """
        transpose、permute 操作虽然没有修改底层一维数组，但是新建了一份Tensor元信息，并在新的
        元信息中的 重新指定 stride。 torch.view 方法约定了不修改数组本身，只是使用新的形状查看数
        据。如果我们在 transpose、permute 操作后执行 view，Pytorch 会抛出错误：view size is not compatible with input tensor's size and stride
        view 仅在底层数组上使用指定的形状进行变形,transpose、 permute 后使用 contiguous 方法
        则会重新开辟一块内存空间保证数据是在逻辑顺序和内存中是 一致的，连续内存布局减少了CPU对对内存的请求次数
        """
        # 3) "Concat" using a view and apply a final linear.
        # x: [batch, head_num, seq_len, d_k]
        # => [batch, seq_len, head_num, d_k]
        # => [batch, seq_len, d_model= head*d_k]
        x = ( x.transpose(1, 2)
                .contiguous() # 直观的解释是Tensor底层一维数组元素的存储顺序与Tensor按行优先一维展开的元素顺序是否一致,如果不一致，则重新开辟内存并整理数据成一致
                .view(batch, seq_len, self.h * self.d_k))
        # x: [batch, seq_len, d_model]
        # out: [batch, seq_len, d_model]
        out = self.linears[-1](x)
        return out

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size:int, self_attn:MultiHeadedAttention, feed_forward:PositionWiseFeedForward, dropout:float):
        super(EncoderLayer, self).__init__()
        self.self_attn:MultiHeadedAttention = self_attn
        self.feed_forward:PositionWiseFeedForward = feed_forward
        self.residualLayers = clones(NormDropoutResidual(size, dropout), N=2)
        self.size = size

    # x:[batch, seq_len, d_model]
    # mask:
    #   encoder时mask:[batch, 1, seq_len]，
    #   decoder时mask:[batch, seq_len-1, seq_len-1]
    def forward(self, x:Tensor, mask:Tensor):
        "Follow Figure 1 (left) for connections."
        """
        1. self attention
        2. layer norm + residual + dropout
        
        3. feed forward 
        4. layer norm + residual + dropout
        """
        # x:[batch, seq_len, d_model]
        # attention_out:[batch, seq_len, d_model]
        # mask:
        #   encoder时mask:[batch, 1, seq_len]，
        #   decoder时mask:[batch, seq_len-1, seq_len-1]
        x = self.residualLayers[0](x, sublayer=lambda x: self.self_attn(query=x, key=x, value=x, mask=mask))
        # x:[batch, seq_len, d_model]
        x = self.residualLayers[1](x, sublayer=self.feed_forward)
        return x

class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model:int, vocab:int):
        super(Generator, self).__init__()
        self.proj = nn.Linear(in_features=d_model, out_features=vocab)

    # x:[batch_size, seq_len-1, d_model]
    def forward(self, x:Tensor):
        # out:[batch_size, seq_len-1, vocab]
        out = log_softmax(input=self.proj(x), dim=-1) # 即: log(softmax(x)), 为了计算稳定性
        return out

def clones(module:nn.Module, N:int):
    "Produce N identical layers."
    return nn.ModuleList(modules=[copy.deepcopy(module) for _ in range(N)])

class Encoders(nn.Module):
    "Core encoders is a stack of N layers"

    def __init__(self, layer:EncoderLayer, N:int=6):
        super(Encoders, self).__init__()
        self.layers:nn.ModuleList = clones(layer, N)
        self.norm = LayerNorm(features=layer.size)

    # x:[batch, seq_len, d_model]
    # mask:
    #   encoder时mask:[batch, 1, seq_len]，
    #   decoder时mask:[batch, seq_len-1, seq_len-1]
    def forward(self, x:Tensor, mask:Tensor):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers: # 6层transformer
            x = layer(x, mask)
        return self.norm(x) # layer_norm

#
# We employ a residual connection
# [(cite)](https://arxiv.org/abs/1512.03385) around each of the two
# sub-layers, followed by layer normalization
# [(cite)](https://arxiv.org/abs/1607.06450).

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features:int, eps=1e-6):
        super(LayerNorm, self).__init__()
        # a_2,b_2为需要学习的参数
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    # x:[batch, seq_len, d_model]
    def forward(self, x:Tensor):
        mean = x.mean(dim=-1, keepdim=True) # 即模型的每一个维度上计算均值与方差
        std = x.std(dim=-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class NormDropoutResidual(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size:int, dropout:float):
        super(NormDropoutResidual, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    # x:[batch, seq_len, d_model]
    def forward(self, x:Tensor, sublayer:nn.Module):
        "Apply residual connection to any sublayer with the same size."
        # residual layer在之前会layer_norm,之后会drop_out
        """
        1. layer norm + sublayer + dropout
        4. add
        """
        normed = self.norm(x)
        sub_out = sublayer(normed)
        dropped = self.dropout(sub_out)
        return x + dropped

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size:int,
                 self_attn:MultiHeadedAttention,
                 src_attn:MultiHeadedAttention,
                 feed_forward:PositionWiseFeedForward,
                 dropout:float):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn:MultiHeadedAttention = self_attn
        self.src_attn:MultiHeadedAttention = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(NormDropoutResidual(size, dropout), 3)

    # x:[batch, seq_len-1, d_model]
    # encoder_memory:[batch, seq_len, d_model]
    # src_mask:[batch,1,seq_len]
    # tgt_mask:[batch,seq_len-1,seq_len-1]
    # return: [batch, seq_len-1, d_model]
    def forward(self, x:Tensor, encoder_memory:Tensor, src_mask:Tensor, tgt_mask:Tensor):
        "Follow Figure 1 (right) for connections."
        """
        1. self attention
        2. layer norm + dropout + residual
        
        3. src-target cross attention
        4. layer norm + dropout + residual
        
        5. feed forward 
        6. layer norm + dropout + residual
        """
        # 先self attention
        x = self.sublayer[0](x, lambda x: self.self_attn(query=x, key=x, value=x, mask=tgt_mask))
        # 之后cross attention
        x = self.sublayer[1](x, lambda x: self.src_attn(query=x, key=encoder_memory, value=encoder_memory, mask=src_mask))
        x = self.sublayer[2](x, self.feed_forward)
        return x

class Decoders(nn.Module):
    "Generic N layer decoders with masking."

    def __init__(self, layer:DecoderLayer, N:int):
        super(Decoders, self).__init__()
        self.layers:nn.ModuleList = clones(layer, N)
        self.norm = LayerNorm(features=layer.size)

    # x:[batch, seq_len-1, d_model]
    # memory:[batch, seq_len, d_model]
    # src_mask:[batch,1,seq_len]
    # tgt_mask:[batch,seq_len-1,seq_len-1]
    def forward(self, x:Tensor, memory:Tensor, src_mask:Tensor, tgt_mask:Tensor):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    def __init__(self, encoder:Encoders, decoder:Decoders, src_embed:EmbeddingPlusPositionEncoding, tgt_embed:EmbeddingPlusPositionEncoding, generator:Generator):
        super(EncoderDecoder, self).__init__()
        self.encoders:Encoders = encoder
        self.decoders:Decoders = decoder
        self.src_embed:EmbeddingPlusPositionEncoding = src_embed
        self.tgt_embed:EmbeddingPlusPositionEncoding = tgt_embed
        self.generator:Generator = generator

    # src_ids:[batch_size, seq_len]
    # tgt_ids:[batch_size, seq_len-1]
    # src_mask:[batch_size, 1, seq_len]
    # tgt_mask:[batch_size, seq_len-1, seq_len-1]
    def forward(self, src_ids:Tensor, tgt_ids:Tensor, src_mask:Tensor, tgt_mask:Tensor):
        "Take in and process masked src and target sequences."
        """
        典型的翻译encoder-decoder架构
        1.先对所有层进行encode生成memory
        2.再对tgt_input_ids进行解码
        """
        # enc_memory:[batch, seq_len, d_model]
        enc_memory = self.encode(src_ids, src_mask)
        # enc_memory:[batch, seq_len, d_model]
        # src_mask:[batch, 1, seq_len]
        # tgt_ids:[batch, seq_len-1]
        # tgt_mask:[batch, seq_len-1, seq_len-1]
        # out:[batch,seq_len-1,d_model]
        out = self.decode(encoder_memory=enc_memory, src_mask=src_mask, tgt_ids=tgt_ids, tgt_mask=tgt_mask)
        return out

    # src_ids:[batch, seq_len]
    # src_mask:[batch, 1, seq_len]
    # out:[batch, seq_len, d_model]
    def encode(self, src_ids:Tensor, src_mask:Tensor):
        # embed:[batch, seq_len, d_model]
        embed = self.src_embed(src_ids)
        # out:[batch, seq_len, d_model]
        out = self.encoders(embed, src_mask)
        return out

    # encoder_memory:[batch, seq_len, d_model]
    # src_mask:[batch, 1, seq_len]
    # tgt_ids:[batch, seq_len-1]
    # tgt_mask:[batch, seq_len-1, seq_len-1]
    def decode(self, encoder_memory:Tensor, src_mask:Tensor, tgt_ids:Tensor, tgt_mask:Tensor):
        # embed:[batch, seq_len-1, d_model]
        embed = self.tgt_embed(tgt_ids)
        # out:[batch, seq_len-1, d_model]
        out = self.decoders(embed, encoder_memory, src_mask, tgt_mask)
        return out

def subsequent_mask(size:int)->Tensor:
    "Mask out subsequent positions."
    """
    >>> a = torch.randn(3, 3)
    >>> a
    tensor([[ 0.2309,  0.5207,  2.0049],
            [ 0.2072, -1.0680,  0.6602],
            [ 0.3480, -0.5211, -0.4573]])
    >>> torch.triu(a, diagonal=1)
    tensor([[ 0.0000,  0.5207,  2.0049],
            [ 0.0000,  0.0000,  0.6602],
            [ 0.0000,  0.0000,  0.0000]])
    """
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8) # 只保留上三角矩阵元素
    # out: [1, seq_len, seq_len], 对角线及下三角为1,其余为0
    return subsequent_mask == 0 # 只取下三角矩阵

"""
query, key, value => [batch, head_num, seq_len, d_k]
mask:
  encoder时mask:[batch, head=1, 1, seq_len]，
  decoder时mask:[batch, head=1, seq_len-1, seq_len-1]

:return
    attn_value: [batch, head, seq_len, d_k]
    p_attn: [batch, head, seq_len, seq_len]
"""
def attention(query:Tensor, key:Tensor, value:Tensor, mask:Tensor=None, dropout:nn.Dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_model = query.size(-1)
    # 之所以除以d_model,因为如果key,query方差为1,均值为0, 则query@key之后，其方差为d_model
    # query: [batch, head_num, seq_len, d_k]
    # key: [batch, head_num, seq_len, d_k]
    #   => [batch, head_num, d_k, seq_len]
    # scores:[batch, head_num, seq_len, seq_len]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_model)
    # 对于decoder cross attn时，mask下三角为1，其余均为0,因此每个位置只能感知当前及其之前的位置的query
    if mask is not None:
        scores = scores.masked_fill(mask == 0, value=-1e9) # 直接给0所在的位置一个非常大的负数，以至于在经过softmax后为会0分

    # scores:[batch, head_num, seq_len, seq_len]
    # p_attn:[batch, head_num, seq_len, seq_len]
    p_attn = scores.softmax(dim=-1) # 可以输出attention map图
    if dropout is not None:
        p_attn = dropout(p_attn)
    # p_attn:[batch, head_num, seq_len, seq_len]
    # value:[batch, head_num, seq_len, seq_len]
    # attn_value:[batch, head_num, seq_len, d_k]
    attn_value = torch.matmul(p_attn, value)
    return attn_value, p_attn

class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, vocab_size:int, padding_idx:int=0, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        # 其实与cross-entropy-loss差不多, KL_divergence = 交叉熵 - 信息熵
        # cross_entropy = -plog(q), 用q的分布来编码p, 其中p为概率分布，-log(q)为编码长度，即出现概率越高，编码长度越短
        # entropy = -plog(p),用p的分布来编码p
        # kl_div = cross_entropy - entropy = -plog(q) - (-plog(p)) = plog(p) - plog(q) = p*(log(p)-log(q))
        # => target*(log(target) - log(pred))
        self.kldiv_loss = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.true_dist = None

    # pred_scores:[batch*(seq_len-1), vocab]
    # target_ids: [batch*(seq_len-1)]
    def forward(self, pred_scores:Tensor, target_ids:Tensor):
        assert pred_scores.size(1) == self.vocab_size
        true_dist = pred_scores.data.clone()
        # Fills self tensor with the specified value.
        # true_dist:[batch*(seq_len-1), vocab]
        true_dist.fill_(value=self.smoothing / (self.vocab_size - 2)) # 值填充，smoothing一般为0
        """
        https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html?highlight=scatter_#torch.Tensor.scatter_
        This is the reverse operation of the manner described in gather().
        注意：index.dim == src.dim,即均为2维或者均为3维
        self[index[i][j][k]] [j][k] = src[i][j][k]  # if dim == 0
        self[i] [index[i][j][k]] [k] = src[i][j][k]  # if dim == 1
        self[i][j] [index[i][j] [k]] = src[i][j][k]  # if dim == 2
        
        >>> src = torch.arange(1, 11).reshape((2, 5))
        >>> src
        tensor([[ 1,  2,  3,  4,  5],
                [ 6,  7,  8,  9, 10]])
        >>> index = torch.tensor([[0, 1, 2, 0]]) 
        # index的坐标:ind[0,0]=0,ind[0,1]=1,ind[0,2]=2,ind[0,3]=0
        
        >>> torch.zeros(3, 5, dtype=src.dtype).scatter_(dim=0, index=index, src=src)
        tensor([[1, 0, 0, 4, 0], # x[ind[0,0]=0,0]=src[0,0]=1, x[ind[0,3]=0,3]=src[0,3]=4
                [0, 2, 0, 0, 0], # x[ind[0,1]=1,1]=src[0,1]=2
                [0, 0, 3, 0, 0]])# x[ind[0,2]=2,2]=src[0,2]=3
        
        >>> index = torch.tensor([[0, 1, 2], 
                                  [0, 1, 4]]) # index的坐标:ind[0,0]=0,ind[0,1]=1,ind[0,2]=2,ind[1,0]=0,ind[1,1]=1,ind[1,2]=4
        >>> torch.zeros(3, 5, dtype=src.dtype).scatter_(dim=1, index=index, src=src)
        tensor([[1, 2, 3, 0, 0], # x[0,ind[0,0]=0]=src[0,0]=1, x[0,ind[0,1]=1]=src[0,1]=2,x[0,ind[0,2]=2]=src[0,2]=3
                [6, 7, 0, 0, 8], # x[1,ind[1,0]=0]=src[1,0]=6, x[1,ind[1,1]=1]=src[1,1]=7,x[1,ind[1,2]=4]=src[1,2]=8,
                [0, 0, 0, 0, 0]])        
                
        # gather
        >>> t = torch.tensor([[1, 2], [3, 4]])
        >>> torch.gather(input=t, dim=1, index=torch.tensor([[0, 0], [1, 0]]))
        tensor([[ 1,  1],
                [ 4,  3]])
               
        out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
        out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2       
        =>
        ind[0,0]=0
        ind[0,1]=0
        ind[1,0]=1
        ind[1,1]=0
        x[0,0]=in[0,ind[0,0]=0]=1
        x[0,1]=in[0,ind[0,1]=0]=1
        x[1,0]=in[1,ind[1,0]=1]=3
        x[1,1]=in[1,ind[1,1]=0]=4
        
        squeeze:
        Returns a tensor with all specified dimensions of input of size 1 removed.如果指定 dim!=1，则什么都不做
        
        unsqueeze:
        Returns a new tensor with a dimension of size one inserted at the specified position.
        """
        # true_dist:[batch*(seq_len-1), vocab]
        # target_ids: [batch*(seq_len-1)]
        #      => [batch*(seq_len-1), 1]
        true_dist.scatter_(dim=1, index=target_ids.data.unsqueeze(1), value=self.confidence) # 直接对于target_ids所在的位置赋值confidence=1
        # true_dist:[batch*(seq_len-1), vocab]
        true_dist[:, self.padding_idx] = 0

        # target_ids: [batch*(seq_len-1)]
        # mask-indexs: []
        mask_indexs = torch.nonzero(target_ids.data == self.padding_idx) # 返回值为padding_idx元素的(i,j)索引值
        if mask_indexs.dim() > 0: # 若mask是向量
            """
            >>> x = torch.tensor([[1, 2, 3], 
                                  [4, 5, 6], 
                                  [7, 8, 9]], dtype=torch.float)
            >>> index = torch.tensor([0, 2])
            >>> x.index_fill_(dim=1, index=index, value=-1)
            tensor([[-1.,  2., -1.],
                    [-1.,  5., -1.],
                    [-1.,  8., -1.]])
            tensor.detach():
            Returns a new Tensor, detached from the current graph. The result will never require gradient
            """
            true_dist.index_fill_(dim=0, index=mask_indexs.squeeze(), value=0.0) # 在mask 的地方赋值0
        self.true_dist = true_dist
        # out:[1]
        out = self.kldiv_loss(input=pred_scores, target=true_dist.clone().detach())
        return out

# > This code predicts a translation using greedy decoding for simplicity.
# src:[batch=1, seq_len]
# src_mask:[batch=1, 1,seq_len]
def greedy_decode(model:EncoderDecoder, src:Tensor, src_mask:Tensor, max_len:int, start_symbol:int)->Tensor:
    # memory:[batch=1, seq_len, d_model]
    memory = model.encode(src, src_mask)
    # ys:[batch=1,seq_len=1]
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        # tgt_mask: [1, cur_seq_len, cur_seq_len], 对角线及下三角为1,其余为0
        cur_seq_len = ys.size(1)
        tgt_mask = subsequent_mask(cur_seq_len).type_as(src.data)
        # src_mask:[batch=1, 1,seq_len]
        # ys:[batch=1, seq_len=1]
        # out:[batch, cur_seq_len, d_model]
        out = model.decode(memory, src_mask, ys, tgt_mask)
        # last_out: [d_model], 取最后一列进行预测
        last_out = out[:, -1]
        # prob:[batch=1, vocab_size]
        prob = model.generator(last_out)
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        # ys:[batch=1,cur_seq_len+1]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)],
            dim=1)

    # ys:[batch=1, seq_len]
    return ys

"""
N: layer number
注意：feedforward一般比d_model大很多
"""
def make_model(src_vocab_size:int, tgt_vocab_size:int, N=6, d_model=512, d_ff=2048, head_num=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    multi_attention = MultiHeadedAttention(head_num, d_model)
    feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        encoder=Encoders(EncoderLayer(d_model, c(multi_attention), c(feed_forward), dropout), N),
        decoder=Decoders(DecoderLayer(d_model, c(multi_attention), c(multi_attention), c(feed_forward), dropout), N),
        # src_embed=nn.Sequential(VocabEmbedding(d_model, src_vocab_size),c(position)),
        # tgt_embed=nn.Sequential(VocabEmbedding(d_model, tgt_vocab_size),c(position)),
        # 注意：src, tgt并没有共享embedding,因为它们的embedding size不一样
        src_embed=EmbeddingPlusPositionEncoding(VocabEmbedding(d_model, src_vocab_size), c(position)),
        tgt_embed=EmbeddingPlusPositionEncoding(VocabEmbedding(d_model, tgt_vocab_size), c(position)),
        generator=Generator(d_model, tgt_vocab_size),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src:Tensor, tgt:Tensor=None, pad=2):  # 2 = <blank>
        # src:[batch, seq_len]
        self.src = src
        # src_mask:[batch, 1, seq_len]
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            # tgt:[batch, seq_len-1]
            self.tgt = tgt[:, :-1]
            # tgt_y:[batch, seq_len-1]
            self.tgt_y = tgt[:, 1:] # y右移一位,即输入什么就输出什么
            # tgt_mask:[batch, seq_len-1, seq_len-1], 下三角矩阵
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            # ntokens:int
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt:Tensor, pad:int):
        "Create a mask to hide padding and future words."
        # tgt:[batch, seq_len]
        # tgt_mask:[batch, 1, seq_len]
        tgt_mask = (tgt != pad).unsqueeze(-2)
        # future_mask:[1, seq_len, seq_len], 下三角矩阵，对角线及下三角全为1，其余均为0
        future_mask = subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        # tgt_mask:[batch, 1, seq_len]
        tgt_mask = tgt_mask & future_mask # 下三角矩阵
        # tgt_mask:[batch, seq_len, seq_len]
        return tgt_mask


# > Next we create a generic training and scoring function to keep
# > track of loss. We pass in a generic loss compute function that
# > also handles parameter updates.
class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed

class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator:Generator, criterion:LabelSmoothing):
        self.generator = generator
        self.criterion = criterion

    # model_out:[batch, seq_len-1, d_model]
    # y_ids:[batch, seq_len-1]
    def __call__(self, model_out, y_ids, norm:float):
        # model_out:[batch, seq_len-1, d_model]
        # predict:[batch, seq_len-1, vocab]
        predict = self.generator(model_out)
        # predict:[batch, seq_len-1, vocab]
        #      => [batch*(seq_len-1), vocab]
        # y_ids: [batch, seq_len-1]
        #     => [batch*(seq_len-1)]
        loss = self.criterion(predict.contiguous().view(-1, predict.size(-1)),
                               y_ids.contiguous().view(-1))
        return loss.data, loss/norm

def run_epoch(data_iter, model:EncoderDecoder,
              loss_compute:SimpleLossCompute,
              optimizer:torch.optim.Optimizer,
              scheduler,
              mode="train",
              accum_iter=1,
              train_state=TrainState(),):
    """Train a single epoch"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0

    # 注意：model与loss一般分开计算
    for i, batch in enumerate(data_iter):
        # src:[batch, seq_len]
        # tgt:[batch, seq_len-1]
        # src_mask:[batch, 1, seq_len]
        # tgt_mask:[batch, seq_len-1, seq_len-1]
        # out:[batch, seq_len-1, d_model]
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        #out = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)

        # out:[batch, seq_len-1, d_model]
        # tgt_y:[batch, seq_len-1]
        # ntokens:720
        # loss: scalar, loss_node:scalar
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        # loss_node = loss_node / accum_iter
        if mode == "train" or mode == "train+log":
            loss_node.backward() # 反向传播，计算所有tensor的梯度
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0: # 其用意是累计多次的亲梯度，然后n个step一起更新
                optimizer.step() # parameter update
                optimizer.zero_grad(set_to_none=True) # 将所有参数梯度置为0
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step() # 更新lr的

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"] # 获取lr
            elapsed = time.time() - start
            print(("Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e" )
                  % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state

"""
可以利用下面的函数进行画图，图形先急骤上升，后面缓慢下降,见 images/lr_rate.png
x=np.arange(0,4000)
import matplotlib.pyplot as plt
f=np.frompyfunc(lambda a:rate(a, model_size=512, factor=1.0, warmup=400),1,1)
y=f(x)
plt.plot(x,y)
"""
def rate(step:int, model_size:int, factor:float, warmup:int):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * ( model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)) )

# ## Data Loading
#
# > We will load the dataset using torchtext and spacy for
# > tokenization.

# %%
# Load spacy tokenizer models, download them if they haven't been
# downloaded already

def load_tokenizers():
    import spacy
    try:
        spacy_de = spacy.load("de_core_news_sm")
    except IOError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_de, spacy_en


# %% id="t4BszXXJTsqL" tags=[]
def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]


def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])


def build_vocabulary(spacy_de, spacy_en):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    print("Building German Vocabulary ...")
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    vocab_src = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_de, index=0),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    print("Building English Vocabulary ...")
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_en, index=1),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt


def load_vocab(spacy_de, spacy_en):
    if not exists("vocab.pt"):
        vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en)
        torch.save((vocab_src, vocab_tgt), "vocab.pt")
    else:
        vocab_src, vocab_tgt = torch.load("vocab.pt")
    print("Finished.\nVocabulary sizes:")
    print(len(vocab_src))
    print(len(vocab_tgt))
    return vocab_src, vocab_tgt


def collate_batch(
    batch,
    src_pipeline,
    tgt_pipeline,
    src_vocab,
    tgt_vocab,
    device,
    max_padding=128,
    pad_id=2,
):
    bs_id = torch.tensor([0], device=device)  # <s> token id
    eos_id = torch.tensor([1], device=device)  # </s> token id
    src_list, tgt_list = [], []
    for (_src, _tgt) in batch:
        processed_src = torch.cat(
            [
                bs_id,
                torch.tensor(
                    src_vocab(src_pipeline(_src)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        processed_tgt = torch.cat(
            [
                bs_id,
                torch.tensor(
                    tgt_vocab(tgt_pipeline(_tgt)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        src_list.append(
            # warning - overwrites values for negative values of padding - len
            pad(
                processed_src,
                (
                    0,
                    max_padding - len(processed_src),
                ),
                value=pad_id,
            )
        )
        tgt_list.append(
            pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),
                value=pad_id,
            )
        )

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    return (src, tgt)


def create_dataloaders(
    device,
    vocab_src,
    vocab_tgt,
    spacy_de,
    spacy_en,
    batch_size=12000,
    max_padding=128,
    is_distributed=True,
):
    # def create_dataloaders(batch_size=12000):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    def collate_fn(batch):
        return collate_batch(
            batch,
            tokenize_de,
            tokenize_en,
            vocab_src,
            vocab_tgt,
            device,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<blank>"],
        )

    train_iter, valid_iter, test_iter = datasets.Multi30k( language_pair=("de", "en") )

    train_iter_map = to_map_style_dataset(
        train_iter
    )  # DistributedSampler needs a dataset len()
    train_sampler = (
        DistributedSampler(train_iter_map) if is_distributed else None
    )
    valid_iter_map = to_map_style_dataset(valid_iter)
    valid_sampler = (
        DistributedSampler(valid_iter_map) if is_distributed else None
    )

    train_dataloader = DataLoader(
        train_iter_map,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_iter_map,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn,
    )
    return train_dataloader, valid_dataloader



# %% [markdown] id="90qM8RzCTsqM"
# ## Training the System

# %%
def train_worker(
    gpu,
    ngpus_per_node,
    vocab_src,
    vocab_tgt,
    spacy_de,
    spacy_en,
    config,
    is_distributed=False,
):
    print(f"Train worker process using GPU: {gpu} for training", flush=True)
    torch.cuda.set_device(gpu)

    pad_idx = vocab_tgt["<blank>"]
    d_model = 512
    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.cuda(gpu)
    module = model
    is_main_process = True
    if is_distributed:
        dist.init_process_group(
            "nccl", init_method="env://", rank=gpu, world_size=ngpus_per_node
        )
        model = DDP(model, device_ids=[gpu])
        module = model.module
        is_main_process = gpu == 0

    criterion = LabelSmoothing(
        vocab_size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1
    )
    criterion.cuda(gpu)

    train_dataloader, valid_dataloader = create_dataloaders(
        gpu,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=config["batch_size"] // ngpus_per_node,
        max_padding=config["max_padding"],
        is_distributed=is_distributed,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, d_model, factor=1, warmup=config["warmup"]
        ),
    )
    train_state = TrainState()

    for epoch in range(config["num_epochs"]):
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch)
            valid_dataloader.sampler.set_epoch(epoch)

        model.train()
        print(f"[GPU{gpu}] Epoch {epoch} Training ====", flush=True)
        _, train_state = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=config["accum_iter"],
            train_state=train_state,
        )

        GPUtil.showUtilization()
        if is_main_process:
            file_path = "%s%.2d.pt" % (config["file_prefix"], epoch)
            torch.save(module.state_dict(), file_path)
        torch.cuda.empty_cache()

        print(f"[GPU{gpu}] Epoch {epoch} Validation ====", flush=True)
        model.eval()
        sloss = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )
        print(sloss)
        torch.cuda.empty_cache()

    if is_main_process:
        file_path = "%sfinal.pt" % config["file_prefix"]
        torch.save(module.state_dict(), file_path)


# %% tags=[]
def train_distributed_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    from origin.the_annotated_transformer import train_worker

    ngpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    print(f"Number of GPUs detected: {ngpus}")
    print("Spawning training processes ...")
    mp.spawn(
        train_worker,
        nprocs=ngpus,
        args=(ngpus, vocab_src, vocab_tgt, spacy_de, spacy_en, config, True),
    )


def train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    if config["distributed"]:
        train_distributed_model(
            vocab_src, vocab_tgt, spacy_de, spacy_en, config
        )
    else:
        train_worker(
            0, 1, vocab_src, vocab_tgt, spacy_de, spacy_en, config, False
        )


def load_trained_model():
    config = {
        "batch_size": 32,
        "distributed": False,
        "num_epochs": 8,
        "accum_iter": 10,
        "base_lr": 1.0,
        "max_padding": 72,
        "warmup": 3000,
        "file_prefix": "multi30k_model_",
    }
    model_path = "multi30k_model_final.pt"
    if not exists(model_path):
        train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config)

    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(torch.load("multi30k_model_final.pt"))
    return model

# if False:
#     model.src_embed[0].lut.weight = model.tgt_embeddings[0].lut.weight
#     model.generator.lut.weight = model.tgt_embed[0].lut.weight


# %% id="hAFEa78JokDB"
def average(model, models):
    "Average models into model"
    for ps in zip(*[m.params() for m in [model] + models]):
        ps[0].copy_(torch.sum(*ps[1:]) / len(ps[1:]))


# Load data and model for output checks

# %%
def check_outputs(
    valid_dataloader,
    model,
    vocab_src,
    vocab_tgt,
    n_examples=15,
    pad_idx=2,
    eos_string="</s>",
):
    results = [()] * n_examples
    for idx in range(n_examples):
        print("\nExample %d ========\n" % idx)
        b = next(iter(valid_dataloader))
        rb = Batch(b[0], b[1], pad_idx)
        greedy_decode(model, rb.src, rb.src_mask, 64, 0)[0]

        src_tokens = [
            vocab_src.get_itos()[x] for x in rb.src[0] if x != pad_idx
        ]
        tgt_tokens = [
            vocab_tgt.get_itos()[x] for x in rb.tgt[0] if x != pad_idx
        ]

        print(
            "Source Text (Input)        : "
            + " ".join(src_tokens).replace("\n", "")
        )
        print(
            "Target Text (Ground Truth) : "
            + " ".join(tgt_tokens).replace("\n", "")
        )
        model_out = greedy_decode(model, rb.src, rb.src_mask, 72, 0)[0]
        model_txt = (
            " ".join(
                [vocab_tgt.get_itos()[x] for x in model_out if x != pad_idx]
            ).split(eos_string, 1)[0]
            + eos_string
        )
        print("Model Output               : " + model_txt.replace("\n", ""))
        results[idx] = (rb, src_tokens, tgt_tokens, model_out, model_txt)
    return results



# execute_example(run_model_example)


# %% [markdown] id="0ZkkNTKLTsqO"
# ## Attention Visualization
#
# > Even with a greedy decoders the translation looks pretty good. We
# > can further visualize it to see what is happening at each layer of
# > the attention

# %%
def mtx2df(m, max_row, max_col, row_tokens, col_tokens):
    "convert a dense matrix to a data frame with row and column indices"
    return pd.DataFrame(
        [
            (
                r,
                c,
                float(m[r, c]),
                "%.3d %s"
                % (r, row_tokens[r] if len(row_tokens) > r else "<blank>"),
                "%.3d %s"
                % (c, col_tokens[c] if len(col_tokens) > c else "<blank>"),
            )
            for r in range(m.shape[0])
            for c in range(m.shape[1])
            if r < max_row and c < max_col
        ],
        # if float(m[r,c]) != 0 and r < max_row and c < max_col],
        columns=["row", "column", "value", "row_token", "col_token"],
    )


def attn_map(attn, layer, head, row_tokens, col_tokens, max_dim=30):
    df = mtx2df(
        attn[0, head].data,
        max_dim,
        max_dim,
        row_tokens,
        col_tokens,
    )
    return (
        alt.Chart(data=df)
        .mark_rect()
        .encode(
            x=alt.X("col_token", axis=alt.Axis(title="")),
            y=alt.Y("row_token", axis=alt.Axis(title="")),
            color="value",
            tooltip=["row", "column", "value", "row_token", "col_token"],
        )
        .properties(height=400, width=400)
        .interactive()
    )


# %% tags=[]
def get_encoder(model, layer):
    return model.encoder.layers[layer].self_attn.attn


def get_decoder_self(model, layer):
    return model.decoder.layers[layer].self_attn.attn


def get_decoder_src(model, layer):
    return model.decoder.layers[layer].src_attn.attn


def visualize_layer(model, layer, getter_fn, ntokens, row_tokens, col_tokens):
    # ntokens = last_example[0].ntokens
    attn = getter_fn(model, layer)
    n_heads = attn.shape[1]
    charts = [
        attn_map(
            attn,
            0,
            h,
            row_tokens=row_tokens,
            col_tokens=col_tokens,
            max_dim=ntokens,
        )
        for h in range(n_heads)
    ]
    assert n_heads == 8
    return alt.vconcat(
        charts[0]
        # | charts[1]
        | charts[2]
        # | charts[3]
        | charts[4]
        # | charts[5]
        | charts[6]
        # | charts[7]
        # layer + 1 due to 0-indexing
    ).properties(title="Layer %d" % (layer + 1))

