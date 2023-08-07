import os
from typing import Optional,Union,Any,List,Tuple,Callable
from os.path import exists
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import altair as alt
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
from my_transformer import *

RUN_EXAMPLES = True

# %%
# Some convenience helper functions used throughout the notebook


def is_interactive_notebook():
    return __name__ == "__main__"


def show_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        return fn(*args)


def execute_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        fn(*args)


PAD = 0
EOS = 1
def data_gen(vocab_size:int, batch_size:int, nbatches:int):
    "Generate random data for a src-tgt copy task."
    seq_len = 10
    for i in range(nbatches):
        data = torch.randint(low=1, high=vocab_size, size=(batch_size, seq_len)) # 生成的值在[1, v-1]之间
        data[:, 0] = EOS # 第0位为EOS=1, PAD = 0
        # copy-task,注意，此处输入与输出一模一样
        # src:[batch,seq_len]
        src = data.requires_grad_(False).clone().detach() # detach就是不需要梯度
        # tgt:[batch,seq_len], 注意：此处src,tgt数据相同
        tgt = data.requires_grad_(False).clone().detach()
        yield Batch(src, tgt, pad=PAD)

def example_mask():
    LS_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Subsequent Mask": subsequent_mask(20)[0][x, y].flatten(),
                    "Window": y,
                    "Masking": x,
                }
            )
            for y in range(20)
            for x in range(20)
        ]
    )
    print(LS_data.shape)
    print(LS_data.head())

    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5,5))
    plt.imshow(subsequent_mask(20)[0])
    
    user guide: 
    https://altair-viz.github.io/user_guide/generated/channels/altair.Color.html#altair.Color
    """
    chart = alt.Chart(LS_data).mark_rect().properties(height=250, width=250).encode(
        alt.X("Window:O"),
        alt.Y("Masking:O"),
        alt.Color("Subsequent Mask:Q", scale=alt.Scale(scheme="viridis")),
    ).interactive()

    chart.save("data/mask.html")
    return chart

def example_positional():
    pe = PositionalEncoding(d_model=20, dropout=0)
    # x:[batch, seq_len, d_model]
    # y:[batch, seq_len, d_model]
    y = pe.forward(x=torch.zeros(1, 100, 20))

    data = pd.concat(
        [
            pd.DataFrame(
                {
                    "embedding": y[0, :, dim],
                    "dimension": dim,
                    "position": list(range(100)),
                }
            )
            for dim in [4, 5, 6, 7]
        ]
    )

    chart = alt.Chart(data)\
        .mark_line()\
        .properties(width=800)\
        .encode(x="position", y="embedding", color="dimension:N")\
        .interactive()

    chart.save("data/position.html")
    return chart


def inference_test():
    test_model = make_model(src_vocab_size=11, tgt_vocab_size=11, layer_num=2)
    test_model.eval() # test mode
    # src:[batch, seq_len]
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    # src_mask:[batch, 1, seq_len]
    src_mask = torch.ones(1, 1, 10)

    # memory:[batch, seq_len, d_model]
    memory = test_model.encode(src, src_mask)
    # ys:[batch=1, seq_len=1]
    ys = torch.zeros(1, 1).type_as(src)

    for i in range(9):
        # out:[batch, seq_len-1, d_model]
        out = test_model.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data))
        # out[:,-1]等价于[out,-1,:]=>取每个batch最后一个seq的所有d_model，即[batch,d_model]
        # prob:[batch, vocab]
        prob = test_model.generator(out[:, -1])
        # next_word:[batch=1]
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        # ys = [batch=1, seq+1]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    print("Example Untrained Model Prediction:", ys)


def run_tests():
    for _ in range(10):
        inference_test()


def example_learning_schedule():
    opts = [
        # dmodel, factor, warm_up
        [512, 1, 4000],  # example 1
        [512, 1, 8000],  # example 2
        [256, 1, 4000],  # example 3
    ]

    dummy_model = torch.nn.Linear(in_features=1, out_features=1)
    learning_rates = []

    # we have 3 examples in opts list.
    for idx, example in enumerate(opts):
        # run 20000 epoch for each example
        optimizer = torch.optim.Adam(dummy_model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9)
        lr_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lambda step: rate(step, *example)) # 可以动态设置lr
        tmp = []
        # take 20K dummy training steps, save the learning rate at each step
        for step in range(20000):
            tmp.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            lr_scheduler.step()
        learning_rates.append(tmp)

    learning_rates = torch.tensor(learning_rates)

    # Enable altair to handle more than 5000 rows
    alt.data_transformers.disable_max_rows()

    opts_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Learning Rate": learning_rates[warmup_idx, :],
                    "model_size:warmup": ["512:4000", "512:8000", "256:4000"][ warmup_idx ],
                    "step": range(20000),
                }
            )
            for warmup_idx in [0, 1, 2]
        ]
    )

    chart = alt.Chart(opts_data)\
        .mark_line()\
        .properties(width=600)\
        .encode(x="step", y="Learning Rate", color="model_size:warmup:N")\
        .interactive()

    chart.save("data/lr_examples.html")
    return chart

def example_label_smoothing():
    crit = LabelSmoothing(vocab_size=5, padding_idx=0, smoothing=0.4)
    predict = torch.FloatTensor(
        [
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
        ]
    )

    # pred_scores:[batch*(seq_len-1), vocab]
    # target_ids: [batch*(seq_len-1)]
    crit.forward(pred_scores=predict.log(), target_ids=torch.LongTensor([2, 1, 0, 3, 3]))

    LS_data:pd.DataFrame = pd.concat(
        [
            pd.DataFrame(
                {
                    # smoothed_target_dist:[batch*(seq_len-1), vocab]
                    "target distribution": crit.smoothed_target_dist[x, y].flatten(),
                    "columns": y,
                    "rows": x,
                }
            )
            for y in range(5)
            for x in range(5)
        ]
    )

    chart=alt.Chart(LS_data)\
        .mark_rect(color="Blue", opacity=1)\
        .properties(height=200, width=200)\
        .encode(
            alt.X("columns:O", title=None),
            alt.Y("rows:O", title=None),
            alt.Color(
                "target distribution:Q", scale=alt.Scale(scheme="viridis")
            ),
        ).interactive()

    chart.save("data/label_smoothing.html")
    return chart

"""
标签平滑实际上是模型在惩罚过于自信的预测值

"""
def loss(count:int, crit:LabelSmoothing):
    smooth_count = 1
    vocab_size = 5
    sum_count = count + (vocab_size - 2) * smooth_count
    jitter_p = smooth_count/sum_count
    smoothed_probs = torch.FloatTensor([[0, count / sum_count, jitter_p, jitter_p, jitter_p]])
    log_smoothed_probs = smoothed_probs.log()
    log_smoothed_probs[0,0] = 0 # 将-inf置为0
    loss = crit.forward(log_smoothed_probs, torch.LongTensor([1])).data
    return loss

def penalization_visualization():
    crit = LabelSmoothing(vocab_size=5, padding_idx=0, smoothing=0.1)
    loss_data = pd.DataFrame(
        {
            "Loss": [loss(x, crit) for x in range(1, 100)],
            "Steps": list(range(99)),
        }
    ).astype("float")

    chart = (
        alt.Chart(loss_data)
            .mark_line()
            .properties(width=350)
            .encode(x="Steps",y="Loss",)
            .interactive()
    )
    chart.save("data/label_smoothing_loss.html")
    return chart


# Train the simple copy task. We can begin by trying out a simple copy-task. Given a random set of input symbols from a
# small vocabulary, the goal is to generate back those same symbols.
def example_simple_copy_model():
    vocab_size = 10 + 1
    criterion = LabelSmoothing(vocab_size=vocab_size, padding_idx=0, smoothing=0.0)
    model = make_model(src_vocab_size=vocab_size, tgt_vocab_size=vocab_size, layer_num=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9)
    # 用来自定义学习率
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(step, model_size=model.src_embed.vocab_embedding.d_model, factor=1.0, warmup=400),)

    batch_size = 80
    epoch_num = 20
    #epoch_num = 5
    for epoch in range(epoch_num):
        # Sets the module in training mode.其中dropout有影响,在train时需要1/p
        model.train()
        # train
        train_loss,_ = run_epoch(
            data_gen(vocab_size, batch_size, nbatches=20),
            model,
            SimpleLossCompute(model.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train",
        )
        print("train loss:", train_loss)

        # eval
        model.eval() # set test mode
        eval_loss,_ = run_epoch(
            data_gen(vocab_size, batch_size, nbatches=5),
            model,
            SimpleLossCompute(model.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )
        print("eval loss:", eval_loss)

    model.eval() # set test mode, 展示部分例子
    # src:[batch, seq_len]
    #src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    src = torch.LongTensor([[1, 1, 2, 2, 3, 0, 4, 5, 6, 7, 8, 9]])
    max_len = src.shape[1] # seq_len=10
    # src_mask:[batch,1,seq_len]
    src_mask = torch.ones(1, 1, max_len)
    print("input:", src)
    # 输出：tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    # 输出：tensor([[0, 2, 3, 4, 5, 6, 5, 7, 8, 8]]), 期望输出与输入src一模一样
    pred=greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0)
    print("greedy_decode:", pred)
    print("acc:", (src==pred).sum()/np.prod(src.shape))

def run_model_example(n_examples=5)->Tuple[EncoderDecoder, List[Tuple[Batch, List[str], List[str], Tensor, str]]]:
    #global vocab_src, vocab_tgt, spacy_de, spacy_en
    spacy_de, spacy_en = load_tokenizers()
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)

    print("Preparing Data ...")
    _, valid_dataloader = create_dataloaders(
        torch.device("cpu"),
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=1,
        is_distributed=False,
    )

    print("Loading Trained Model ...")
    # 加载模型进行预测
    #model = make_model(len(vocab_src), len(vocab_tgt), layer_num=6)
    config = get_model_config()
    model = make_model(src_vocab_size=len(vocab_src), tgt_vocab_size=len(vocab_tgt),
                       layer_num=config['layer_num'], d_model=config['d_model'], d_ff=config['d_ff'], head_num=config['head_num'])
    model.load_state_dict(torch.load("multi30k_model_final.pt", map_location=torch.device("cpu")) )

    print("Checking Model Outputs:")
    example_data = check_outputs(valid_dataloader, model, vocab_src, vocab_tgt, n_examples=n_examples)
    return model, example_data


def viz_encoder_self():
    """
    # example_data: List[(batch_ids, src_tokens, tgt_tokens, model_out, model_txt)]
    """
    model, example_data = run_model_example(n_examples=1)
    example = example_data[len(example_data) - 1]  # batch object for the final example

    layer_viz = [
        visualize_layer(
            model=model,
            layer=layer,
            get_attn_map_fn=get_encoder,
            ntokens=len(example[1]),
            row_tokens=example[1], # self-attention中，token都来源于src_tokens
            col_tokens=example[1]
        )
        for layer in range(6)
    ]
    return alt.hconcat(
        layer_viz[0]
        # & layer_viz[1]
        & layer_viz[2]
        # & layer_viz[3]
        & layer_viz[4]
        # & layer_viz[5]
    )


def viz_decoder_self():
    """
    # example_data: List[(batch_ids, src_tokens, tgt_tokens, model_out, model_txt)]
    """
    model, example_data = run_model_example(n_examples=1)
    example = example_data[len(example_data) - 1]

    layer_viz = [
        visualize_layer(
            model,
            layer,
            get_decoder_self,
            len(example[1]),
            row_tokens=example[1], # decoder-self-attention中，token都来源于src_tokens
            col_tokens=example[1],
        )
        for layer in range(6)
    ]
    return alt.hconcat(
        layer_viz[0]
        & layer_viz[1]
        & layer_viz[2]
        & layer_viz[3]
        & layer_viz[4]
        & layer_viz[5]
    )

def viz_decoder_src(): # decoder-encoder-cross-attention
    """
    # example_data: List[(batch_ids, src_tokens, tgt_tokens, model_out, model_txt)]
    """
    model, example_data = run_model_example(n_examples=1)
    example = example_data[len(example_data) - 1]

    layer_viz = [
        visualize_layer(
            model,
            layer,
            get_decoder_src,
            max(len(example[1]), len(example[2])),
            row_tokens=example[1],# decoder-cross-attention中，row_tokens来源于src_tokens
            col_tokens=example[2], # decoder-cross-attention中，col_tokens来源于tgt_tokens
        )
        for layer in range(6)
    ]
    return alt.hconcat(
        layer_viz[0]
        & layer_viz[1]
        & layer_viz[2]
        & layer_viz[3]
        & layer_viz[4]
        & layer_viz[5]
    )


def run_de_translate_to_en():
    # global variables used later in the script
    spacy_de, spacy_en = show_example(load_tokenizers)
    vocab_src, vocab_tgt = show_example(load_vocab, args=[spacy_de, spacy_en])

def test_loss():
    crit = LabelSmoothing(vocab_size=5, padding_idx=0, smoothing=0.1)
    data = torch.cat([loss(x, crit).reshape(1,1) for x in range(1, 100)], dim=1)
    print(data)

def get_model_config():
    # 原始论文配置
    # "batch_size": 32, # 原始论文设置，内存超限
    # "num_epochs": 8,
    # d_model = 512
    # d_ff = 2048
    # head_num = 8
    # layer_num = 6
    config = {
        "batch_size": 4,
        "num_epochs": 8,
        "d_model":512,
        "d_ff":2048,
        "head_num":8,
        "layer_num":6,
        "distributed": False,
        "accum_iter": 10,
        "base_lr": 1.0,
        "max_padding": 72,
        "warmup": 3000,
        "file_prefix": "multi30k_model_",
    }
    return config

def load_or_train_model():
    print("begin to load or train model")
    spacy_de, spacy_en = load_tokenizers()
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)

    config = get_model_config()
    model_path = "multi30k_model_final.pt"
    if not exists(model_path):
        print(f"not exists train model:{model_path}, begin to train")
        train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config)
    else:
        print(f"load model from:{model_path}")

    model = make_model(src_vocab_size=len(vocab_src), tgt_vocab_size=len(vocab_tgt),
                       layer_num=config['layer_num'], d_model=config['d_model'], d_ff=config['d_ff'], head_num=config['head_num'])
    # 为model加载参数
    model.load_state_dict(torch.load(model_path))
    return model

"""
Shared Embeddings
When using BPE with shared vocabulary we can share the same
weight vectors between the source / target / generator. See the (cite) (cite) (cite) (cite) for details. To
add this to the model simply do this:
"""
# if False:
#     model.src_embed[0].lut.weight = model.tgt_embeddings[0].lut.weight # lut:lookup_table
#     model.generator.lut.weight = model.tgt_embed[0].lut.weight

"""
The paper averages the last k checkpoints to create an ensembling
effect. We can do this after the fact if we have a bunch of models
"""
def average(model, models):
    "Average models into model"
    for ps in zip(*[m.params() for m in [model] + models]):
        ps[0].copy_(src=torch.sum(*ps[1:]) / len(ps[1:]))

# Load data and model for output checks
def check_outputs(
        valid_dataloader,
        model:EncoderDecoder,
        vocab_src:Vocab,
        vocab_tgt:Vocab,
        n_examples=15,
        pad_idx=2,
        eos_string="</s>",
)->List[Tuple[Batch, List[str], List[str], Tensor, str]]:
    results = [()] * n_examples
    for idx in range(n_examples):
        print("\nExample %d ========\n" % idx)
        # dataloader:返回(src_ids, tgt_ids)
        b = next(iter(valid_dataloader))
        rb = Batch(src=b[0], tgt=b[1], pad=pad_idx)
        ys = greedy_decode(model, rb.src, rb.src_mask, max_len=64, start_symbol=0)[0]

        src_tokens = [vocab_src.get_itos()[x] for x in rb.src[0] if x != pad_idx ] # int -> str
        tgt_tokens = [vocab_tgt.get_itos()[x] for x in rb.tgt[0] if x != pad_idx ]

        print( "Source Text (Input)        : " + " ".join(src_tokens).replace("\n", "") )
        print( "Target Text (Ground Truth) : " + " ".join(tgt_tokens).replace("\n", "") )

        model_out = greedy_decode(model, rb.src, rb.src_mask, max_len=72, start_symbol=0)[0]
        model_txt = (
                " ".join(
                    [vocab_tgt.get_itos()[x] for x in model_out if x != pad_idx]
                ).split(eos_string, maxsplit=1)[0]
                + eos_string
        )
        print("Model Output               : " + model_txt.replace("\n", ""))
        results[idx] = (rb, src_tokens, tgt_tokens, model_out, model_txt)

    return results


# execute_example(run_model_example)
# ## Attention Visualization
#
# > Even with a greedy decoders the translation looks pretty good. We
# > can further visualize it to see what is happening at each layer of
# > the attention

def matrix_to_dataframe(m:Tensor, max_row:int, max_col:int, row_tokens:List[str], col_tokens:List[str]):
    "convert a dense matrix to a data frame with row and column indices"
    return pd.DataFrame(
        [
            (
                r,
                c,
                float(m[r, c]),
                "%.3d %s"
                % (r, row_tokens[r] if len(row_tokens) > r else "<blank>"), # <blank>是padding字符
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


# attn: [batch, head, seq_len, seq_len]
def attn_map(attn:Tensor, head:int, row_tokens:List[str], col_tokens:List[str], max_dim=30):
    batch = 0
    df = matrix_to_dataframe(
        attn[batch, head].data,
        max_dim,
        max_dim,
        row_tokens,
        col_tokens,
    )
    chart = (
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
    chart.save("images/attn_map.html")
    return chart

# attn: [batch, head, seq_len, seq_len]
def get_encoder(model:EncoderDecoder, layer:int)->Tensor:
    return model.encoders.layers[layer].self_attn.attn


# attn: [batch, head, seq_len, seq_len]
def get_decoder_self(model:EncoderDecoder, layer:int)->Tensor:
    return model.decoders.layers[layer].self_attn.attn

# attn: [batch, head, seq_len, seq_len]
def get_decoder_src(model:EncoderDecoder, layer:int)->Tensor:
    return model.decoders.layers[layer].src_attn.attn # cross-attention

def visualize_layer(model:EncoderDecoder,
                    layer:int,
                    get_attn_map_fn:Callable[[EncoderLayer, int], Tensor],
                    ntokens:int,
                    row_tokens:List[str],
                    col_tokens:List[str]):
    # ntokens = last_example[0].ntokens
    # attn: [batch, head, seq_len, seq_len]
    attn = get_attn_map_fn(model, layer)
    n_heads = attn.shape[1]
    charts = [
        attn_map(
            attn,
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

def test_norm():
    x = np.random.randn(2, 3)
    print(x.shape)
    t = x.ravel()
    print(t.shape)
    print(t)
    print(np.linalg.norm(t)) #F-范式，即为2范式
    print(np.sqrt((t*t).sum())) # 与上面的2范式结果相同

def test_layer_norm():
    batch=4
    seq_len=3
    model_size=2
    x = torch.Tensor(np.random.randn(batch, seq_len, model_size))
    norm = LayerNorm(features=model_size)
    y = norm(x)
    print(x)
    print(y)
    print(x.shape)
    print(y.shape)


if __name__ == '__main__':
    test_layer_norm()
    #load_or_train_model()
    #example_simple_copy_model()

    if False:
        test_norm()
        load_or_train_model()
        viz_encoder_self()
        run_model_example()
        run_de_translate_to_en()
        load_tokenizers()
        #execute_example(example_simple_copy_model)
        test_loss()
        example_learning_schedule()
        run_tests()
        inference_test()
        example_mask()
        run_model_example()
        # model = load_trained_model()
        # run_de_translate_to_en()
        # show_example(viz_decoder_src)
        # show_example(run_tests)
        # show_example(penalization_visualization)
        # show_example(example_label_smoothing)
        # show_example(example_positional)
