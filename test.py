import os
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

RUN_EXAMPLES = True

# %%
# Some convenience helper functions used throughout the notebook
from my_transformer import *


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
        )[0]
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


def run_model_example(n_examples=5):
    global vocab_src, vocab_tgt, spacy_de, spacy_en

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
    model = make_model(len(vocab_src), len(vocab_tgt), layer_num=6)
    model.load_state_dict(
        torch.load("multi30k_model_final.pt", map_location=torch.device("cpu"))
    )

    print("Checking Model Outputs:")
    example_data = check_outputs(
        valid_dataloader, model, vocab_src, vocab_tgt, n_examples=n_examples
    )
    return model, example_data


# %% tags=[]
def viz_encoder_self():
    model, example_data = run_model_example(n_examples=1)
    example = example_data[
        len(example_data) - 1
        ]  # batch object for the final example

    layer_viz = [
        visualize_layer(
            model, layer, get_encoder, len(example[1]), example[1], example[1]
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


# %% tags=[]
def viz_decoder_self():
    model, example_data = run_model_example(n_examples=1)
    example = example_data[len(example_data) - 1]

    layer_viz = [
        visualize_layer(
            model,
            layer,
            get_decoder_self,
            len(example[1]),
            example[1],
            example[1],
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

def viz_decoder_src():
    model, example_data = run_model_example(n_examples=1)
    example = example_data[len(example_data) - 1]

    layer_viz = [
        visualize_layer(
            model,
            layer,
            get_decoder_src,
            max(len(example[1]), len(example[2])),
            example[1],
            example[2],
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

if __name__ == '__main__':
    load_or_train_model()

    if False:
        run_de_translate_to_en()
        load_tokenizers()
        example_simple_copy_model()
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
