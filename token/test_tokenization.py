# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
import tokenization
import six
import tensorflow as tf
#from tensorflow.python.framework.test_util import assertAllEqual


def test_chinese():
    tokenizer = tokenization.BasicTokenizer()
    tokens = tokenizer.tokenize("你好，我是一名开发人员")
    # ['你', '好', '，', '我', '是', '一', '名', '开', '发', '人', '员']
    print("tokens:", tokens)
    tf.assert_equal(tokenizer.tokenize(u"ah\u535A\u63A8zz"),  # ah博雅zz
                        [u"ah", u"\u535A", u"\u63A8", u"zz"])

def test_full_tokenizer():
    vocab_tokens = [
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "want",
        "##want",
        "##ed",
        "wa",
        "un",
        "runn",
        "##ing",
        ","
    ]

    with tempfile.NamedTemporaryFile(delete=False) as vocab_writer:
        if six.PY2:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))
        else:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]).encode("utf-8"))

        vocab_file = vocab_writer.name

    tokenizer = tokenization.FullTokenizer(vocab_file)
    os.unlink(vocab_file) # 方法用于删除文件，如果文件是一个目录则返回一个错误

    tokens = tokenizer.tokenize(u"UNwant\u00E9d,running")
    print("tokens:", tokens)
    tf.assert_equal(tokens, ["un", "##want", "##ed", ",", "runn", "##ing"])
    tf.assert_equal(tokenizer.convert_tokens_to_ids(tokens), [7, 4, 5, 10, 8, 9])

def test_basic_tokenizer_lower():
    tokenizer = tokenization.BasicTokenizer(do_lower_case=True)

    tf.assert_equal(tokenizer.tokenize(u" \tHeLLo!how  \n Are yoU?  "),
        ["hello", "!", "how", "are", "you", "?"])
    tf.assert_equal(tokenizer.tokenize(u"H\u00E9llo"), ["hello"])


def test_basic_tokenizer_no_lower():
    tokenizer = tokenization.BasicTokenizer(do_lower_case=False)

    tf.assert_equal(tokenizer.tokenize(u" \tHeLLo!how  \n Are yoU?  "),
        ["HeLLo", "!", "how", "Are", "yoU", "?"])


def test_wordpiece_tokenizer():
    vocab_tokens = [
        "[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn",
        "##ing"
    ]

    vocab = {}
    for (i, token) in enumerate(vocab_tokens):
        vocab[token] = i
    tokenizer = tokenization.WordpieceTokenizer(vocab=vocab)

    tf.assert_equal(tokenizer.tokenize(""), [])

    tf.assert_equal(tokenizer.tokenize("unwanted running"),
                        ["un", "##want", "##ed", "runn", "##ing"])

    tf.assert_equal(tokenizer.tokenize("unwantedX running"), ["[UNK]", "runn", "##ing"])


def test_convert_tokens_to_ids():
    vocab_tokens = [
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "want",
        "##want",
        "##ed",
        "wa",
        "un",
        "runn",
        "##ing"
    ]

    vocab = {}
    for (i, token) in enumerate(vocab_tokens):
        vocab[token] = i

    tf.assert_equal(
        tokenization.convert_tokens_to_ids(vocab, ["un", "##want", "##ed", "runn", "##ing"]), [7, 4, 5, 8, 9])

def test_is_whitespace():
    assert (tokenization._is_whitespace(u" "))
    assert (tokenization._is_whitespace(u"\t"))
    assert (tokenization._is_whitespace(u"\r"))
    assert (tokenization._is_whitespace(u"\n"))
    assert (tokenization._is_whitespace(u"\u00A0"))

    assert not (tokenization._is_whitespace(u"A"))
    assert not (tokenization._is_whitespace(u"-"))


def test_is_control():
    assert (tokenization._is_control(u"\u0005"))
    assert not (tokenization._is_control(u"A"))
    assert not(tokenization._is_control(u" "))
    assert not(tokenization._is_control(u"\t"))
    assert not(tokenization._is_control(u"\r"))
    assert not(tokenization._is_control(u"\U0001F4A9"))


def test_is_punctuation():
    assert (tokenization._is_punctuation(u"-"))
    assert (tokenization._is_punctuation(u"$"))
    assert (tokenization._is_punctuation(u"`"))
    assert (tokenization._is_punctuation(u"."))

    assert not(tokenization._is_punctuation(u"A"))
    assert not(tokenization._is_punctuation(u" "))


if __name__ == "__main__":
    test_chinese()
    test_full_tokenizer()
    test_basic_tokenizer_lower()
    test_basic_tokenizer_no_lower()
    test_wordpiece_tokenizer()
    test_convert_tokens_to_ids()
    test_is_whitespace()
    test_is_control()
    test_is_punctuation()