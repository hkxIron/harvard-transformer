#from transformers import BertTokenizer, BartForConditionalGeneration, Text2TextGenerationPipeline

"""
https://arxiv.org/abs/1910.13461
BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension
https://huggingface.co/fnlp/bart-base-chinese
"""
# tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
# model = BartForConditionalGeneration.from_pretrained("fnlp/bart-base-chinese")
# text2text_generator = Text2TextGenerationPipeline(model, tokenizer)

def test_transformer():
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    inputs = tokenizer("Hello world!", return_tensors="pt")
    outputs = model(**inputs)
    print(outputs)

# 白化操作：
# x_hat = (x-mu)*W
# 作者：JMXGODLZ
# 链接：https://www.zhihu.com/question/460991118/answer/2746361651
import numpy as np
def compute_kernel_bias(vecs):
    """计算kernel和bias
    vecs.shape = [num_samples, embedding_size]，
    最后的变换：y = (x + bias).dot(kernel)
    """
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T) # 计算x的协方差, [embed_size, embed_size]
    # u:[embed_size, embed_size]
    # s: [embed_size]
    # vh:[embed_size, embed_size]
    u, s, vh = np.linalg.svd(cov)
    # W:[embed_size, embed_size]
    # s:[embed_size]
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    return W, -mu

if __name__ == '__main__':
    data = np.random.randn(10, 3)
    compute_kernel_bias(data)
