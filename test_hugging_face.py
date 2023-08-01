#from transformers import BertTokenizer, BartForConditionalGeneration, Text2TextGenerationPipeline

"""
https://arxiv.org/abs/1910.13461
BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension
https://huggingface.co/fnlp/bart-base-chinese
"""
# tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
# model = BartForConditionalGeneration.from_pretrained("fnlp/bart-base-chinese")
# text2text_generator = Text2TextGenerationPipeline(model, tokenizer)

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
inputs = tokenizer("Hello world!", return_tensors="pt")
outputs = model(**inputs)
print(outputs)
