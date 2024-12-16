import jieba

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoConfig, BertTokenizer

tokenizer = BertTokenizer.from_pretrained("/mnt/nfs/algo/intern/haoyunx11/models/model/alignment/model_without_co")
src = "你好美丽的世界！"

tokenizer.basic_tokenizer.tokenize_chinese_chars = True
tokenized = tokenizer.basic_tokenizer.tokenize(src)
print(tokenized)
tokenized = jieba.lcut(src)
print(tokenized)
for t in tokenized:
    print(t)



