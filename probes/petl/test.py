import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from transformers import LlamaTokenizer
from transformers.models.llama.modeling_llama import LlamaMLP
from modeling import LlamaModel

text = 'hello world'
tokenizer = LlamaTokenizer.from_pretrained(r'/mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf')
tokenied = tokenizer(text, return_tensors='pt')
model = LlamaModel.from_pretrained(r'/mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf')


model.to("cuda")
tokenied.to("cuda")
model.train()

# for k, v in model.named_parameters():
#     if "up_proj" in k:
#         v.requires_grad_(False)
#         print(k)
#         print(v.shape)
#
#         print(v[0].shape)
#         v[0].requires_grad_(True)
#         print(v[0].requires_grad)
#         print(v[1].requires_grad)
#         print(v)
#         print('+++++++++++++++++++++++++++++++++++++++')
#
#
#     else:
#         v.requires_grad = False
# for k, v in model.named_parameters():
#     print(k)
#     print(v)
#     break

for name, module in model.named_modules():
    names = name.split('.')
    if len(names) == 1:
        print(names[0])
    else:
        print(names[-1])



out = model(**tokenied, output_hidden_states=True, output_attentions=False, return_dict=True, use_cache=False)
print(out.shape)
