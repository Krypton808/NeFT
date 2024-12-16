import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import gc
import einops
import torch
import argparse
import numpy as np
import pandas as pd
import datasets
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import peft_model
from transformer_lens import HookedTransformer

def test1():
    path = r'/mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf'
    tokenizer = AutoTokenizer.from_pretrained(path)
    hf_model = AutoModelForCausalLM.from_pretrained(path)
    model = HookedTransformer.from_pretrained(r'Llama-2-7b-chat-hf', hf_model=hf_model, device="cpu",
                                              fold_ln=False,
                                              center_writing_weights=False, center_unembed=True,
                                              tokenizer=tokenizer)
    print(model.W_in[0])
    print(model.W_in.shape)


if __name__ == '__main__':
    test1()




