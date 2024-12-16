import datasets
import torch
import pickle
from make_datasets import load_tokenizer
from transformers import AutoTokenizer
import pandas as pd


def test1():
    dataset = datasets.load_dataset("/mnt/nfs/algo/intern/haoyunx9/data/NLI/xnli/xnli/zh")
    train_set = dataset['train']

    print(dataset)

    print(train_set[:10])

def test2():
    data = datasets.load_from_disk(r'/mnt/nfs/algo/intern/haoyunx11_/idea/MI/sn/data/xnli/es/simple_prompt')
    print(data)

    # data = data.select(range(10))
    # print(data)
    idx = 0

    print(data['content'][idx])
    print(data['input_ids'][idx])
    print(len(data['input_ids'][idx]))

    tokenizer = load_tokenizer(model_path=r'/mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf')
    tokenized_text = tokenizer.decode(data['input_ids'][idx])
    print(tokenized_text)

    print(data['mask'][idx])
    print(len(data['mask'][idx]))

    print(data['label'][idx])


def test2_2():
    data = datasets.load_from_disk(r'/mnt/nfs/algo/intern/haoyunx11_/idea/MI/sn/data/xnli/en/standard_prompt')
    print(data)
    idx = 0
    # data = data.select(range(10))
    # print(data)
    print(data['content'][idx])
    print(data['input_ids'][idx])

    tokenizer = load_tokenizer(model_path=r'/mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf')
    tokenized_text = tokenizer.decode(data['input_ids'][idx])
    print(tokenized_text)

    print(data['mask'][idx])

    print(len(data['mask'][idx]))

    print(data['label'][idx])


def test3():
    save_path = r'/mnt/nfs/algo/intern/haoyunx11_/idea/MI/sn/activations/Llama-2-7b-chat-hf/xnli/en/xnli.en.mean.simple_prompt.3.pt'
    activations = torch.load(save_path)
    print(activations.shape)
    print(activations)


def test4():
    activations = torch.load(r"/mnt/nfs/algo/intern/haoyunx11_/idea/MI/sn/activations/Llama-2-7b-chat-hf/xnli/en/xnli.en.mean.simple_prompt.0.pt").dequantize()
    print(activations.shape)

def test5():
    probe_result = pickle.load(open(
        r"/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/Llama-2-7b-chat-hf/xnli/en/xnli.en.mean.simple_prompt_pca_layer12.p",
        'rb'))

    print(probe_result)

    print('******************')
    # coef_
    t = torch.tensor(probe_result['probe_directions'][2048][:, 1])

    print(t)
    print(t.shape)


def test6():
    data = datasets.load_from_disk(r'/data5/haoyun.xu/study/MI/world-models/data/prompt_datasets/Llama-2/world_place/random')
    print(data)

    print(data['entity'][0])
    print(data['input_ids'][0])

    tokenizer = load_tokenizer(model_path=r'/mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf')
    tokenized_text = tokenizer.decode(data['input_ids'][0])
    print(tokenized_text)



    print(data['entity_mask'][0])


def test7():
    temp_mask = []
    temp_mask.append(False)
    temp_mask = temp_mask + [True] * 10

    print(temp_mask)


def test8():
    dataset = datasets.load_dataset("/mnt/nfs/algo/intern/haoyunx9/data/NLI/xnli/xnli/en")
    train_set = dataset['train']

    print(train_set[:10]['premise'])

    tokenizer = AutoTokenizer.from_pretrained("/mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf")
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token

    token_ids = tokenizer(train_set)

    random_prompts = torch.randint(
        low=100, high=token_ids.max().item(),
        size=(token_ids.shape[0], 10),
        dtype=torch.long
    )
    token_ids = torch.cat([random_prompts, token_ids], dim=1)
    print(token_ids)


#
#     # add bos token
#
#
# token_ids = torch.cat([
#     torch.ones(token_ids.shape[0], 1,
#                dtype=torch.long) * tokenizer.bos_token_id,
#     token_ids], dim=1
# )
#
#     print(dataset)
#
#     print(train_set[:10])



def test9():
    feature = 'in'

    t = pd.Series([feature] * 6)
    print(t)





if __name__ == '__main__':
    # test1()
    # test2()
    # test2_2()
    test3()
    # test4()
    # test5()
    # test6()
    # test7()
    # test8()
    # test9()