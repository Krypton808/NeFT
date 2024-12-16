# 在 make_prompt_datasets.py 的基础上修改的

import datasets
import torch
from transformers import AutoTokenizer
from feature_datasets import xnli

# torch.random.manual_seed(1)

PROMPTS = {
    "xnli": xnli.xnli_PROMPTS
}

"""
train: Dataset({
        features: ['premise', 'hypothesis', 'label'],
        num_rows: 392702
    })
"""


def get_idx_list(language='en'):
    dataset = datasets.load_dataset("/mnt/nfs/algo/intern/haoyunx9/data/NLI/xnli/xnli/" + language)['train']
    # dataset = dataset[:amount+1]

    max_len = 0
    count = 0
    data_list = []
    idx_list = []
    content_list = []
    label_list = []
    for idx, data in enumerate(zip(dataset['premise'], dataset['hypothesis'], dataset['label'])):
        for prompt_dict in PROMPTS['xnli'].items():

            if prompt_dict[0] == 'simple_prompt':
                prompt_text = prompt_dict[1].format_map({"sentence1": data[0], "sentence2": data[1]})

                if len(prompt_text) > 150:
                    continue

                idx_list.append(idx)

    return idx_list


def save_xnli_tokenized_datasets(tokenizer, language='ru', amount=10, prompt_type="random_prompt"):
    dataset = datasets.load_dataset("/mnt/nfs/algo/intern/haoyunx9/data/NLI/xnli/xnli/" + language)['train']
    # dataset = dataset[:amount+1]

    max_len = 0
    count = 0
    data_list = []
    # idx_list = []
    content_list = []
    label_list = []

    idx_list = get_idx_list()

    for idx, data in enumerate(zip(dataset['premise'], dataset['hypothesis'], dataset['label'])):
        if idx not in idx_list:
            continue

        for prompt_dict in PROMPTS['xnli'].items():

            if prompt_dict[0] == prompt_type:
                prompt_text = prompt_dict[1].format_map({"sentence1": data[0], "sentence2": data[1]})

                # if len(prompt_text) > 150:
                #     continue

                # count += 1
                # if count == 132658:
                #     print(prompt_text)
                #     print(len(prompt_text))
                # idx_list.append(idx)
                data_list.append(prompt_text)
                label_list.append(data[2])

                temp_dict = {}
                temp_dict['sentence1'] = data[0]
                temp_dict['sentence2'] = data[1]
                content_list.append(temp_dict)

    tokenized = tokenizer(data_list, return_tensors='pt', padding=True)

    mask = []
    for attn_mask in tokenized['attention_mask']:
        temp_mask = []
        for idx, m in enumerate(attn_mask):
            if idx == 0:
                temp_mask.append(False)
                if prompt_type == "random_prompt":
                    temp_mask = temp_mask + [True] * 10

                continue
            if m == 1:
                temp_mask.append(True)

            else:
                temp_mask.append(False)
        mask.append(temp_mask)

    token_ids = tokenized['input_ids']

    # random prompt
    if prompt_type == "random_prompt":
        random_prompts = torch.randint(
            low=100, high=tokenized['input_ids'].max().item(),
            size=(tokenized['input_ids'].shape[0], 10),
            dtype=torch.long
        )

        token_ids = torch.cat([tokenized['input_ids'][:, :1], random_prompts, tokenized['input_ids'][:, 1:]], dim=1)

    dataset_tosave = datasets.Dataset.from_dict({
        'content': content_list,
        'input_ids': token_ids.tolist(),
        # 'input_ids': token_ids.tolist(),
        'mask': mask,
        'label': label_list
    })

    dataset_tosave.set_format(type='torch', columns=['input_ids'])

    save_path = r'/mnt/nfs/algo/intern/haoyunx11_/idea/MI/sn/data/' + "xnli" + "/" + language + "/" + prompt_type
    dataset_tosave.save_to_disk(save_path)


def load_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained("/mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf")
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


if __name__ == '__main__':
    tokenizer = load_tokenizer(model_path=r'/mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf')

    lang_list = ['bg', 'es', 'fr']
    prompt_type_list = ["random_prompt", "simple_prompt", "standard_prompt"]

    for lang in lang_list:
        for prompt_type in prompt_type_list:
            save_xnli_tokenized_datasets(tokenizer, language=lang, prompt_type=prompt_type)
