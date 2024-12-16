import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import einops
import numpy as np

import datasets
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def generate_hs(system_output_1, source_path, human_score_path,
                prompt=r'Translate the following text from English to Chinese.',
                model_path=r'/mnt/nfs/algo/intern/haoyunx11/models/sft/final/mt/org/enzh/checkpoint-3200'):
    f_1 = open(system_output_1, 'r', encoding='utf-8')
    f_s = open(source_path, 'r', encoding='utf-8')

    system_1_lines = f_1.readlines()
    source_lines = f_s.readlines()

    system_name_1 = system_output_1.split('/')[-1].replace('.txt', '').strip()

    system_1_score_list = locate_system_human_score(system_name_1, human_score_path)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=1024,
        padding_side="left",
        truncation_side='left',
        padding=True,
        truncation=True
    )

    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path)

    model.eval()
    model.to('cuda')

    lines_length = len(system_1_lines)

    layers = list(range(model.config.num_hidden_layers))

    print('rows_number: ' + str(lines_length))
    layer_activations = {
        l: torch.zeros(lines_length, model.config.hidden_size,
                       dtype=torch.float16)
        for l in layers
    }

    dir_path = '/mnt/nfs/algo/intern/haoyunx11_/idea/scorer/mt_activations/enzh' + '/' + system_name_1
    for i in range(lines_length):
        path = dir_path + '/' + str(i)
        if not os.path.exists(path):
            os.mkdir(dir_path + '/' + str(i))

    for idx, line in enumerate(zip(system_1_lines, source_lines, system_1_score_list)):
        system_1_score = line[2]

        if system_1_score == None:
            print("None score!")
            print(idx)
            continue

        input_s = prompt + ' ' + line[1].strip()
        output_s_1 = line[0].strip()

        tokenized_1 = tokenizer([input_s, output_s_1], return_tensors='pt', padding=True)
        inputs = {k: v.to('cuda') for k, v in tokenized_1.items()}
        # batch = [tokenized_1[''], tokenized_2]

        out = model(**inputs, output_hidden_states=True,
                    output_attentions=False, return_dict=True, use_cache=False)

        hidden_states = out.hidden_states
        # print(len(hidden_states))

        for layer_idx, hidden_state in enumerate(hidden_states):
            hidden_state = hidden_state.detach().cpu().to(torch.float).numpy()

            save_path = dir_path + '/' + str(idx) + '/' + f'layer_{layer_idx}.npy'
            np.save(save_path, hidden_state)




def locate_system_human_score(system_name_1, human_score_path):
    f_human_score = open(human_score_path, 'r', encoding='utf-8')

    system_1_score_list = []

    lines = f_human_score.readlines()
    for line in lines:
        if system_name_1 in line:
            system_1_score_list.append(line.split('	')[-1].strip())

    return system_1_score_list


def process_activation_batch(activation_aggregation, batch_activations, batch_mask=None):
    cur_batch_size = batch_activations.shape[0]

    if activation_aggregation is None:
        # only save the activations for the required indices
        batch_activations = einops.rearrange(
            batch_activations, 'b c d -> (b c) d')  # batch, context, dim
        processed_activations = batch_activations[batch_mask]

    if activation_aggregation == 'last':
        last_ix = batch_activations.shape[1] - 1
        batch_mask = batch_mask.to(int)
        last_entity_token = last_ix - \
                            torch.argmax(batch_mask.flip(dims=[1]), dim=1)
        d_act = batch_activations.shape[2]
        expanded_mask = last_entity_token.unsqueeze(-1).expand(-1, d_act)
        processed_activations = batch_activations[
            torch.arange(cur_batch_size).unsqueeze(-1),
            expanded_mask,
            torch.arange(d_act)
        ]
        assert processed_activations.shape == (cur_batch_size, d_act)

    elif activation_aggregation == 'mean':
        # average over the context dimension for valid tokens only
        shape_0 = batch_mask.shape[0]
        shape_1 = batch_mask.shape[1]
        batch_mask_ = batch_mask.reshape(shape_0, shape_1, 1)

        batch_valid_ixs = batch_mask.sum(dim=1)  # batch_mask 中为 True 的长度

        masked_activations = batch_activations * batch_mask_

        processed_activations = masked_activations.sum(dim=1) / batch_valid_ixs[:, None]

    elif activation_aggregation == 'max':
        # max over the context dimension for valid tokens only (set invalid tokens to -1)
        batch_mask = batch_mask[:, :, None].to(int)
        # set masked tokens to -1
        masked_activations = batch_activations * batch_mask + (batch_mask - 1)
        processed_activations = masked_activations.max(dim=1)[0]

    return processed_activations


if __name__ == '__main__':
    generate_hs(
        system_output_1=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/scorer/data/enzh/system_outputs/Baidu-system.6932.txt',
        source_path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/scorer/data/enzh/source/en-zh.txt',
        human_score_path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/scorer/data/enzh/human_scores/en-zh.wmt-z.seg.score',
        prompt=r'Translate the following text from English to Chinese.',
        model_path=r'/mnt/nfs/algo/intern/haoyunx11/models/sft/final/mt/org/enzh/checkpoint-3200')
