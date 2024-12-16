from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import pandas as pd
import csv
import json


def cal_model_weight_abs(model_path_1=r'/mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf',
                         model_path_2=r'/mnt/nfs/algo/intern/haoyunx11/models/sft/summary/enfr/train_enfr_org/checkpoint-250',
                         save_dir=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/summary_neurons_sim/enfr/org_allsft/250steps',
                         is_lora=False):
    from transformer_lens import HookedTransformer
    tokenizer = AutoTokenizer.from_pretrained(model_path_1)

    hf_model_1 = AutoModelForCausalLM.from_pretrained(model_path_1)
    model_1 = HookedTransformer.from_pretrained(r'Llama-2-7b-chat-hf', hf_model=hf_model_1, device="cpu",
                                                fold_ln=False,
                                                center_writing_weights=False, center_unembed=True,
                                                tokenizer=tokenizer)

    hf_model_2 = AutoModelForCausalLM.from_pretrained(model_path_2)
    model_2 = HookedTransformer.from_pretrained(r'Llama-2-7b-chat-hf', hf_model=hf_model_2, device="cpu",
                                                fold_ln=False,
                                                center_writing_weights=False, center_unembed=True,
                                                tokenizer=tokenizer)

    W_in_1 = model_1.W_in
    W_in_1 = W_in_1.swapaxes(1, 2).detach().to(torch.float).cpu().numpy()
    print(W_in_1.shape)  # (11008, 4096)

    W_gate_1 = model_1.W_gate
    W_gate_1 = W_gate_1.swapaxes(1, 2).detach().to(torch.float).cpu().numpy()
    print(W_gate_1.shape)

    W_out_1 = model_1.W_out.detach().to(torch.float).cpu().numpy()
    print(W_out_1.shape)

    W_in_2 = model_2.W_in
    W_in_2 = W_in_2.swapaxes(1, 2).detach().to(torch.float).cpu().numpy()

    W_gate_2 = model_2.W_gate
    W_gate_2 = W_gate_2.swapaxes(1, 2).detach().to(torch.float).cpu().numpy()

    W_out_2 = model_2.W_out.detach().to(torch.float).cpu().numpy()

    # igo_list = ['in', 'gate', 'out']
    igo_list = ['in', 'out']
    # igo_list = ['gate']

    count = 0

    for igo in igo_list:
        print(igo)
        if igo == 'in':
            path_tail = '/abs_in.csv'
            W_1 = W_in_1
            W_2 = W_in_2
        elif igo == 'out':
            path_tail = '/abs_out.csv'
            W_1 = W_out_1
            W_2 = W_out_2
        elif igo == 'gate':
            path_tail = '/cos_gate.csv'
            W_1 = W_gate_1
            W_2 = W_gate_2

        weights_abs = {}
        for layer_idx, w_layers in enumerate(zip(W_1, W_2)):
            print(layer_idx)
            print('********************')
            for neuron_idx, w in enumerate(zip(w_layers[0], w_layers[1])):
                for idx, w_ in enumerate(zip(w[0], w[1])):
                    diff = abs(w_[0] - w_[1])
                    if diff > 4.5e-4:
                        weights_abs[(layer_idx, neuron_idx, idx)] = diff
                        count += 1

                print('count: ' + str(count))

        corr_df = pd.DataFrame({'corr': pd.Series(weights_abs)})
        corr_df.index.names = ['layer_idx', 'neuron_idx', 'idx']
        corr_df = corr_df.reset_index()
        print('start write file')
        corr_df.to_csv(save_dir + path_tail, index=False)


def cal_model_weight_abs_txt(model_path_1=r'/mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf',
                             model_path_2=r'/mnt/nfs/algo/intern/haoyunx11/models/sft/summary/enfr/train_enfr_org/checkpoint-250',
                             save_dir=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/summary_neurons_sim/enfr/org_allsft/250steps',
                             is_lora=False):
    from transformer_lens import HookedTransformer
    tokenizer = AutoTokenizer.from_pretrained(model_path_1)

    hf_model_1 = AutoModelForCausalLM.from_pretrained(model_path_1)
    model_1 = HookedTransformer.from_pretrained(r'Llama-2-7b-chat-hf', hf_model=hf_model_1, device="cpu",
                                                fold_ln=False,
                                                center_writing_weights=False, center_unembed=True,
                                                tokenizer=tokenizer)

    hf_model_2 = AutoModelForCausalLM.from_pretrained(model_path_2)
    model_2 = HookedTransformer.from_pretrained(r'Llama-2-7b-chat-hf', hf_model=hf_model_2, device="cpu",
                                                fold_ln=False,
                                                center_writing_weights=False, center_unembed=True,
                                                tokenizer=tokenizer)

    W_in_1 = model_1.W_in
    W_in_1 = W_in_1.swapaxes(1, 2).detach().to(torch.float).cpu().numpy()
    print(W_in_1.shape)  # (11008, 4096)

    W_gate_1 = model_1.W_gate
    W_gate_1 = W_gate_1.swapaxes(1, 2).detach().to(torch.float).cpu().numpy()
    print(W_gate_1.shape)

    W_out_1 = model_1.W_out.detach().to(torch.float).cpu().numpy()
    print(W_out_1.shape)

    W_in_2 = model_2.W_in
    W_in_2 = W_in_2.swapaxes(1, 2).detach().to(torch.float).cpu().numpy()

    W_gate_2 = model_2.W_gate
    W_gate_2 = W_gate_2.swapaxes(1, 2).detach().to(torch.float).cpu().numpy()

    W_out_2 = model_2.W_out.detach().to(torch.float).cpu().numpy()

    # igo_list = ['in', 'gate', 'out']
    igo_list = ['in', 'out']
    # igo_list = ['gate']

    for igo in igo_list:
        print(igo)
        if igo == 'in':
            path_tail = '/abs_in.csv'
            W_1 = W_in_1
            W_2 = W_in_2
        elif igo == 'out':
            path_tail = '/abs_out.csv'
            W_1 = W_out_1
            W_2 = W_out_2
        elif igo == 'gate':
            path_tail = '/cos_gate.csv'
            W_1 = W_gate_1
            W_2 = W_gate_2

        writer = open(save_dir + path_tail, 'w', encoding='utf-8')

        # weights_abs = {}
        for layer_idx, w_layers in enumerate(zip(W_1, W_2)):
            print(layer_idx)
            for neuron_idx, w in enumerate(zip(w_layers[0], w_layers[1])):
                for idx, w_ in enumerate(zip(w[0], w[1])):
                    diff = abs(w_[0] - w_[1])
                    writer.write(str(diff) + '\n')

        # corr_df = pd.DataFrame({'corr': pd.Series(weights_abs)})
        # corr_df.index.names = ['layer_idx', 'neuron_idx', 'idx']
        # # corr_df = corr_df.reset_index()
        # print('start write file')
        # corr_df.to_csv(save_dir + path_tail, index=False)


def get_threhold_score(
        path_1=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/org_allsft/2500steps/cos_in.csv',
        path_2=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/org_allsft/2500steps/cos_out.csv',
        number=614400000):
    f_1 = open(path_1, 'r', encoding='utf-8')
    rows_1 = csv.reader(f_1)

    f_2 = open(path_2, 'r', encoding='utf-8')
    rows_2 = csv.reader(f_2)

    sum = 0
    score_list = []
    print('row_1 start')
    for idx1, row in enumerate(rows_1):
        if idx1 == 0:
            continue
        score = float(row[-1])
        # print(score)
        sum += score
        score_list.append(score)

    print('row_2 start')
    for idx2, row in enumerate(rows_2):
        if idx2 == 0:
            continue
        score = float(row[-1])
        sum += score
        score_list.append(score)

    # score_list.sort()
    print('start sorting')
    score_list.sort(reverse=True)

    avg = sum / (idx1 + idx2)

    # print(score_list[:neuron_number])
    print(avg)
    # print(count)

    # 204800000
    print('For 819200000')
    print(score_list[:819200000][-1])
    print('***************************')

    # 614400000
    print('For 614400000')
    print(score_list[:614400000][-1])   # 0.001132274
    print('***************************')

    # 409600000
    print('For 409600000')
    print(score_list[:409600000][-1])
    print('***************************')

    # 204800000
    print('For 204800000')
    print(score_list[:204800000][-1])
    print('***************************')

    return score_list[:number][-1]


# get mask 跳过 合并到 make_mask_with_dict_and_save
def get_mask(
        path_dir=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/sparse_training/results/enzh',
        threhold=0.001132274,
        save_path=r"/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/sparse_training/results/enzh/150000/mask_cos.pt"):
    cos_in_path = path_dir + '/abs_in.csv'
    # cos_gate_path = path_dir + '/cos_gate.csv'
    cos_out_path = path_dir + '/abs_out.csv'
    count = 0

    mask_dict = {}
    for path_idx, path in enumerate([cos_in_path, cos_out_path]):
        f = open(path, 'r', encoding='utf-8')
        rows = csv.reader(f)
        if path_idx == 0:
            igo = 'in'
        elif path_idx == 1:
            igo = 'out'



        print(igo)
        temp_layer_name = ''

        for idx, row in enumerate(rows):
            if idx == 0:
                continue

            neuron_layer = int(row[0])
            neuron = int(row[1])
            w = int(row[2])
            score = float(row[-1])

            if score >= threhold:
                count += 1
                layer_name = str(neuron_layer) + '_' + igo
                if layer_name != temp_layer_name:
                    temp_layer_name = layer_name
                    print(temp_layer_name)

                if layer_name not in mask_dict.keys():
                    mask_dict[layer_name] = torch.zeros(11008 * 4096)

                if igo == 'in' or igo == 'gate':
                    mask_dict[layer_name][neuron * 4096 + w] = 1

                else:
                    index = w * 11008 + neuron
                    mask_dict[layer_name][index] = 1
    print(count)
    torch.save(mask_dict, save_path)


def test():
    neuron = 2
    w = 0
    s = neuron * 4096 + w
    print(s)


def make_mask_with_dict_and_save(
        path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/org_allsft/500steps/neuron_dict.json',
        save_path=r"/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/org_allsft/500steps/mask_cos.pt"):
    f = open(path, 'r', encoding='utf-8')
    neuron_dict = json.load(f)
    mask_dict = {}

    for layer_idx in range(32):
        print(layer_idx)
        for igo in ['in', 'out']:
            layer_name = str(layer_idx) + '_' + igo
            mask_dict[layer_name] = torch.zeros(11008 * 4096)
            if layer_name not in neuron_dict.keys():
                continue
            if igo == 'in' or igo == 'gate':
                for neuron in neuron_dict[layer_name]:
                    mask_dict[layer_name][neuron * 4096: (neuron + 1) * 4096] = 1
            else:
                for neuron in neuron_dict[layer_name]:
                    for i in range(4096):
                        index = i * 11008 + neuron
                        mask_dict[layer_name][index] = 1

    torch.save(mask_dict, save_path)

    return mask_dict


if __name__ == '__main__':
    # cal_model_weight_abs(model_path_1=r'/data2/haoyun/models/llm/Llama-2-7b-chat-hf',
    #                      model_path_2=r'/mnt/nfs/algo/intern/haoyunx11/models/sft/final/mt/org/hizh/checkpoint-750',
    #                      save_dir=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/sparse_training/results/hizh')

    # get_threhold_score(
    #     path_1=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/sparse_training/results/hizh/abs_in.csv',
    #     path_2=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/sparse_training/results/hizh/abs_out.csv')

    get_mask(path_dir=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/sparse_training/results/hizh',
             threhold=0.0004841424,
             save_path=r"/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/sparse_training/results/hizh/200000/mask_abs.pt")

# python get_mask.py >get_threhold_score_hizh.log 2>&1 &
# python get_mask.py >cal_hizh.log 2>&1 &
