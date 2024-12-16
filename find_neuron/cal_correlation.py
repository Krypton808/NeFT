import numpy as np
import einops
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import json
import math
import gc
import csv
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr, pearsonr
from peft import PeftModel
from find_by_overlap import two_neuron_dict_overlap, two_neuron_dict_disjoint


# from torch.nn.functional import cosine_similarity


def get_top_neurons(learned_probes, model, k=50, W='W_in'):
    if W == 'W_in':
        W_norm = (model.W_in / model.W_in.norm(dim=1, keepdim=True)).swapaxes(1, 2)

    elif W == 'W_gate':
        W_norm = (model.W_gate / model.W_in.norm(dim=1, keepdim=True)).swapaxes(1, 2)


    else:
        W_norm = (model.W_out / model.W_out.norm(dim=-1, keepdim=True))

    n_layers, d_mlp, d_model = W_norm.shape
    # print(W_norm.shape)

    W_comp = einops.einsum(W_norm, learned_probes.float(), 'l1 m d, l2 d-> l2 l1 m')

    top_neurons = W_comp.flatten().abs().argsort()
    _, top_layers, top_neurons = np.unravel_index(top_neurons[-k:], (learned_probes.shape[0], n_layers, d_mlp))
    print("top_layers: ")
    print(top_layers)
    print()
    print("top_neurons: ")
    print(top_neurons)
    return top_layers, top_neurons


def get_hs(
        path_dir=r'/mnt/nfs/algo/intern/haoyunx11_/idea/MI/sn/mt_activations/layer/enzh_sft_layer8_deepspeed/2500steps/flores101/enzh/after_gen'):
    hs_all_layers_list = []
    for i in range(33):
        path = path_dir + '/layer_' + str(i) + '.npy'

        layer_hs = np.load(path)
        # print(layer_hs.shape) # (997, 4096)
        hs_all_layers_list.append(layer_hs)

    return hs_all_layers_list


def cal_corr_by_neuron_demo(W_in, hs_all_layers_list, w_model_name='', hs_model_name='', neuron_weight=''):
    print(len(W_in))
    hs_all_layers_list = hs_all_layers_list[1:]

    score_dict = {}

    W_in = W_in.detach().to(torch.float).cpu().numpy()

    for idx, w_layer in enumerate(W_in):
        for hs_idx, hs in enumerate(hs_all_layers_list):
            # w_layer = np.mean(w_layer, axis=0, keepdims=True)
            hs = np.mean(hs, axis=0, keepdims=True)

            # print(w_layer.shape)
            # print(hs.shape)

            # hs = torch.tensor(hs)

            score = cosine_similarity(w_layer, hs)

            # print(score)
            # print(score.shape)
            if str(idx) + '_' + str(hs_idx) not in score_dict.keys():
                score_dict[str(idx) + '_' + str(hs_idx)] = [s[0] for s in score]
            else:
                score_dict[str(idx) + '_' + str(hs_idx)].append(score)

    neuron_corr = {}

    for k, values in score_dict.items():
        # print(values)
        for idx, v in enumerate(values):
            abs_v = abs(v)
            if abs_v > 0.05:
                neuron_layer, activation_layer = k.split('_')
                neuron = idx
                corr = v

                neuron_corr[(neuron_layer, neuron, activation_layer)] = corr

    corr_df = pd.DataFrame({'corr': pd.Series(neuron_corr)})
    corr_df.index.names = ['neuron_layer', 'neuron', 'activation_layer']
    corr_df = corr_df.reset_index()
    corr_df['neuron_weight'] = pd.Series([neuron_weight] * corr_df.shape[0])
    corr_df['abs_corr'] = np.abs(corr_df['corr'].values)

    return corr_df


def cal_corr_by_neuron_demo_pearsonr(W_in, hs_all_layers_list, w_model_name='', hs_model_name='', neuron_weight=''):
    print(len(W_in))
    hs_all_layers_list = hs_all_layers_list[1:]

    score_dict = {}

    W_in = W_in.detach().to(torch.float).cpu().numpy()

    for idx, w_layer in enumerate(W_in):
        print('hs_all_layers_list length: ' + str(len(hs_all_layers_list)))

        for hs_idx, hs in enumerate(hs_all_layers_list):
            # w_layer = np.mean(w_layer, axis=0, keepdims=True)
            hs = np.mean(hs, axis=0, keepdims=True)

            score = []

            # print('w_layer length: ' + str(len(w_layer)))   # 11008
            for w in w_layer:
                s = pearsonr(w, hs[0]).statistic
                score.append(s)

            # print(score)
            # print(score.shape)
            if str(idx) + '_' + str(hs_idx) not in score_dict.keys():
                score_dict[str(idx) + '_' + str(hs_idx)] = score
            else:
                score_dict[str(idx) + '_' + str(hs_idx)].append(score)

    neuron_corr = {}

    for k, values in score_dict.items():
        # print(values)
        for idx, v in enumerate(values):
            # abs_v = abs(v)
            # print('abs_v: ' + str(abs_v))
            # if abs_v > 0.05:
            #     neuron_layer, activation_layer = k.split('_')
            #     neuron = idx
            #     corr = v
            #
            #     neuron_corr[(neuron_layer, neuron, activation_layer)] = corr

            neuron_layer, activation_layer = k.split('_')
            neuron = idx
            corr = v

            neuron_corr[(neuron_layer, neuron, activation_layer)] = corr

    corr_df = pd.DataFrame({'corr': pd.Series(neuron_corr)})
    corr_df.index.names = ['neuron_layer', 'neuron', 'activation_layer']
    corr_df = corr_df.reset_index()
    corr_df['neuron_weight'] = pd.Series([neuron_weight] * corr_df.shape[0])
    corr_df['abs_corr'] = np.abs(corr_df['corr'].values)

    return corr_df


def get_neuron():
    hs_path = '/mnt/nfs/algo/intern/haoyunx11_/idea/MI/sn/mt_activations/train_enzh_org/2500steps/flores_101/enzh/after_gen/'
    hs_all_layers_list = get_hs(hs_path)
    hs_all_layers_list = hs_all_layers_list[1:]  # 去除embedding layer

    model_path = '/mnt/nfs/algo/intern/haoyunx11/models/sft/mt/train_enzh_org/checkpoint-2500'

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    hf_model = AutoModelForCausalLM.from_pretrained(model_path)
    model = HookedTransformer.from_pretrained(r'Llama-2-7b-chat-hf', hf_model=hf_model, device="cpu",
                                              fold_ln=False,
                                              center_writing_weights=False, center_unembed=True,
                                              tokenizer=tokenizer)

    W_in = model.W_in
    W_in = W_in.swapaxes(1, 2)
    print(W_in.shape)

    # W_gate = model.W_gate
    # W_gate = W_gate.swapaxes(1, 2)

    W_out = model.W_out
    print(W_out.shape)

    W_dict = {'W_in': W_in, 'W_out': W_out}

    count = 0

    neuron_dict = {}

    for k, W in W_dict.items():
        score_dict = {}
        if k == 'W_in':
            igo = 'in'
        elif k == 'W_out':
            igo = 'out'

        print(len(W))

        W = W.detach().to(torch.float).cpu().numpy()

        for idx, w_layer in enumerate(W):
            for hs_idx, hs in enumerate(hs_all_layers_list):
                hs = np.mean(hs, axis=0, keepdims=True)

                if w_layer.shape != (11008, 4096) or hs.shape != (1, 4096):
                    print(w_layer.shape)
                    print(hs.shape)
                    print(idx)
                    print(hs_idx)

                score = cosine_similarity(w_layer, hs)

                if str(idx) + '_' + str(hs_idx) not in score_dict.keys():
                    score_dict[str(idx) + '_' + str(hs_idx)] = [s[0] for s in score]
                else:
                    score_dict[str(idx) + '_' + str(hs_idx)].append(score)

        for k, values in score_dict.items():
            # print(values)
            for idx, v in enumerate(values):
                abs_v = abs(v)

                try:
                    if abs_v > 0.05:
                        neuron_layer, activation_layer = k.split('_')
                        neuron = idx

                        layer_name = neuron_layer + '_' + igo

                        if layer_name not in neuron_dict.keys():
                            neuron_dict[layer_name] = [neuron]
                        else:
                            if neuron not in neuron_dict[layer_name]:
                                count += 1
                                neuron_dict[layer_name].append(neuron)
                except:
                    print(k)
                    print(abs_v)
                    return

    print(count)
    with open(
            "/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/W_HS_self/fullparam_after_gen_neuron_dict.json",
            "w") as f:
        json.dump(neuron_dict, f)
    print("加载入文件完成...")


def cal_corr_by_layer_demo(W_in, hs_all_layers_list):
    print(len(W_in))
    hs_all_layers_list = hs_all_layers_list[1:]

    score_dict = {}

    W_in = W_in.detach().to(torch.float).cpu().numpy()

    for idx, w_layer in enumerate(W_in):
        for hs in hs_all_layers_list:
            w_layer = np.mean(w_layer, axis=0, keepdims=True)
            hs = np.mean(hs, axis=0, keepdims=True)

            # print(w_layer.shape)
            # print(hs.shape)

            # hs = torch.tensor(hs)
            score = cosine_similarity(w_layer, hs)
            # print(score)
            # print(score.shape)
            if str(idx) not in score_dict.keys():
                score_dict[str(idx)] = [score]
            else:
                score_dict[str(idx)].append(score)

    for k, values in score_dict.items():
        # print(values)
        for idx, v in enumerate(values):
            abs_v = abs(v)
            if abs_v > 0.03:
                print(k)
                print(v)
                print(idx)
                print('*************************')
            # if v[0][0] > 0.03:
            #     print(v[0][0])
            #     print(idx)


def abs_difference(path_1, path_2, save_path):
    f_1 = open(path_1, 'r', encoding='utf-8')
    f_2 = open(path_2, 'r', encoding='utf-8')

    rows_1 = csv.reader(f_1)
    rows_2 = csv.reader(f_2)

    w = open(save_path, 'w', encoding='utf-8')
    writer = csv.writer(w)
    writer.writerow(
        ('neuron_layer', 'neuron', 'activation_layer', 'neuron_weight', 'abs_corr_1', 'abs_corr_2', 'diff', 'abs_diff'))

    for idx, row in enumerate(zip(rows_1, rows_2)):
        if idx == 0:
            continue

        neuron_layer = row[0][0]
        neuron = row[0][1]
        activation_layer = row[0][2]
        neuron_weight = row[0][4]
        abs_corr_1 = row[0][5]
        abs_corr_2 = row[1][5]

        diff = float(abs_corr_1) - float(abs_corr_2)
        abs_diff = abs(diff)

        writer.writerow(
            (neuron_layer, neuron, activation_layer, neuron_weight, abs_corr_1, abs_corr_2, diff, abs_diff))


# bying setting threhold
def check_diff(
        path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/W_HS_self/pearsonr/all/abs_diff_W_in_correlaton_W.llama-2.Llama-2-7b-chat-hf_W_in_correlaton_W.train_enzh_org.checkpoint-2500.csv'):
    f = open(path, 'r', encoding='utf-8')
    rows = csv.reader(f)
    ret_list = []
    name_list = []

    diff_mean = 0
    for idx, row in enumerate(rows):
        if idx == 0:
            print(row)
            continue

        # if idx <= 10:
        #     print(row)
        # print('**********************************')

        diff = row[7]
        diff_mean += float(diff)

        if float(diff) >= 0.01:
            print(row)
            ret_list.append(row)
            name_list.append(str(row[0]) + '_' + str(row[1]))

    diff_mean = diff_mean / idx
    print(diff_mean)
    print('row length: ' + str(idx))

    name_list = list(set(name_list))

    return ret_list, name_list


def index_diff_and_cos_score(
        diff_path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/W_HS_self/pearsonr/all/abs_diff_W_in_correlaton_W.llama-2.Llama-2-7b-chat-hf_W_in_correlaton_W.train_enzh_org.checkpoint-2500.csv',
        cos_path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/org_allsft/2500steps/cos_in.csv'):
    f = open(cos_path, 'r', encoding='utf-8')
    rows = csv.reader(f)

    ret_list, name_list = check_diff(diff_path)
    for row in rows:
        name = str(row[0]) + '_' + str(row[1])
        if name in name_list:
            print(row)


# rank abs_diff_in , abs_diff_out
def rank_and_get_topK(
        path_in=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/xnli/en/en_org-en_1000step_full/abs_diff_in.csv',
        path_out=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/xnli/en/en_org-en_1000step_full/abs_diff_out.csv',
        path_dir_w='/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/xnli/en/en_org-en_1000step_full',
        k=100000, rank_name='abs_diff'):
    # 读取文件
    f_in = open(path_in, 'r', encoding='utf-8')
    rows_in = csv.reader(f_in)
    f_out = open(path_out, 'r', encoding='utf-8')
    rows_out = csv.reader(f_out)
    # 'neuron_layer', 'neuron', 'activation_layer', 'neuron_weight', 'abs_corr_1', 'abs_corr_2', 'diff', 'abs_diff'
    neuron_name_dict = {}
    if rank_name == 'abs_diff':
        rank_name_idx = 7
    elif rank_name == 'abs_corr_1':
        rank_name_idx = 4
    elif rank_name == 'abs_corr_2':
        rank_name_idx = 5

    for rows in [rows_in, rows_out]:
        for idx, row in enumerate(rows):
            if idx == 0:
                continue

            neuron_layer = row[0]
            neuron = row[1]
            activation_layer = row[2]

            if row[3] == 'W_in':
                neuron_weight = 'in'
            else:
                neuron_weight = 'out'

            rank_value = float(row[rank_name_idx])
            # print(rank_value)

            neuron_name = str(neuron_layer) + '_' + str(neuron) + '_' + activation_layer + '_' + neuron_weight

            if neuron_name not in neuron_name_dict.keys():
                neuron_name_dict[neuron_name] = rank_value
            else:
                if rank_value > neuron_name_dict[neuron_name]:
                    neuron_name_dict[neuron_name] = rank_value

    neuron_name_dict_sorted = sorted(neuron_name_dict.items(), key=lambda d: d[1], reverse=True)

    neuron_dict = {}
    count = 0
    for line in neuron_name_dict_sorted:
        print(line)
        if count >= k:
            break
        neuron_name = line[0].split('_')
        neuron_layer = neuron_name[0]
        neuron = int(neuron_name[1])
        igo = neuron_name[2]
        layer_name = str(neuron_layer) + '_' + igo

        if layer_name not in neuron_dict.keys():
            neuron_dict[layer_name] = [neuron]
        else:
            neuron_dict[layer_name].append(neuron)

        count += 1

    new_dict = {}
    for i in range(32):
        key_in = str(i) + '_' + 'in'
        key_out = str(i) + '_' + 'out'

        if key_in in neuron_dict.keys():
            neuron_list_in = neuron_dict[key_in]
            neuron_list_in.sort()

        if key_out in neuron_dict.keys():
            neuron_list_out = neuron_dict[key_out]
            neuron_list_out.sort()

        new_dict[key_in] = neuron_list_in
        new_dict[key_out] = neuron_list_out

    # with open(path_dir_w + '/' + str(k) + "/neuron_dict_" + rank_name + ".json", "w") as f:
    #     json.dump(new_dict, f)
    #     print("加载入文件完成...")


def get_only_non_zero(
        path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/xnli/en/en_org-en_1000step_full/abs_diff_in.csv',
        path_w='/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/xnli/en/en_org-en_1000step_full'):
    # 读取文件
    # f = open(path, 'r', encoding='utf-8')
    df = pd.read_csv(
        path)  # 'neuron_layer', 'neuron', 'activation_layer', 'neuron_weight', 'abs_corr_1', 'abs_corr_2', 'diff', 'abs_diff'

    df = df.loc[df['abs_corr_1'] != 0]
    df.to_csv(path_w, index=False)
    print(df)


def rank_and_check(
        path_in=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/xnli/en/en_org-en_1000step_full/abs_diff_in.csv',
        path_out=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/xnli/en/en_org-en_1000step_full/abs_diff_out.csv',
        path_cos_in=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_enzh_org/3200step/cos_in.csv',
        path_cos_out=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_enzh_org/3200step/cos_out.csv'):
    # 读取文件
    f_cos_in = open(path_cos_in, 'r', encoding='utf-8')  # length 352257
    rows_cos_in = csv.reader(f_cos_in)

    f_cos_out = open(path_cos_out, 'r', encoding='utf-8')  # length 352257
    rows_cos_out = csv.reader(f_cos_out)

    neuron_name_cos_dict = {}
    sum = 0
    for idx1, row in enumerate(rows_cos_in):
        if idx1 == 0:
            continue
        neuron_name = str(row[0]) + '_' + str(row[1]) + '_' + 'in'
        score = float(row[3])
        neuron_name_cos_dict[neuron_name] = score
        sum += score

    for idx2, row in enumerate(rows_cos_out):
        if idx2 == 0:
            continue
        neuron_name = str(row[0]) + '_' + str(row[1]) + '_' + 'out'
        score = float(row[3])
        neuron_name_cos_dict[neuron_name] = score
        sum += score

    avg = sum / (idx1 + idx2)

    f_in = open(path_in, 'r', encoding='utf-8')
    rows_in = csv.reader(f_in)
    f_out = open(path_out, 'r', encoding='utf-8')
    rows_out = csv.reader(f_out)
    # 'neuron_layer', 'neuron', 'activation_layer', 'neuron_weight', 'abs_corr_1', 'abs_corr_2', 'diff', 'abs_diff'
    neuron_name_dict_abs_diff = {}
    neuron_name_dict_diff = {}

    for rows in [rows_in, rows_out]:
        for idx, row in enumerate(rows):
            if idx == 0:
                continue

            neuron_layer = row[0]
            neuron = row[1]

            if row[3] == 'W_in':
                neuron_weight = 'in'
            else:
                neuron_weight = 'out'

            diff = row[6]
            abs_diff = float(row[7])

            neuron_name = str(neuron_layer) + '_' + str(neuron) + '_' + neuron_weight
            if abs_diff > 0.01:
                print('==============================')
                print(neuron_name)
                print(diff)
                print('abs_diff > 0.01')
                print('==============================')

            if neuron_name not in neuron_name_dict_abs_diff.keys():
                neuron_name_dict_abs_diff[neuron_name] = abs_diff
                neuron_name_dict_diff[neuron_name] = diff
            else:
                if abs_diff > neuron_name_dict_abs_diff[neuron_name]:
                    neuron_name_dict_abs_diff[neuron_name] = abs_diff
                    neuron_name_dict_diff[neuron_name] = diff

    neuron_name_dict_sorted = sorted(neuron_name_dict_abs_diff.items(), key=lambda d: d[1], reverse=True)

    neuron_dict = {}
    count = 0
    count_align = 0
    for line in neuron_name_dict_sorted:
        if count >= 10000:
            break

        print('neuron name: ')
        print(line[0])
        print('cos score: ')
        cos_score = neuron_name_cos_dict[line[0]]
        print(cos_score)
        print('diff score: ')
        diff = neuron_name_dict_diff[line[0]]
        print(diff)

        if cos_score < avg:
            count_align += 1
        else:
            print('not aligned')

        print('****************************')

        neuron_name = line[0].split('_')
        neuron_layer = neuron_name[0]
        neuron = int(neuron_name[1])
        igo = neuron_name[2]
        layer_name = str(neuron_layer) + '_' + igo

        if layer_name not in neuron_dict.keys():
            neuron_dict[layer_name] = [neuron]
        else:
            neuron_dict[layer_name].append(neuron)

        count += 1

    print('++++++++++++++++++++++++++++++++++++')
    print(avg)
    print('++++++++++++++++++++++++++++++++++++')
    print('count align')
    print(count_align)


# get pearsonr
def run_1():
    from transformer_lens import HookedTransformer
    is_lora = True
    # W_in = model.W_in
    # print(W_in.shape)   # torch.Size([32, 4096, 11008])
    #
    # W_gate = model.W_gate
    # print(W_gate.shape) # torch.Size([32, 4096, 11008])
    #
    # W_out = model.W_out
    # print(W_out.shape)  # torch.Size([32, 11008, 4096])

    hs_path = '/data2/haoyun/MA_project/mt_activations/LoRA/enzh/rank_256/after_gen'

    hs_model_name = hs_path.split('/')[-6] + '.' + hs_path.split('/')[-5] + '.' + hs_path.split('/')[-2]
    print(hs_model_name)
    hs_all_layers_list = get_hs(hs_path)

    # org
    # model_path = '/data2/haoyun/models/llm/Llama-2-7b-chat-hf'

    # trained model
    model_path = '/data2/haoyun/models/sft/final/mt/LoRA/rank_256/adapter_model'
    # model_path = '/mnt/nfs/algo/intern/haoyunx11/models/sft/final/mt/lora/enzh/10w/rank_64/checkpoint-50001/adapter_model'

    tokenizer = AutoTokenizer.from_pretrained('/data2/haoyun/models/llm/Llama-2-7b-chat-hf')
    w_model_name = model_path.split('/')[-2] + '.' + model_path.split('/')[-1]
    print(w_model_name)

    if is_lora:
        hf_model_1 = AutoModelForCausalLM.from_pretrained('/data2/haoyun/models/llm/Llama-2-7b-chat-hf')
        hf_model = PeftModel.from_pretrained(hf_model_1, model_path, is_trainable=False)
        hf_model = hf_model.merge_and_unload()
        model = HookedTransformer.from_pretrained(r'Llama-2-7b-chat-hf', hf_model=hf_model, device="cpu",
                                                  fold_ln=False,
                                                  center_writing_weights=False, center_unembed=True,
                                                  tokenizer=tokenizer)
    else:
        hf_model = AutoModelForCausalLM.from_pretrained(model_path)
        model = HookedTransformer.from_pretrained(r'Llama-2-7b-chat-hf', hf_model=hf_model, device="cpu",
                                                  fold_ln=False,
                                                  center_writing_weights=False, center_unembed=True,
                                                  tokenizer=tokenizer)
    W_in = model.W_in
    W_in = W_in.swapaxes(1, 2)

    # W_gate = model.W_gate
    # W_gate = W_gate.swapaxes(1, 2)

    W_out = model.W_out

    W_dict = {'W_in': W_in, 'W_out': W_out}

    # corr_list = []
    for k, W in W_dict.items():
        corr_df = cal_corr_by_neuron_demo_pearsonr(W, hs_all_layers_list, w_model_name, hs_model_name, k)
        # print(k)
        # print(corr_df)
        save_dir = "/data2/haoyun/MA_project/LoRA_neurons/pearson/enzh/rank_256"
        corr_df.to_csv(
            os.path.join(save_dir, k + '_correlaton' + '_W.' + w_model_name + '_HS.' + hs_model_name + '_pearsonr.csv'),
            index=False)

        # corr_list.append(corr_df)

    # concated = pd.concat(corr_list)
    # save_dir = "/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/W_HS_self"

    # concated.to_csv(
    #     os.path.join(save_dir, 'k_correlaton' + '_W.' + w_model_name + '_HS.' + hs_model_name + '_pearsonr.csv'),
    #     index=False)


# in out abs 手动两次
def run_2():
    # path_1 = r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/enzh_org-enzh_3200step_full/W_in_correlaton_W.enzh.checkpoint-3200_HS.mt_activations.train_enzh_org.enzh_pearsonr.csv'
    # path_2 = r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/enzh_org-enzh_3200step_full/W_in_correlaton_W.llama-2.Llama-2-7b-chat-hf_HS.mt_activations.train_enzh_org.enzh_pearsonr.csv'

    # trained model pearsonr
    path_1 = r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/enzh_org-enzh_in_out/W_out_correlaton_W.enzh.checkpoint-3200_HS.mt_activations.train_enzh_in_out.enzh_pearsonr.csv'

    # # org model pearsonr
    path_2 = r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/enzh_org-enzh_in_out/W_out_correlaton_W.llm.Llama-2-7b-chat-hf_HS.mt_activations.train_enzh_in_out.enzh_pearsonr.csv'

    save_path = r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/enzh_org-enzh_in_out/abs_diff_out.csv'

    cos_path = r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_enzh_org/3200step/cos_out.csv'

    abs_difference(path_1, path_2, save_path)

    index_diff_and_cos_score(diff_path=save_path, cos_path=cos_path)


# get by number
# get abs pearson Top 100000
def run_3():
    path_dir = r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/enzh_org-enzh_sn_3200_50000'
    path_in = path_dir + '/abs_diff_in.csv'
    # path_out = path_dir + '/abs_diff_out.csv'

    # for name in ['abs_diff']:
    # for name in ['abs_corr_1', 'abs_corr_2']:
    #     rank_and_get_topK(path_in=path_in, path_out=path_out, path_dir_w=path_dir, k=200000, rank_name=name)
    #     rank_and_get_topK(path_in=path_in, path_out=path_out, path_dir_w=path_dir, k=200000, rank_name=name)
    get_only_non_zero(path=path_in, path_dir_w=path_dir)


# sort abs_diff
def run_4():
    path_dir = r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/enzh_org-enzh_3200step_full/abs_diff'

    neuron_weight_list = ['in', 'out']
    for neuron_weight in neuron_weight_list:

        path = path_dir + '/abs_diff_' + neuron_weight + '.csv'

        path_non_zero = path_dir + '/non_zero_' + neuron_weight + '.csv'
        path_w = path_dir + '/non_zero_' + neuron_weight + '_max_sorted.csv'

        get_only_non_zero(path=path, path_w=path_non_zero)

        f = open(path_non_zero, 'r', encoding='utf-8')
        w = open(path_w, 'w', encoding='utf-8')
        writer = csv.writer(w)
        writer.writerow(
            ('neuron_layer', 'neuron', 'activation_layer', 'neuron_weight', 'diff', 'abs_diff'))
        rows = csv.reader(f)
        neuron_abs_diff_dict = {}
        for idx, row in enumerate(rows):
            # print(row)
            if idx == 0:
                print(row)
                continue
            neuron_layer = row[0]
            neuron = row[1]
            activation_layer = row[2]
            # neuron_weight = row[3]
            diff = row[6]
            abs_diff = row[7]

            if int(neuron_layer) > int(activation_layer):
                continue

            neuron_name = str(neuron_layer) + "_" + str(neuron)
            if neuron_name not in neuron_abs_diff_dict.keys():
                neuron_abs_diff_dict[neuron_name] = {}
                neuron_abs_diff_dict[neuron_name]['max_abs_diff'] = abs_diff
                neuron_abs_diff_dict[neuron_name]['diff'] = diff
                neuron_abs_diff_dict[neuron_name]['max_abs_diff_activation_layer'] = activation_layer

            else:
                if float(abs_diff) > float(neuron_abs_diff_dict[neuron_name]['max_abs_diff']):
                    neuron_abs_diff_dict[neuron_name]['max_abs_diff'] = abs_diff
                    neuron_abs_diff_dict[neuron_name]['diff'] = diff
                    neuron_abs_diff_dict[neuron_name]['max_abs_diff_activation_layer'] = activation_layer

        neuron_name_dict_sorted = sorted(neuron_abs_diff_dict.items(), key=lambda d: float(d[1]['max_abs_diff']),
                                         reverse=True)

        for item in neuron_name_dict_sorted:
            neuron_name = item[0].split('_')
            neuron_layer = neuron_name[0]
            neuron = neuron_name[1]
            abs_diff = item[1]['max_abs_diff']
            activation_layer = item[1]['max_abs_diff_activation_layer']
            diff = item[1]['diff']

            writer.writerow(
                (neuron_layer, neuron, activation_layer, neuron_weight, diff, abs_diff))


# sort pearson
def run_5():
    path_w_dir = r'/data2/haoyun/MA_project/LoRA_neurons/pearson/enzh/rank_256/pearson'
    neuron_weight_list = ['in', 'out']

    for neuron_weight in neuron_weight_list:
        sum = 0

        if neuron_weight == 'in':
            path = r'/data2/haoyun/MA_project/LoRA_neurons/pearson/enzh/rank_256/W_in_correlaton_W.rank_256.adapter_model_HS.MA_project.mt_activations.rank_256_pearsonr.csv'
        elif neuron_weight == 'out':
            path = r'/data2/haoyun/MA_project/LoRA_neurons/pearson/enzh/rank_256/W_out_correlaton_W.rank_256.adapter_model_HS.MA_project.mt_activations.rank_256_pearsonr.csv'

        path_w = path_w_dir + '/' + 'max_sorted_' + neuron_weight + '.csv'
        f = open(path, 'r', encoding='utf-8')
        w = open(path_w, 'w', encoding='utf-8')
        writer = csv.writer(w)
        writer.writerow(
            ('neuron_layer', 'neuron', 'activation_layer', 'neuron_weight', 'corr', 'abs_corr'))
        rows = csv.reader(f)
        neuron_abs_corr_dict = {}
        for idx, row in enumerate(rows):
            # print(row)
            if idx == 0:
                print(row)
                continue
            neuron_layer = row[0]
            neuron = row[1]
            activation_layer = row[2]
            # neuron_weight = row[4]
            corr = row[3]
            abs_corr = row[5]

            sum += float(abs_corr)

            if int(neuron_layer) > int(activation_layer):
                continue

            neuron_name = str(neuron_layer) + "_" + str(neuron)
            if neuron_name not in neuron_abs_corr_dict.keys():
                neuron_abs_corr_dict[neuron_name] = {}
                neuron_abs_corr_dict[neuron_name]['max_abs_corr'] = abs_corr
                neuron_abs_corr_dict[neuron_name]['corr'] = corr
                neuron_abs_corr_dict[neuron_name]['max_abs_corr_activation_layer'] = activation_layer

            else:
                if float(abs_corr) > float(neuron_abs_corr_dict[neuron_name]['max_abs_corr']):
                    neuron_abs_corr_dict[neuron_name]['max_abs_corr'] = abs_corr
                    neuron_abs_corr_dict[neuron_name]['corr'] = corr
                    neuron_abs_corr_dict[neuron_name]['max_abs_corr_activation_layer'] = activation_layer

        neuron_name_dict_sorted = sorted(neuron_abs_corr_dict.items(), key=lambda d: float(d[1]['max_abs_corr']),
                                         reverse=True)

        for item in neuron_name_dict_sorted:
            neuron_name = item[0].split('_')
            neuron_layer = neuron_name[0]
            neuron = neuron_name[1]
            abs_corr = item[1]['max_abs_corr']
            activation_layer = item[1]['max_abs_corr_activation_layer']
            corr = item[1]['corr']

            writer.writerow(
                (neuron_layer, neuron, activation_layer, neuron_weight, corr, abs_corr))

        avg = sum / idx
        print(neuron_weight + ' avg: ' + str(avg))


# get top k
def run_6():
    path_dir = '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/hizh/from_hizh/org-hizh_3200step_full/pearson'
    path_in = path_dir + '/max_sorted_in.csv'
    path_out = path_dir + '/max_sorted_out.csv'

    k = 100000
    neuron_dict = {}
    for path_idx, path in enumerate([path_in, path_out]):
        if path_idx == 0:
            neuron_weight = 'in'
        elif path_idx == 1:
            neuron_weight = 'out'

        f = open(path, 'r', encoding='utf-8')  # length 352257
        rows = csv.reader(f)

        for idx, row in enumerate(rows):
            if idx > k:
                break
            if idx == 0:
                print(row)
                continue

            neuron_layer = row[0]
            neuron = row[1]
            neuron_layer_name = neuron_layer + '_' + neuron_weight
            if neuron_layer_name not in neuron_dict.keys():
                neuron_dict[neuron_layer_name] = []
            neuron_dict[neuron_layer_name].append(int(neuron))

    new_dict = {}
    for igo in ['in', 'out']:
        for i in range(32):
            name = str(i) + "_" + igo
            if name not in neuron_dict.keys():
                new_dict[name] = []
            else:
                new_dict[name] = sorted(neuron_dict[name])

    with open(path_dir + '/' + str(k) + "x2/neuron_dict.json", "w") as f:
        json.dump(new_dict, f)
        print("加载入文件完成...")


# direct neuron, indirect neuron
def run_7():
    cos_neuron_path = '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_enzh_org/3200step/100000/neuron_dict_0.99847424.json'
    pearson_max_ranked_path_dir = '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/enzh_org-enzh_sn_3200_175000/pearson'
    indirect_neuron_w_path = pearson_max_ranked_path_dir + '/indirect_neuron.json'
    direct_neuron_w_path = pearson_max_ranked_path_dir + '/direct_neuron.json'

    in_path = pearson_max_ranked_path_dir + '/max_sorted_in.csv'
    out_path = pearson_max_ranked_path_dir + '/max_sorted_out.csv'
    f = open(cos_neuron_path, 'r', encoding='utf-8')
    neuron_dict = json.load(f)

    direct_neuron_dict = {}
    indirect_neuron_dict = {}
    for igo in ['in', 'out']:
        if igo == 'in':
            df = pd.read_csv(in_path)
            length = len(df)
            print(length)
        elif igo == 'out':
            df = pd.read_csv(out_path)
            length = len(df)
            print(length)

        for i in range(32):
            neuron_name = str(i) + '_' + igo
            if neuron_name in neuron_dict.keys():
                neuron_list = neuron_dict[neuron_name]
                for n in neuron_list:
                    neuron_df = df.loc[(df['neuron_layer'] == i) & (df['neuron'] == n)]
                    # print(neuron_df)
                    index = neuron_df.index
                    if len(index) > 1:
                        print('index length error')
                        break

                    if int(index[0]) > (length / 2):
                        if neuron_name not in indirect_neuron_dict.keys():
                            indirect_neuron_dict[neuron_name] = []

                        indirect_neuron_dict[neuron_name].append(n)

                    if int(index[0]) <= 50000:
                        if neuron_name not in direct_neuron_dict.keys():
                            direct_neuron_dict[neuron_name] = []

                        direct_neuron_dict[neuron_name].append(n)

    new_dict = {}
    for igo in ['in', 'out']:
        for i in range(32):
            name = str(i) + "_" + igo
            if name not in indirect_neuron_dict.keys():
                new_dict[name] = []
            else:
                new_dict[name] = sorted(indirect_neuron_dict[name])

    with open(indirect_neuron_w_path, "w") as f:
        json.dump(new_dict, f)
        print("加载入文件完成...")

    new_dict = {}

    for igo in ['in', 'out']:
        for i in range(32):
            name = str(i) + "_" + igo
            if name not in direct_neuron_dict.keys():
                new_dict[name] = []
            else:
                new_dict[name] = sorted(direct_neuron_dict[name])

    with open(direct_neuron_w_path, "w") as f:
        json.dump(new_dict, f)
        print("加载入文件完成...")


# diff rank between two cos-x
# 正的说明 导致这些导致排名上升，负的说明排名下降
def run_8():
    model_1_neuron_dict_path = '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/overlap/x-zh/200000_find_118103/overlap_neuron_dict.json'
    cos_neuron_x = model_1_neuron_dict_path.split('/')[-2]
    path_dir_1 = '/data2/haoyun/MA_project/neurons/pearson/frzh/enzh_200000_find_118103_/pearson'
    path_dir_2 = '/data2/haoyun/MA_project/neurons/pearson/frzh/frzh_200000_find_118103_/pearson'
    path_w = '/data2/haoyun/MA_project/neurons/rank_diff/frzh-frzh/enzh_200000_find_118103-frzh_200000_find_118103_/abs_rank_diff_larger_10W.csv'
    abs_rank_diff_larger_10W_neuron_dict_w_prefix = '/data2/haoyun/MA_project/neurons/rank_diff/frzh-frzh/enzh_200000_find_118103-frzh_200000_find_118103_'
    abs_rank_diff_larger_10W_neuron_dict_w_path = abs_rank_diff_larger_10W_neuron_dict_w_prefix + '/abs_rank_diff_larger_10W.json'

    abs_rank_diff_larger_10W_neuron_dict = {}
    abs_rank_diff_larger_10W_neuron_dict_rise = {}
    abs_rank_diff_larger_10W_neuron_dict_fall = {}

    in_path_1 = path_dir_1 + '/max_sorted_in.csv'
    out_path_1 = path_dir_1 + '/max_sorted_out.csv'
    in_path_2 = path_dir_2 + '/max_sorted_in.csv'
    out_path_2 = path_dir_2 + '/max_sorted_out.csv'

    w = open(path_w, 'w', encoding='utf-8')
    writer = csv.writer(w)
    writer.writerow(
        ('neuron_layer', 'neuron', 'activation_layer', 'neuron_weight', 'corr', 'abs_corr', 'rank_in_model1',
         'rank_in_model2', 'rank_diff', 'abs_rank_diff'))

    neuron_rank_diff_larger_10w = {}
    sum_all = 0
    for igo in ['in', 'out']:
        sum = 0
        if igo == 'in':
            path_1 = in_path_1
            path_2 = in_path_2
        elif igo == 'out':
            path_1 = out_path_1
            path_2 = out_path_2

        df_1 = pd.read_csv(path_1)
        df_2 = pd.read_csv(path_2)

        for index_1, row in df_1.iterrows():
            # print(index_1)
            neuron_layer = row['neuron_layer']
            neuron = row['neuron']
            activation_layer = row['activation_layer']
            neuron_weight = row['neuron_weight']
            corr = row['corr']
            abs_corr = row['abs_corr']

            neuron_df_2 = df_2.loc[(df_2['neuron_layer'] == neuron_layer) & (df_2['neuron'] == neuron)]
            index_2 = neuron_df_2.index[0]
            # print(index_2)

            rank_diff = index_2 - index_1
            # print(rank_diff)
            abs_rank_diff = abs(rank_diff)
            sum += abs_rank_diff
            sum_all += abs_rank_diff

            if abs_rank_diff >= 100000:
                neuron_name_for_dict = str(neuron_layer) + '_' + neuron_weight

                if neuron_name_for_dict not in abs_rank_diff_larger_10W_neuron_dict.keys():
                    abs_rank_diff_larger_10W_neuron_dict[neuron_name_for_dict] = []
                abs_rank_diff_larger_10W_neuron_dict[neuron_name_for_dict].append(int(neuron))

                if rank_diff > 0:
                    if neuron_name_for_dict not in abs_rank_diff_larger_10W_neuron_dict_rise.keys():
                        abs_rank_diff_larger_10W_neuron_dict_rise[neuron_name_for_dict] = []
                    abs_rank_diff_larger_10W_neuron_dict_rise[neuron_name_for_dict].append(int(neuron))
                else:
                    if neuron_name_for_dict not in abs_rank_diff_larger_10W_neuron_dict_fall.keys():
                        abs_rank_diff_larger_10W_neuron_dict_fall[neuron_name_for_dict] = []
                    abs_rank_diff_larger_10W_neuron_dict_fall[neuron_name_for_dict].append(int(neuron))

                neuron_name = str(neuron_layer) + '_' + str(neuron) + '_' + neuron_weight
                neuron_rank_diff_larger_10w[neuron_name] = {}
                neuron_rank_diff_larger_10w[neuron_name]['activation_layer'] = activation_layer
                neuron_rank_diff_larger_10w[neuron_name]['corr'] = corr
                neuron_rank_diff_larger_10w[neuron_name]['abs_corr'] = abs_corr
                neuron_rank_diff_larger_10w[neuron_name]['rank_in_model1'] = index_1
                neuron_rank_diff_larger_10w[neuron_name]['rank_in_model2'] = index_2
                neuron_rank_diff_larger_10w[neuron_name]['rank_diff'] = rank_diff
                neuron_rank_diff_larger_10w[neuron_name]['abs_rank_diff'] = abs_rank_diff

        avg = sum / (index_1 + 1)
        print(igo + ' avg: ' + str(avg))
        print(index_1 + 1)
        print(index_1 + 1)

    avg_all = sum_all / ((index_1 + 1) * 2)
    print("avg_all: " + str(avg_all))

    neuron_name_dict_sorted = sorted(neuron_rank_diff_larger_10w.items(), key=lambda d: float(d[1]['rank_diff']),
                                     reverse=True)

    for item in neuron_name_dict_sorted:
        neuron_name = item[0].split('_')
        neuron_layer = neuron_name[0]
        neuron = neuron_name[1]
        neuron_weight = neuron_name[2]
        activation_layer = item[1]['activation_layer']
        rank_in_model1 = item[1]['rank_in_model1']
        rank_in_model2 = item[1]['rank_in_model2']
        abs_corr = item[1]['abs_corr']
        corr = item[1]['corr']
        rank_diff = item[1]['rank_diff']
        abs_rank_diff = item[1]['abs_rank_diff']

        writer.writerow(
            (neuron_layer, neuron, activation_layer, neuron_weight, corr, abs_corr, rank_in_model1, rank_in_model2,
             rank_diff, abs_rank_diff))

    new_dict = {}
    for igo in ['in', 'out']:
        for i in range(32):
            name = str(i) + "_" + igo
            if name not in abs_rank_diff_larger_10W_neuron_dict.keys():
                new_dict[name] = []
            else:
                new_dict[name] = sorted(abs_rank_diff_larger_10W_neuron_dict[name])

    with open(abs_rank_diff_larger_10W_neuron_dict_w_path, "w") as f:
        json.dump(new_dict, f)
        print("加载入文件完成...")

    new_dict = {}
    for igo in ['in', 'out']:
        for i in range(32):
            name = str(i) + "_" + igo
            if name not in abs_rank_diff_larger_10W_neuron_dict_rise.keys():
                new_dict[name] = []
            else:
                new_dict[name] = sorted(abs_rank_diff_larger_10W_neuron_dict_rise[name])

    with open(abs_rank_diff_larger_10W_neuron_dict_w_prefix + '/abs_rank_diff_larger_10W_rise.json', "w") as f:
        json.dump(new_dict, f)
        print("加载入文件完成...")

    new_dict = {}
    for igo in ['in', 'out']:
        for i in range(32):
            name = str(i) + "_" + igo
            if name not in abs_rank_diff_larger_10W_neuron_dict_fall.keys():
                new_dict[name] = []
            else:
                new_dict[name] = sorted(abs_rank_diff_larger_10W_neuron_dict_fall[name])

    with open(abs_rank_diff_larger_10W_neuron_dict_w_prefix + '/abs_rank_diff_larger_10W_fall.json', "w") as f:
        json.dump(new_dict, f)
        print("加载入文件完成...")

    # overlap abs_rank_diff_larger_10W_neuron_dict & cos-model_1-neuron_dict
    two_neuron_dict_overlap([abs_rank_diff_larger_10W_neuron_dict_w_path, model_1_neuron_dict_path],
                            save_path=abs_rank_diff_larger_10W_neuron_dict_w_prefix + '/overlap_abs-rank-diff-larger-10W_cos-' + cos_neuron_x + '_neuron_dict.json')

    # disjoint abs_rank_diff_larger_10W_neuron_dict & cos-model_1-neuron_dict
    two_neuron_dict_disjoint([abs_rank_diff_larger_10W_neuron_dict_w_path, model_1_neuron_dict_path],
                             save_path=abs_rank_diff_larger_10W_neuron_dict_w_prefix + '/disjoint_abs-rank-diff-larger-10W_cos-' + cos_neuron_x + '_neuron_dict.json')


def run_8_2():
    model_1_neuron_dict_path = '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_enzh_org/3200step/100000/neuron_dict_0.99847424.json'
    cos_neuron_x = model_1_neuron_dict_path.split('/')[-2]
    abs_rank_diff_larger_10W_neuron_dict_w_prefix = '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/rank_diff/cos-100000_cos-50000'
    abs_rank_diff_larger_10W_neuron_dict_w_path = abs_rank_diff_larger_10W_neuron_dict_w_prefix + '/abs_rank_diff_larger_10W.json'

    # overlap abs_rank_diff_larger_10W_neuron_dict & cos-model_1-neuron_dict
    two_neuron_dict_overlap([abs_rank_diff_larger_10W_neuron_dict_w_path, model_1_neuron_dict_path],
                            save_path=abs_rank_diff_larger_10W_neuron_dict_w_prefix + '/overlap_abs-rank-diff-larger-10W_cos-' + cos_neuron_x + '_neuron_dict.json')

    # disjoint abs_rank_diff_larger_10W_neuron_dict & cos-model_1-neuron_dict
    two_neuron_dict_disjoint([abs_rank_diff_larger_10W_neuron_dict_w_path, model_1_neuron_dict_path],
                             save_path=abs_rank_diff_larger_10W_neuron_dict_w_prefix + '/disjoint_abs-rank-diff-larger-10W_cos-' + cos_neuron_x + '_neuron_dict.json')

def run_8_3():
    path_dir_1 = '/data2/haoyun/MA_project/LoRA_neurons/pearson/enzh/rank_256/pearson'
    path_dir_2 = '/data2/haoyun/MA_project/LoRA_neurons/pearson/enzh/rank_128/pearson'
    path_w = '/data2/haoyun/MA_project/LoRA_neurons/rank_diff/rank_256-rank_128/abs_rank_diff_larger_10W.csv'
    abs_rank_diff_larger_10W_neuron_dict_w_prefix = '/data2/haoyun/MA_project/LoRA_neurons/rank_diff/rank_256-rank_128'
    abs_rank_diff_larger_10W_neuron_dict_w_path = abs_rank_diff_larger_10W_neuron_dict_w_prefix + '/abs_rank_diff_larger_10W.json'

    abs_rank_diff_larger_10W_neuron_dict = {}
    abs_rank_diff_larger_10W_neuron_dict_rise = {}
    abs_rank_diff_larger_10W_neuron_dict_fall = {}

    in_path_1 = path_dir_1 + '/max_sorted_in.csv'
    out_path_1 = path_dir_1 + '/max_sorted_out.csv'
    in_path_2 = path_dir_2 + '/max_sorted_in.csv'
    out_path_2 = path_dir_2 + '/max_sorted_out.csv'

    w = open(path_w, 'w', encoding='utf-8')
    writer = csv.writer(w)
    writer.writerow(
        ('neuron_layer', 'neuron', 'activation_layer', 'neuron_weight', 'corr', 'abs_corr', 'rank_in_model1',
         'rank_in_model2', 'rank_diff', 'abs_rank_diff'))

    neuron_rank_diff_larger_10w = {}
    sum_all = 0
    for igo in ['in', 'out']:
        sum = 0
        if igo == 'in':
            path_1 = in_path_1
            path_2 = in_path_2
        elif igo == 'out':
            path_1 = out_path_1
            path_2 = out_path_2

        df_1 = pd.read_csv(path_1)
        df_2 = pd.read_csv(path_2)

        for index_1, row in df_1.iterrows():
            # print(index_1)
            neuron_layer = row['neuron_layer']
            neuron = row['neuron']
            activation_layer = row['activation_layer']
            neuron_weight = row['neuron_weight']
            corr = row['corr']
            abs_corr = row['abs_corr']

            neuron_df_2 = df_2.loc[(df_2['neuron_layer'] == neuron_layer) & (df_2['neuron'] == neuron)]
            index_2 = neuron_df_2.index[0]
            # print(index_2)

            rank_diff = index_2 - index_1
            # print(rank_diff)
            abs_rank_diff = abs(rank_diff)
            sum += abs_rank_diff
            sum_all += abs_rank_diff

            if abs_rank_diff >= 100000:
                neuron_name_for_dict = str(neuron_layer) + '_' + neuron_weight

                if neuron_name_for_dict not in abs_rank_diff_larger_10W_neuron_dict.keys():
                    abs_rank_diff_larger_10W_neuron_dict[neuron_name_for_dict] = []
                abs_rank_diff_larger_10W_neuron_dict[neuron_name_for_dict].append(int(neuron))

                if rank_diff > 0:
                    if neuron_name_for_dict not in abs_rank_diff_larger_10W_neuron_dict_rise.keys():
                        abs_rank_diff_larger_10W_neuron_dict_rise[neuron_name_for_dict] = []
                    abs_rank_diff_larger_10W_neuron_dict_rise[neuron_name_for_dict].append(int(neuron))
                else:
                    if neuron_name_for_dict not in abs_rank_diff_larger_10W_neuron_dict_fall.keys():
                        abs_rank_diff_larger_10W_neuron_dict_fall[neuron_name_for_dict] = []
                    abs_rank_diff_larger_10W_neuron_dict_fall[neuron_name_for_dict].append(int(neuron))

                neuron_name = str(neuron_layer) + '_' + str(neuron) + '_' + neuron_weight
                neuron_rank_diff_larger_10w[neuron_name] = {}
                neuron_rank_diff_larger_10w[neuron_name]['activation_layer'] = activation_layer
                neuron_rank_diff_larger_10w[neuron_name]['corr'] = corr
                neuron_rank_diff_larger_10w[neuron_name]['abs_corr'] = abs_corr
                neuron_rank_diff_larger_10w[neuron_name]['rank_in_model1'] = index_1
                neuron_rank_diff_larger_10w[neuron_name]['rank_in_model2'] = index_2
                neuron_rank_diff_larger_10w[neuron_name]['rank_diff'] = rank_diff
                neuron_rank_diff_larger_10w[neuron_name]['abs_rank_diff'] = abs_rank_diff

        avg = sum / (index_1 + 1)
        print(igo + ' avg: ' + str(avg))
        print(index_1 + 1)
        print(index_1 + 1)

    avg_all = sum_all / ((index_1 + 1) * 2)
    print("avg_all: " + str(avg_all))

    neuron_name_dict_sorted = sorted(neuron_rank_diff_larger_10w.items(), key=lambda d: float(d[1]['rank_diff']),
                                     reverse=True)

    for item in neuron_name_dict_sorted:
        neuron_name = item[0].split('_')
        neuron_layer = neuron_name[0]
        neuron = neuron_name[1]
        neuron_weight = neuron_name[2]
        activation_layer = item[1]['activation_layer']
        rank_in_model1 = item[1]['rank_in_model1']
        rank_in_model2 = item[1]['rank_in_model2']
        abs_corr = item[1]['abs_corr']
        corr = item[1]['corr']
        rank_diff = item[1]['rank_diff']
        abs_rank_diff = item[1]['abs_rank_diff']

        writer.writerow(
            (neuron_layer, neuron, activation_layer, neuron_weight, corr, abs_corr, rank_in_model1, rank_in_model2,
             rank_diff, abs_rank_diff))

    new_dict = {}
    for igo in ['in', 'out']:
        for i in range(32):
            name = str(i) + "_" + igo
            if name not in abs_rank_diff_larger_10W_neuron_dict.keys():
                new_dict[name] = []
            else:
                new_dict[name] = sorted(abs_rank_diff_larger_10W_neuron_dict[name])

    with open(abs_rank_diff_larger_10W_neuron_dict_w_path, "w") as f:
        json.dump(new_dict, f)
        print("加载入文件完成...")

    new_dict = {}
    for igo in ['in', 'out']:
        for i in range(32):
            name = str(i) + "_" + igo
            if name not in abs_rank_diff_larger_10W_neuron_dict_rise.keys():
                new_dict[name] = []
            else:
                new_dict[name] = sorted(abs_rank_diff_larger_10W_neuron_dict_rise[name])

    with open(abs_rank_diff_larger_10W_neuron_dict_w_prefix + '/abs_rank_diff_larger_10W_rise.json', "w") as f:
        json.dump(new_dict, f)
        print("加载入文件完成...")

    new_dict = {}
    for igo in ['in', 'out']:
        for i in range(32):
            name = str(i) + "_" + igo
            if name not in abs_rank_diff_larger_10W_neuron_dict_fall.keys():
                new_dict[name] = []
            else:
                new_dict[name] = sorted(abs_rank_diff_larger_10W_neuron_dict_fall[name])

    with open(abs_rank_diff_larger_10W_neuron_dict_w_prefix + '/abs_rank_diff_larger_10W_fall.json', "w") as f:
        json.dump(new_dict, f)
        print("加载入文件完成...")



# max sorted add rank_diff
def run_9():
    path_dir_1 = '/data2/haoyun/MA_project/neurons/pearson/enzh/frzh_100000_find_49305/pearson'
    path_dir_2 = '/data2/haoyun/MA_project/neurons/pearson/frzh/frzh_100000_find_49305/pearson'
    path_prefix = '/data2/haoyun/MA_project/neurons/rank_diff/frzh-frzh/enzh_200000_find_118103-frzh_200000_find_118103_/max_sorted_add_rank_diff'

    in_path_1 = path_dir_1 + '/max_sorted_in.csv'
    out_path_1 = path_dir_1 + '/max_sorted_out.csv'
    in_path_2 = path_dir_2 + '/max_sorted_in.csv'
    out_path_2 = path_dir_2 + '/max_sorted_out.csv'

    for igo in ['in', 'out']:
        # for igo in ['in']:
        add_neuron_rank_diff = {}
        sum = 0
        path_w = path_prefix + '/max_sorted_add_rank_diff_' + igo + '.csv'
        w = open(path_w, 'w', encoding='utf-8')
        writer = csv.writer(w)
        writer.writerow(
            ('neuron_layer', 'neuron', 'activation_layer', 'neuron_weight', 'corr', 'abs_corr', 'rank_in_model1',
             'rank_in_model2', 'rank_diff', 'abs_rank_diff'))

        if igo == 'in':
            path_1 = in_path_1
            path_2 = in_path_2
        elif igo == 'out':
            path_1 = out_path_1
            path_2 = out_path_2

        df_1 = pd.read_csv(path_1)
        df_2 = pd.read_csv(path_2)

        for index_1, row in df_1.iterrows():
            # print(index_1)
            neuron_layer = row['neuron_layer']
            neuron = row['neuron']
            activation_layer = row['activation_layer']
            neuron_weight = row['neuron_weight']
            corr = row['corr']
            abs_corr = row['abs_corr']

            neuron_df_2 = df_2.loc[(df_2['neuron_layer'] == neuron_layer) & (df_2['neuron'] == neuron)]
            index_2 = neuron_df_2.index[0]
            # print(index_2)

            rank_diff = index_2 - index_1
            # print(rank_diff)
            abs_rank_diff = abs(rank_diff)
            sum += abs_rank_diff

            neuron_name = str(neuron_layer) + '_' + str(neuron) + '_' + neuron_weight
            add_neuron_rank_diff[neuron_name] = {}
            add_neuron_rank_diff[neuron_name]['activation_layer'] = activation_layer
            add_neuron_rank_diff[neuron_name]['corr'] = corr
            add_neuron_rank_diff[neuron_name]['abs_corr'] = abs_corr
            add_neuron_rank_diff[neuron_name]['rank_in_model1'] = index_1
            add_neuron_rank_diff[neuron_name]['rank_in_model2'] = index_2
            add_neuron_rank_diff[neuron_name]['rank_diff'] = rank_diff
            add_neuron_rank_diff[neuron_name]['abs_rank_diff'] = abs_rank_diff

        avg = sum / (index_1 + 1)
        print(igo + ' avg: ' + str(avg))

        neuron_name_dict_sorted = sorted(add_neuron_rank_diff.items(), key=lambda d: float(d[1]['abs_corr']),
                                         reverse=True)

        for item in neuron_name_dict_sorted:
            neuron_name = item[0].split('_')
            neuron_layer = neuron_name[0]
            neuron = neuron_name[1]
            neuron_weight = neuron_name[2]
            activation_layer = item[1]['activation_layer']
            rank_in_model1 = item[1]['rank_in_model1']
            rank_in_model2 = item[1]['rank_in_model2']
            abs_corr = item[1]['abs_corr']
            corr = item[1]['corr']
            rank_diff = item[1]['rank_diff']
            abs_rank_diff = item[1]['abs_rank_diff']

            writer.writerow(
                (neuron_layer, neuron, activation_layer, neuron_weight, corr, abs_corr, rank_in_model1, rank_in_model2,
                 rank_diff, abs_rank_diff))


# max_sorted 每一块的 rank_diff 均值，可视化部分在 visualization
def run_10():
    path_dir = '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/rank_diff/cos-100000_cos-50000/max_sorted_add_rank_diff'
    path_in = path_dir + '/max_sorted_add_rank_diff_in.csv'
    path_out = path_dir + '/max_sorted_add_rank_diff_out.csv'
    # all length: 352256

    # end 有小数向前，start 有小数向后，
    # [0: 3522], [3523: 7045], [7046: 10567], [10568: 105676], [105677: 211353], [211354: 317030], [317031: 341688], [341689: 345210], [345211: 348733], [348734: 352256]
    slice_cut = [{'start': 0, 'end': 0.01}, {'start': 0.01, 'end': 0.02}, {'start': 0.02, 'end': 0.03},
                 {'start': 0.03, 'end': 0.3}, {'start': 0.3, 'end': 0.6}, {'start': 0.6, 'end': 0.9},
                 {'start': 0.9, 'end': 0.97}, {'start': 0.97, 'end': 0.98}, {'start': 0.98, 'end': 0.99},
                 {'start': 0.99, 'end': 1}]

    for igo in ['in', 'out']:
        if igo == 'in':
            path = path_in
        elif igo == 'out':
            path = path_out

        df = pd.read_csv(path)
        row_number = df.shape[0]
        print(row_number)
        for s in slice_cut:
            start_idx = math.ceil(row_number * s['start'])
            end_idx = math.floor(row_number * s['end'])
            print(start_idx)
            print(end_idx)
            df_s = df[start_idx: end_idx]

            abs_rank_diff_mean = df_s['abs_rank_diff'].mean()
            print(abs_rank_diff_mean)
            print('**********************')


# max_sorted 每一块的神经元 layer 占比，可视化部分在 visualization
def run_11():
    path_dir = '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/rank_diff/cos-100000_cos-50000/max_sorted_add_rank_diff'
    path_in = path_dir + '/max_sorted_add_rank_diff_in.csv'
    path_out = path_dir + '/max_sorted_add_rank_diff_out.csv'
    # all length: 352256

    # end 有小数向前，start 有小数向后，
    # [0: 3522], [3523: 7045], [7046: 10567], [10568: 105676], [105677: 211353], [211354: 317030], [317031: 341688], [341689: 345210], [345211: 348733], [348734: 352256]
    slice_cut = [{'start': 0, 'end': 0.01}, {'start': 0.01, 'end': 0.02}, {'start': 0.02, 'end': 0.03},
                 {'start': 0.03, 'end': 0.3}, {'start': 0.3, 'end': 0.6}, {'start': 0.6, 'end': 0.9},
                 {'start': 0.9, 'end': 0.97}, {'start': 0.97, 'end': 0.98}, {'start': 0.98, 'end': 0.99},
                 {'start': 0.99, 'end': 1}]

    for igo in ['in']:
        percentage_dict_list = []
        if igo == 'in':
            path = path_in
        elif igo == 'out':
            path = path_out

        df = pd.read_csv(path)
        row_number = df.shape[0]
        # print(row_number)
        for s in slice_cut:
            start_idx = math.ceil(row_number * s['start'])
            end_idx = math.floor(row_number * s['end'])
            print(start_idx)
            print(end_idx)
            df_s = df[start_idx: end_idx]

            abs_rank_diff_mean = df_s['abs_rank_diff'].mean()
            print(abs_rank_diff_mean)
            length = len(df_s)
            # print(length)

            layer_counts = df_s['neuron_layer'].value_counts()
            print(layer_counts)
            layer_counts = sorted(layer_counts.items(), key=lambda d: d[0], reverse=False)
            percentage_dict = {}
            for item in layer_counts:
                layer = item[0]
                number = item[1]

                percentage_dict[str(layer)] = number / length

            percentage_dict_list.append(percentage_dict)

            print(percentage_dict)

            print('**********************')


# get pearson rank last k
def run_12():
    path_dir = '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/enzh_org-enzh_3200step_full/pearson'
    path_in = path_dir + '/max_sorted_in.csv'
    path_out = path_dir + '/max_sorted_out.csv'

    k = 500
    neuron_dict = {}
    for path_idx, path in enumerate([path_in, path_out]):
        count = 0
        if path_idx == 0:
            neuron_weight = 'in'
        elif path_idx == 1:
            neuron_weight = 'out'

        f = open(path, 'r', encoding='utf-8')  # length 352257
        rows = csv.reader(f)

        thr_k = 352257 - k

        for idx, row in enumerate(rows):
            if idx < thr_k:
                continue
            if idx == 0:
                print(row)
                continue

            neuron_layer = row[0]
            neuron = row[1]
            neuron_layer_name = neuron_layer + '_' + neuron_weight
            if neuron_layer_name not in neuron_dict.keys():
                neuron_dict[neuron_layer_name] = []
            neuron_dict[neuron_layer_name].append(int(neuron))
            count += 1

        print(count)

    new_dict = {}
    for igo in ['in', 'out']:
        for i in range(32):
            name = str(i) + "_" + igo
            if name not in neuron_dict.keys():
                new_dict[name] = []
            else:
                new_dict[name] = sorted(neuron_dict[name])

    with open(path_dir + '/reversed/' + str(k) + "x2/neuron_dict.json", "w") as f:
        json.dump(new_dict, f)
        print("加载入文件完成...")


# 计算被训练的神经元的 avg_rank_diff
def run_13():
    max_sorted_add_rank_diff_dir_path = r'/data2/haoyun/MA_project/neurons/rank_diff/frzh-frzh/enzh_200000_find_118103-frzh_200000_find_118103_/max_sorted_add_rank_diff'
    max_sorted_add_rank_diff_in_path = max_sorted_add_rank_diff_dir_path + '/max_sorted_add_rank_diff_in.csv'
    max_sorted_add_rank_diff_out_path = max_sorted_add_rank_diff_dir_path + '/max_sorted_add_rank_diff_out.csv'
    neuron_dict_path = r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_enzh_org/3200step/150000/neuron_dict_0.9985444.json'

    df_in = pd.read_csv(max_sorted_add_rank_diff_in_path)
    df_out = pd.read_csv(max_sorted_add_rank_diff_out_path)

    f = open(neuron_dict_path, 'r', encoding='utf-8')
    neuron_dict = json.load(f)
    count = 0
    abs_rank_diff_all = 0
    for k, values in neuron_dict.items():
        print(k)
        if 'in' in k:
            neuron_layer = int(k.split('_')[0])
            for v in values:
                abs_rank_diff = df_in.loc[(df_in['neuron_layer'] == neuron_layer) & (df_in['neuron'] == int(v))][
                    'abs_rank_diff']
                abs_rank_diff_all += int(abs_rank_diff)
                count += 1

        elif 'out' in k:
            neuron_layer = int(k.split('_')[0])
            for v in values:
                abs_rank_diff = df_out.loc[(df_out['neuron_layer'] == neuron_layer) & (df_out['neuron'] == int(v))][
                    'abs_rank_diff']
                abs_rank_diff_all += int(abs_rank_diff)
                count += 1

    average = abs_rank_diff_all / count
    print(average)

#
def run_14():
    path_dir = '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/rank_diff/cos-100000_cos-50000/max_sorted_add_rank_diff'
    path_in = path_dir + '/max_sorted_add_rank_diff_in.csv'
    path_out = path_dir + '/max_sorted_add_rank_diff_out.csv'

    # all length: 352256

    # end 有小数向前，start 有小数向后，
    # [0: 3522], [3523: 7045], [7046: 10567], [10568: 105676], [105677: 211353], [211354: 317030], [317031: 341688], [341689: 345210], [345211: 348733], [348734: 352256]
    slice_cut = [{'start': 0, 'end': 0.01}, {'start': 0.01, 'end': 0.02}, {'start': 0.02, 'end': 0.03},
                 {'start': 0.03, 'end': 0.1}, {'start': 0.1, 'end': 0.2}, {'start': 0.2, 'end': 0.3},
                 {'start': 0.3, 'end': 0.4}, {'start': 0.4, 'end': 0.5}, {'start': 0.5, 'end': 0.6},
                 {'start': 0.6, 'end': 0.7}, {'start': 0.7, 'end': 0.8}, {'start': 0.8, 'end': 0.9},
                 {'start': 0.9, 'end': 0.97}, {'start': 0.97, 'end': 0.98}, {'start': 0.98, 'end': 0.99},
                 {'start': 0.99, 'end': 1}]

    # slice_cut = [{'start': 0, 'end': 0.01}]
    # slice_cut = [{'start': 0.99, 'end': 1}]

    df1 = pd.read_csv(path_in)
    df2 = pd.read_csv(path_out)

    df3 = pd.concat([df1, df2])
    df3 = df3.sort_values(by=['abs_corr'], ascending=False)
    abs_rank_diff_mean_all = df3['abs_rank_diff'].mean()
    print('abs_rank_diff_mean_all')
    print(abs_rank_diff_mean_all)

    row_number = df3.shape[0]
    print(row_number)
    for s in slice_cut:
        start_idx = math.ceil(row_number * s['start'])
        end_idx = math.floor(row_number * s['end'])
        print(start_idx)
        print(end_idx)
        w_path = path_dir + '/' + str(s['start']) + '-' + str(s['end']) + '_neuron_dict.json'
        df_s = df3[start_idx: end_idx]
        print(df_s)
        neuron_dict = {}
        for item in zip(df_s['neuron_layer'], df_s['neuron'], df_s['neuron_weight']):
            neuron_layer = item[0]
            neuron = int(item[1])
            neuron_weight = item[2]
            neuron_name = str(neuron_layer) + '_' + neuron_weight
            if neuron_name not in neuron_dict.keys():
                neuron_dict[neuron_name] = []
            neuron_dict[neuron_name].append(neuron)

        new_dict = {}
        for igo in ['in', 'out']:
            for i in range(32):
                name = str(i) + "_" + igo
                if name not in neuron_dict.keys():
                    new_dict[name] = []
                else:
                    new_dict[name] = sorted(neuron_dict[name])


        with open(w_path, "w") as f:
            json.dump(new_dict, f)
        print("加载入文件完成...")





# pearson index cos_score_and_rank
def run_15():
    rank_and_check(
        path_in=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/enzh_org-enzh_3200step_full/abs_diff_in.csv',
        path_out=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/enzh_org-enzh_3200step_full/abs_diff_out.csv',
        path_cos_in=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_enzh_org/3200step/cos_in.csv',
        path_cos_out=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_enzh_org/3200step/cos_out.csv')


if __name__ == '__main__':

    # run_1() #
    # run_2()
    # run_3()
    # run_4()
    # run_5() #
    # run_6()
    # run_7()
    # run_8() #
    # run_8_2() #
    run_8_3() #
    # run_9() #
    # run_10()
    # run_11()
    # run_12()
    # run_13()
    # run_14()

"""
python cal_correlation.py >log.out 2>&1 &



"""
