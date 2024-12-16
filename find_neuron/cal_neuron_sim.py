import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import csv
import json

from peft import PeftModel


def cal_model_weight_cos(model_path_1=r'/mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf',
                         model_path_2=r'/mnt/nfs/algo/intern/haoyunx11/models/sft/summary/enfr/train_enfr_org/checkpoint-250',
                         save_dir=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/summary_neurons_sim/enfr/org_allsft/250steps',
                         is_lora=False):
    from transformer_lens import HookedTransformer
    tokenizer = AutoTokenizer.from_pretrained(model_path_1)

    hf_model_1 = AutoModelForCausalLM.from_pretrained(model_path_1)
    model_1 = HookedTransformer.from_pretrained(r'meta-llama/Llama-3.2-1B', hf_model=hf_model_1, device="cpu",
                                                fold_ln=False,
                                                center_writing_weights=False, center_unembed=True,
                                                tokenizer=tokenizer)

    if is_lora:
        hf_model_2 = PeftModel.from_pretrained(hf_model_1, model_path_2, is_trainable=False)
        hf_model_2 = hf_model_2.merge_and_unload()
        model_2 = HookedTransformer.from_pretrained(r'meta-llama/Llama-3.2-1B', hf_model=hf_model_2, device="cpu",
                                                    fold_ln=False,
                                                    center_writing_weights=False, center_unembed=True,
                                                    tokenizer=tokenizer)
    else:
        hf_model_2 = AutoModelForCausalLM.from_pretrained(model_path_2)
        model_2 = HookedTransformer.from_pretrained(r'meta-llama/Llama-3.2-1B', hf_model=hf_model_2, device="cpu",
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
        if igo == 'in':
            path_tail = '/cos_in.csv'
            W_1 = W_in_1
            W_2 = W_in_2
        elif igo == 'out':
            path_tail = '/cos_out.csv'
            W_1 = W_out_1
            W_2 = W_out_2
        elif igo == 'gate':
            path_tail = '/cos_gate.csv'
            W_1 = W_gate_1
            W_2 = W_gate_2

        neuron_corr = {}
        for layer_idx, w_layers in enumerate(zip(W_1, W_2)):
            for neuron_idx, w in enumerate(zip(w_layers[0], w_layers[1])):
                score = cosine_similarity([w[0]], [w[1]])
                neuron_corr[(layer_idx, neuron_idx)] = score[0][0]

        corr_df = pd.DataFrame({'corr': pd.Series(neuron_corr)})
        corr_df.index.names = ['layer_idx', 'neuron_idx']
        corr_df = corr_df.reset_index()
        corr_df['abs_corr'] = np.abs(corr_df['corr'].values)

        corr_df.to_csv(save_dir + path_tail, index=False)

    # neuron_corr = {}
    # for layer_idx, w_layers in enumerate(zip(W_gate_1, W_gate_2)):
    #     for neuron_idx, w in enumerate(zip(w_layers[0], w_layers[1])):
    #         # print(w[0])
    #         # print(w[1])
    #         # print('---------------------------')
    #
    #         #     if neuron_idx > 6:
    #         #         break
    #         # break
    #
    #         score = cosine_similarity([w[0]], [w[1]])
    #
    #         if score < 0.9:
    #             print(layer_idx)
    #             print(neuron_idx)
    #             print('---------------------------')
    #
    #         neuron_corr[(layer_idx, neuron_idx)] = score[0][0]
    #
    # corr_df = pd.DataFrame({'corr': pd.Series(neuron_corr)})
    # corr_df.index.names = ['layer_idx', 'neuron_idx']
    # corr_df = corr_df.reset_index()
    # corr_df['abs_corr'] = np.abs(corr_df['corr'].values)
    #
    # corr_df.to_csv(
    #     '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/summary_neurons_sim/enfr/org_allsft/250steps/cos_gate.csv',
    #     index=False)
    #
    # neuron_corr = {}
    # for layer_idx, w_layers in enumerate(zip(W_out_1, W_out_2)):
    #     for neuron_idx, w in enumerate(zip(w_layers[0], w_layers[1])):
    #         # print(w[0])
    #         # print(w[1])
    #         # print('---------------------------')
    #
    #         #     if neuron_idx > 6:
    #         #         break
    #         # break
    #
    #         score = cosine_similarity([w[0]], [w[1]])
    #
    #         if score < 0.9:
    #             print(layer_idx)
    #             print(neuron_idx)
    #             print('---------------------------')
    #
    #         neuron_corr[(layer_idx, neuron_idx)] = score[0][0]
    #
    # corr_df = pd.DataFrame({'corr': pd.Series(neuron_corr)})
    # corr_df.index.names = ['layer_idx', 'neuron_idx']
    # corr_df = corr_df.reset_index()
    # corr_df['abs_corr'] = np.abs(corr_df['corr'].values)
    #
    # corr_df.to_csv(
    #     '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/summary_neurons_sim/enfr/org_allsft/250steps/cos_out.csv',
    #     index=False)


def cal_model_weight_cos_Mistral_7B_v1(model_path_1=r'/mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf',
                         model_path_2=r'/mnt/nfs/algo/intern/haoyunx11/models/sft/summary/enfr/train_enfr_org/checkpoint-250',
                         save_dir=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/summary_neurons_sim/enfr/org_allsft/250steps',
                         is_lora=False):
    from transformer_lens import HookedTransformer
    tokenizer = AutoTokenizer.from_pretrained(model_path_1)

    hf_model_1 = AutoModelForCausalLM.from_pretrained(model_path_1)
    model_1 = HookedTransformer.from_pretrained(r'mistralai/Mistral-7B-v0.1', hf_model=hf_model_1, device="cpu",
                                                fold_ln=False,
                                                center_writing_weights=False, center_unembed=True,
                                                tokenizer=tokenizer)

    if is_lora:
        hf_model_2 = PeftModel.from_pretrained(hf_model_1, model_path_2, is_trainable=False)
        hf_model_2 = hf_model_2.merge_and_unload()
        model_2 = HookedTransformer.from_pretrained(r'mistralai/Mistral-7B-v0.1', hf_model=hf_model_2, device="cpu",
                                                    fold_ln=False,
                                                    center_writing_weights=False, center_unembed=True,
                                                    tokenizer=tokenizer)
    else:
        hf_model_2 = AutoModelForCausalLM.from_pretrained(model_path_2)
        model_2 = HookedTransformer.from_pretrained(r'mistralai/Mistral-7B-v0.1', hf_model=hf_model_2, device="cpu",
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
        if igo == 'in':
            path_tail = '/cos_in.csv'
            W_1 = W_in_1
            W_2 = W_in_2
        elif igo == 'out':
            path_tail = '/cos_out.csv'
            W_1 = W_out_1
            W_2 = W_out_2
        elif igo == 'gate':
            path_tail = '/cos_gate.csv'
            W_1 = W_gate_1
            W_2 = W_gate_2

        neuron_corr = {}
        for layer_idx, w_layers in enumerate(zip(W_1, W_2)):
            for neuron_idx, w in enumerate(zip(w_layers[0], w_layers[1])):
                score = cosine_similarity([w[0]], [w[1]])
                neuron_corr[(layer_idx, neuron_idx)] = score[0][0]

        corr_df = pd.DataFrame({'corr': pd.Series(neuron_corr)})
        corr_df.index.names = ['layer_idx', 'neuron_idx']
        corr_df = corr_df.reset_index()
        corr_df['abs_corr'] = np.abs(corr_df['corr'].values)

        corr_df.to_csv(save_dir + path_tail, index=False)

    # neuron_corr = {}
    # for layer_idx, w_layers in enumerate(zip(W_gate_1, W_gate_2)):
    #     for neuron_idx, w in enumerate(zip(w_layers[0], w_layers[1])):
    #         # print(w[0])
    #         # print(w[1])
    #         # print('---------------------------')
    #
    #         #     if neuron_idx > 6:
    #         #         break
    #         # break
    #
    #         score = cosine_similarity([w[0]], [w[1]])
    #
    #         if score < 0.9:
    #             print(layer_idx)
    #             print(neuron_idx)
    #             print('---------------------------')
    #
    #         neuron_corr[(layer_idx, neuron_idx)] = score[0][0]
    #
    # corr_df = pd.DataFrame({'corr': pd.Series(neuron_corr)})
    # corr_df.index.names = ['layer_idx', 'neuron_idx']
    # corr_df = corr_df.reset_index()
    # corr_df['abs_corr'] = np.abs(corr_df['corr'].values)
    #
    # corr_df.to_csv(
    #     '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/summary_neurons_sim/enfr/org_allsft/250steps/cos_gate.csv',
    #     index=False)
    #
    # neuron_corr = {}
    # for layer_idx, w_layers in enumerate(zip(W_out_1, W_out_2)):
    #     for neuron_idx, w in enumerate(zip(w_layers[0], w_layers[1])):
    #         # print(w[0])
    #         # print(w[1])
    #         # print('---------------------------')
    #
    #         #     if neuron_idx > 6:
    #         #         break
    #         # break
    #
    #         score = cosine_similarity([w[0]], [w[1]])
    #
    #         if score < 0.9:
    #             print(layer_idx)
    #             print(neuron_idx)
    #             print('---------------------------')
    #
    #         neuron_corr[(layer_idx, neuron_idx)] = score[0][0]
    #
    # corr_df = pd.DataFrame({'corr': pd.Series(neuron_corr)})
    # corr_df.index.names = ['layer_idx', 'neuron_idx']
    # corr_df = corr_df.reset_index()
    # corr_df['abs_corr'] = np.abs(corr_df['corr'].values)
    #
    # corr_df.to_csv(
    #     '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/summary_neurons_sim/enfr/org_allsft/250steps/cos_out.csv',
    #     index=False)

def cal_model_weight_cos_on_select_neuron(
        model_path_1=r'/mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf',
        model_path_2=r'/mnt/nfs/algo/intern/haoyunx11/models/sft/summary/enfr/train_enfr_org/checkpoint-250',
        save_dir=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/overlap/x-zh/100000_find_49305/cos_on_select_neuron',
        neuron_dict_path='/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/overlap/x-zh/100000_find_49305/overlap_neuron_dict.json',
        is_lora=False):
    from transformer_lens import HookedTransformer
    tokenizer = AutoTokenizer.from_pretrained(model_path_1)

    hf_model_1 = AutoModelForCausalLM.from_pretrained(model_path_1)
    model_1 = HookedTransformer.from_pretrained(r'Llama-2-7b-chat-hf', hf_model=hf_model_1, device="cpu",
                                                fold_ln=False,
                                                center_writing_weights=False, center_unembed=True,
                                                tokenizer=tokenizer)

    if is_lora:
        hf_model_2 = PeftModel.from_pretrained(hf_model_1, model_path_2, is_trainable=False)
        hf_model_2 = hf_model_2.merge_and_unload()
        model_2 = HookedTransformer.from_pretrained(r'Llama-2-7b-chat-hf', hf_model=hf_model_2, device="cpu",
                                                    fold_ln=False,
                                                    center_writing_weights=False, center_unembed=True,
                                                    tokenizer=tokenizer)
    else:
        hf_model_2 = AutoModelForCausalLM.from_pretrained(model_path_2)
        model_2 = HookedTransformer.from_pretrained(r'Llama-2-7b-chat-hf', hf_model=hf_model_2, device="cpu",
                                                    fold_ln=False,
                                                    center_writing_weights=False, center_unembed=True,
                                                    tokenizer=tokenizer)

    # if neuron_dict_path != '':
    f = open(neuron_dict_path, 'r', encoding='utf-8')
    neuron_dict = json.load(f)

    W_in_1 = model_1.W_in
    W_in_1 = W_in_1.swapaxes(1, 2).detach().to(torch.float).cpu().numpy()
    print(W_in_1.shape)  # (11008, 4096)

    W_out_1 = model_1.W_out.detach().to(torch.float).cpu().numpy()
    print(W_out_1.shape)

    W_in_2 = model_2.W_in
    W_in_2 = W_in_2.swapaxes(1, 2).detach().to(torch.float).cpu().numpy()

    W_out_2 = model_2.W_out.detach().to(torch.float).cpu().numpy()

    igo_list = ['in', 'out']

    cos_sum = 0
    count = 0
    cos_max = float('-inf')
    cos_min = float('inf')

    for igo in igo_list:
        if igo == 'in':
            path_tail = '/cos_in.csv'
            W_1 = W_in_1
            W_2 = W_in_2
        elif igo == 'out':
            path_tail = '/cos_out.csv'
            W_1 = W_out_1
            W_2 = W_out_2

        neuron_corr = {}
        for layer_idx, w_layers in enumerate(zip(W_1, W_2)):

            # if neuron_dict_path != '':
            neuron_dict_key = str(layer_idx) + '_' + igo
            if neuron_dict_key not in neuron_dict.keys():
                continue

            for neuron_idx, w in enumerate(zip(w_layers[0], w_layers[1])):
                # print(layer_idx)
                # print(neuron_idx)
                # if neuron_dict_path != '':
                if neuron_idx not in neuron_dict[neuron_dict_key]:
                    continue

                score = cosine_similarity([w[0]], [w[1]])
                if cos_max < score:
                    cos_max = score

                if cos_min > score:
                    if score == 0:
                        print(layer_idx)
                        print(neuron_idx)
                        break
                    cos_min = score

                cos_sum += score
                count += 1
                neuron_corr[(layer_idx, neuron_idx)] = score[0][0]

        corr_df = pd.DataFrame({'corr': pd.Series(neuron_corr)})
        corr_df.index.names = ['layer_idx', 'neuron_idx']
        corr_df = corr_df.reset_index()
        corr_df['abs_corr'] = np.abs(corr_df['corr'].values)

        corr_df.to_csv(save_dir + path_tail, index=False)

    cos_avg = cos_sum / count
    print('cos avg: ' + str(cos_avg))
    print('cos max: ' + str(cos_max))
    print('cos min: ' + str(cos_min))


def cal_model_weight_cos_with_self_layers(
        model_path=r'/mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf'):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    hf_model = AutoModelForCausalLM.from_pretrained(model_path)
    model = HookedTransformer.from_pretrained(r'Llama-2-7b-chat-hf', hf_model=hf_model, device="cpu",
                                              fold_ln=False,
                                              center_writing_weights=False, center_unembed=True,
                                              tokenizer=tokenizer)

    W_in = model.W_in
    W_in = W_in.swapaxes(1, 2).detach().to(torch.float).cpu().numpy()
    print(W_in.shape)

    W_out = model.W_out.detach().to(torch.float).cpu().numpy()
    print(W_out.shape)

    # prepare layer weight mean aggregrate

    W_dict = {'W_in': W_in, 'W_out': W_out}

    for k, W in W_dict.items():

        layer_weight_mean_list = []
        for layer_idx, w_layers in enumerate(W):
            w_layer = np.mean(w_layers, axis=0, keepdims=True)
            # print(w_layer.shape)
            layer_weight_mean_list.append(w_layer)

        neuron_corr = {}
        for layer_idx, w_layers in enumerate(W):
            for neuron_idx, w in enumerate(w_layers):
                for layer_mean_idx, layer_weight_mean in enumerate(layer_weight_mean_list):
                    score = cosine_similarity([w], layer_weight_mean)

                    # if score < 0.9:
                    #     print(layer_idx)
                    #     print(neuron_idx)
                    #     print('---------------------------')

                    neuron_corr[(layer_idx, neuron_idx, layer_mean_idx)] = score[0][0]

        corr_df = pd.DataFrame({'corr': pd.Series(neuron_corr)})
        corr_df.index.names = ['layer_idx', 'neuron_idx', 'layer_mean_idx']
        corr_df = corr_df.reset_index()
        corr_df['abs_corr'] = np.abs(corr_df['corr'].values)

        if k == 'W_in':
            path_tail = 'cos_in.csv'
        elif k == 'W_out':
            path_tail = 'cos_out.csv'

        path = '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/self_layer_weight_mean_sim/org/' + path_tail
        corr_df.to_csv(path, index=False)


"""
2500steps
cos_in
avg: 0.9995646243624559
count: 65578

cos_out
avg: 0.9995627778714731
count: 69471
---------------------------

500steps
cos_in
avg: 0.999777701526417
count: 67664

cos_out
avg: 0.9997769582582584
count: 68569
---------------------------

"""


def cos_score_mean(
        path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/org_allsft/2500steps/cos_in.csv',
        threhold=0.9985):
    f = open(path, 'r', encoding='utf-8')
    rows = csv.reader(f)
    sum = 0
    count = 0
    for idx, row in enumerate(rows):
        if idx == 0:
            continue

        if float(row[2]) < threhold:
            count += 1
        # if float(row[2]) < 0.9995:
        #     count += 1

        sum += float(row[2])
    avg = sum / idx
    print(avg)
    print(count)


def get_threhold_score(
        path_1=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/org_allsft/2500steps/cos_in.csv',
        path_2=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/org_allsft/2500steps/cos_out.csv',
        neuron_number=100000):
    f_1 = open(path_1, 'r', encoding='utf-8')
    rows_1 = csv.reader(f_1)

    f_2 = open(path_2, 'r', encoding='utf-8')
    rows_2 = csv.reader(f_2)

    sum = 0
    score_list = []
    for idx1, row in enumerate(rows_1):
        if idx1 == 0:
            continue
        score = float(row[2])
        sum += score
        score_list.append(score)

    for idx2, row in enumerate(rows_2):
        if idx2 == 0:
            continue
        score = float(row[2])
        sum += score
        score_list.append(score)

    score_list.sort()
    # score_list.sort(reverse=True)

    # avg = sum / (idx1 + idx2)

    # print(score_list[:neuron_number])
    # print(avg)
    # print(count)

    print(score_list[:neuron_number][-1])

    return score_list[:neuron_number][-1]


def get_threhold_score_add_gate(
        path_1=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/org_allsft/2500steps/cos_in.csv',
        path_2=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/org_allsft/2500steps/cos_out.csv',
        path_3=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/org_allsft/2500steps/cos_gate.csv',
        neuron_number=100000):
    f_1 = open(path_1, 'r', encoding='utf-8')
    rows_1 = csv.reader(f_1)

    f_2 = open(path_2, 'r', encoding='utf-8')
    rows_2 = csv.reader(f_2)

    f_3 = open(path_3, 'r', encoding='utf-8')
    rows_3 = csv.reader(f_3)

    sum = 0
    score_list = []
    for idx1, row in enumerate(rows_1):
        if idx1 == 0:
            continue
        score = float(row[2])
        sum += score
        score_list.append(score)

    for idx2, row in enumerate(rows_2):
        if idx2 == 0:
            continue
        score = float(row[2])
        sum += score
        score_list.append(score)

    for idx3, row in enumerate(rows_3):
        if idx3 == 0:
            continue
        score = float(row[2])
        sum += score
        score_list.append(score)

    score_list.sort()

    print(score_list[:neuron_number][-1])

    return score_list[:neuron_number][-1]


def get_neuron(
        path_dir=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/org_allsft/500steps',
        threhold=0.9995, neuron_number=100000):
    cos_in_path = path_dir + '/cos_in.csv'
    # cos_gate_path = path_dir + '/cos_gate.csv'
    cos_out_path = path_dir + '/cos_out.csv'
    count = 0

    neuron_dict = {}
    for path_idx, path in enumerate([cos_in_path, cos_out_path]):
        f = open(path, 'r', encoding='utf-8')
        rows = csv.reader(f)
        if path_idx == 0:
            igo = 'in'
        elif path_idx == 1:
            igo = 'out'

        for idx, row in enumerate(rows):
            if idx == 0:
                continue

            neuron_layer = int(row[0])
            neuron = int(row[1])
            score = float(row[2])

            # if score < 0.9997438:
            if score < threhold:
                # if score < 0.9995:
                count += 1
                layer_name = str(neuron_layer) + '_' + igo
                if layer_name not in neuron_dict.keys():
                    neuron_dict[layer_name] = [neuron]
                else:
                    neuron_dict[layer_name].append(neuron)

    with open(path_dir + '/' + str(neuron_number) + "/neuron_dict_" + str(threhold) + ".json", "w") as f:
        json.dump(neuron_dict, f)
    print("加载入文件完成...")
    print(count)


def get_neuron_add_gate(
        path_dir=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/org_allsft/500steps',
        threhold=0.9995, neuron_number=100000):
    cos_in_path = path_dir + '/cos_in.csv'
    cos_gate_path = path_dir + '/cos_gate.csv'
    cos_out_path = path_dir + '/cos_out.csv'
    count = 0

    neuron_dict = {}
    for path_idx, path in enumerate([cos_in_path, cos_out_path, cos_gate_path]):
        f = open(path, 'r', encoding='utf-8')
        rows = csv.reader(f)
        if path_idx == 0:
            igo = 'in'
        elif path_idx == 1:
            igo = 'out'
        elif path_idx == 2:
            igo = 'gate'

        for idx, row in enumerate(rows):
            if idx == 0:
                continue

            neuron_layer = int(row[0])
            neuron = int(row[1])
            score = float(row[2])

            # if score < 0.9997438:
            if score < threhold:
                # if score < 0.9995:
                count += 1
                layer_name = str(neuron_layer) + '_' + igo
                if layer_name not in neuron_dict.keys():
                    neuron_dict[layer_name] = [neuron]
                else:
                    neuron_dict[layer_name].append(neuron)

    with open(path_dir + '/' + str(neuron_number) + "/neuron_dict_" + str(threhold) + ".json", "w") as f:
        json.dump(neuron_dict, f)
    print("加载入文件完成...")


def make_mask_with_dict(
        path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/org_allsft/2500steps/neuron_dict.json'):
    f = open(path, 'r', encoding='utf-8')
    neuron_dict = json.load(f)
    mask_dict = {}

    for layer_idx in range(32):
        print(layer_idx)
        for igo in ['in', 'gate', 'out']:
            layer_name = str(layer_idx) + '_' + igo
            mask_dict[layer_name] = torch.zeros(11008 * 4096)
            if igo == 'in' or igo == 'gate':
                for neuron in neuron_dict[layer_name]:
                    mask_dict[layer_name][neuron * 4096: (neuron + 1) * 4096] = 1
            else:
                for neuron in neuron_dict[layer_name]:
                    for i in range(4096):
                        index = i * 11008 + neuron
                        mask_dict[layer_name][index] = 1
    return mask_dict


def make_mask_with_dict_and_save_for_offload(
        path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/org_allsft/2500steps/neuron_dict.json',
        save_path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/org_allsft/2500steps/mask_cos_for_offload.pt'):
    f = open(path, 'r', encoding='utf-8')
    neuron_dict = json.load(f)
    mask_dict = {}

    for layer_idx in range(32):
        # print(layer_idx)
        for igo in ['in', 'out']:
            layer_name = str(layer_idx) + '_' + igo
            # print(layer_name)

            if igo == 'in' or igo == 'gate':
                mask_dict[layer_name] = torch.zeros(11008, 4096)
                # print(mask_dict[layer_name].shape)
                for neuron in neuron_dict[layer_name]:
                    mask_dict[layer_name][neuron][:] = 1
            else:
                mask_dict[layer_name] = torch.zeros(4096, 11008)
                # print(mask_dict[layer_name].shape)

                for neuron in neuron_dict[layer_name]:
                    # print(neuron)
                    # print(layer_name)
                    # print(len(mask_dict[layer_name][0]))
                    for i in range(4096):
                        mask_dict[layer_name][i][neuron] = 1
                    # print(mask_dict[layer_name].shape)

    torch.save(mask_dict, save_path)

    return mask_dict


def make_mask_with_dict_for_offload(
        path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/org_allsft/2500steps/neuron_dict.json',
        save_path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/org_allsft/2500steps/mask_cos_for_offload.pt'):
    f = open(path, 'r', encoding='utf-8')
    neuron_dict = json.load(f)
    mask_dict = {}

    for layer_idx in range(32):
        # print(layer_idx)
        for igo in ['in', 'out']:
            layer_name = str(layer_idx) + '_' + igo
            # print(layer_name)

            if igo == 'in' or igo == 'gate':
                mask_dict[layer_name] = torch.zeros(11008, 4096)
                # print(mask_dict[layer_name].shape)
                for neuron in neuron_dict[layer_name]:
                    mask_dict[layer_name][neuron][:] = 1
            else:
                mask_dict[layer_name] = torch.zeros(4096, 11008)
                # print(mask_dict[layer_name].shape)

                for neuron in neuron_dict[layer_name]:
                    # print(neuron)
                    # print(layer_name)
                    # print(len(mask_dict[layer_name][0]))
                    for i in range(4096):
                        mask_dict[layer_name][i][neuron] = 1
                    # print(mask_dict[layer_name].shape)
    return mask_dict


def make_mask_with_dict_and_save(
        path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/alpaca/base_full/39000step/150000/neuron_dict_0.98869854.json',
        save_path=r"/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/alpaca/base_full/39000step/150000/mask_cos.pt"):
    f = open(path, 'r', encoding='utf-8')
    neuron_dict = json.load(f)
    mask_dict = {}

    for layer_idx in range(16):
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

def make_mask_with_dict_and_save_llama3_2_1B_neuron(
        path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/alpaca/base_full/39000step/150000/neuron_dict_0.98869854.json',
        save_path=r"/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/alpaca/base_full/39000step/150000/mask_cos.pt"):
    f = open(path, 'r', encoding='utf-8')
    neuron_dict = json.load(f)
    mask_dict = {}

    for layer_idx in range(16):
        print(layer_idx)
        for igo in ['in', 'out']:
            layer_name = str(layer_idx) + '_' + igo
            mask_dict[layer_name] = torch.zeros(8192 * 2048)
            if layer_name not in neuron_dict.keys():
                continue
            if igo == 'in' or igo == 'gate':
                for neuron in neuron_dict[layer_name]:
                    mask_dict[layer_name][neuron * 2048: (neuron + 1) * 2048] = 1
            else:
                for neuron in neuron_dict[layer_name]:
                    for i in range(2048):
                        index = i * 8192 + neuron
                        mask_dict[layer_name][index] = 1

    torch.save(mask_dict, save_path)

    return mask_dict

def make_mask_with_dict_and_save_Mistral_7B_v1_neuron(
        path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/alpaca/base_full/39000step/150000/neuron_dict_0.98869854.json',
        save_path=r"/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/alpaca/base_full/39000step/150000/mask_cos.pt"):
    f = open(path, 'r', encoding='utf-8')
    neuron_dict = json.load(f)
    mask_dict = {}

    for layer_idx in range(32):
        print(layer_idx)
        for igo in ['in', 'out']:
            layer_name = str(layer_idx) + '_' + igo
            mask_dict[layer_name] = torch.zeros(14336 * 4096)
            if layer_name not in neuron_dict.keys():
                continue
            if igo == 'in' or igo == 'gate':
                for neuron in neuron_dict[layer_name]:
                    mask_dict[layer_name][neuron * 4096: (neuron + 1) * 4096] = 1
            else:
                for neuron in neuron_dict[layer_name]:
                    for i in range(4096):
                        index = i * 14336 + neuron
                        mask_dict[layer_name][index] = 1

    torch.save(mask_dict, save_path)

    return mask_dict


def make_mask_with_dict_and_save_add_gate(
        path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/org_allsft/500steps/neuron_dict.json',
        save_path=r"/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/org_allsft/500steps/mask_cos.pt"):
    f = open(path, 'r', encoding='utf-8')
    neuron_dict = json.load(f)
    mask_dict = {}

    for layer_idx in range(32):
        print(layer_idx)
        for igo in ['in', 'out', 'gate']:
            layer_name = str(layer_idx) + '_' + igo
            mask_dict[layer_name] = torch.zeros(11008 * 4096)
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


def load_mask_dict(
        path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/org_allsft/2500steps/mask_cos_for_offload.pt'):
    mask_dict = torch.load(path)

    for k, v in mask_dict.items():
        if k == '0_in':
            print(k)
            for i in range(6):
                print(i)
                print(v[i])
                print(v[i].shape)
                print('*******************')

            for i in range(11000, 11008):
                print(i)
                print(v[i])
                print(v[i].shape)
                print('*******************')


"""
len_500steps: 136233
len_2500steps: 135049
len_overlap:100450




"""


def cal_cos_overlap(
        path_1=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/org_allsft/500steps/neuron_dict.json',
        path_2=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/org_allsft/2500steps/neuron_dict.json'):
    f_1 = open(path_1, 'r', encoding='utf-8')
    data_1 = json.load(f_1)

    f_2 = open(path_2, 'r', encoding='utf-8')
    data_2 = json.load(f_2)

    data_1_len = 0
    data_2_len = 0
    overlap_len = 0

    for idx, line in enumerate(zip(data_1.items(), data_2.items())):
        k_1 = line[0][0]
        k_2 = line[1][0]

        if 'gate' in k_1:
            continue

        data_list_1 = line[0][1]
        data_list_2 = line[1][1]

        res = list(set(data_list_1) & set(data_list_2))

        data_1_len += len(data_list_1)
        data_2_len += len(data_list_2)
        overlap_len += len(res)

        print(k_1)

        print('500steps: ' + str(len(data_list_1)))
        print('2500steps: ' + str(len(data_list_2)))
        print('overlap: ' + str(len(res)))
        print('------------------------------')

    print(data_1_len)
    print(data_2_len)
    print(overlap_len)


# Ratio of front and rear layers
def get_Ratio(
        path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_enzh_org/3200step/100000/neuron_dict_0.99847424.json'):
    f = open(path, 'r', encoding='utf-8')
    neuron_dict = json.load(f)
    each_layer_in_number = []
    each_layer_in_ratio = {}
    each_layer_out_number = []
    each_layer_out_ratio = {}
    in_all_number = 0
    out_all_number = 0
    for item in neuron_dict.items():
        if 'in' in item[0]:
            length = len(item[1])
            in_all_number += length
            each_layer_in_number.append(length)
        elif 'out' in item[0]:
            length = len(item[1])
            out_all_number += length
            each_layer_out_number.append(length)

    in_out_all = in_all_number + out_all_number
    in_out_all_ratio = in_all_number / in_out_all
    print(in_out_all_ratio)
    print('**************************************')

    max_ = -1

    for idx, numbers in enumerate(zip(each_layer_in_number, each_layer_out_number)):
        # each in vs all in
        print(str(idx) + '_' + 'in')
        ratio = numbers[0] / in_all_number
        print(ratio)

        # each out vs all out
        print(str(idx) + '_' + 'out')
        ratio = numbers[1] / out_all_number
        print(ratio)

        # each all vs all
        each_all = numbers[0] + numbers[1]
        ratio_all = each_all / in_out_all
        print(ratio_all)

        if max_ < ratio_all:
            max_ = ratio_all

    print('Max')
    print(max_)


def count_igo(
        path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_enzh_org/3200step/add_gate/100000/neuron_dict_0.998407.json'):
    f = open(path, 'r', encoding='utf-8')

    in_all_number = 0
    out_all_number = 0
    gate_all_number = 0

    neuron_dict = json.load(f)
    for item in neuron_dict.items():
        if 'in' in item[0]:
            length = len(item[1])
            in_all_number += length
        elif 'out' in item[0]:
            length = len(item[1])
            out_all_number += length
        elif 'gate' in item[0]:
            length = len(item[1])
            gate_all_number += length

    print(in_all_number)
    print(out_all_number)
    print(gate_all_number)


def compare_full_sft_and_neuron_train_cos_score_on_sn(cos_path1, cos_path2, neuron_dict_path):
    f_1 = open(cos_path1, 'r', encoding='utf-8')
    f_2 = open(cos_path2, 'r', encoding='utf-8')
    f_neuron_dict = open(neuron_dict_path, 'r', encoding='utf-8')
    neuron_dict = json.load(f_neuron_dict)

    rows_1 = csv.reader(f_1)
    rows_2 = csv.reader(f_2)

    count_diff_large = 0
    count_diff_minus = 0

    for idx, row in enumerate(zip(rows_1, rows_2)):
        if idx == 0:
            continue
        layer_idx = row[0][0]
        neuron_idx = int(row[0][1])
        neurom_name = str(layer_idx) + '_' + str(neuron_idx)
        dict_key = str(layer_idx) + '_in'

        if neuron_idx in neuron_dict[dict_key]:
            abs_corr_1 = row[0][3]
            abs_corr_2 = row[1][3]

            diff = float(abs_corr_1) - float(abs_corr_2)
            abs_diff = abs(diff)

            if abs_diff >= 0.01:
                print(neurom_name)
                print(abs_corr_1)
                print(abs_corr_2)
                count_diff_large += 1
                print('*' * 20)

            if diff <= 0:
                print(neurom_name)
                print(abs_corr_1)
                print(abs_corr_2)
                count_diff_minus += 1
                print('-' * 20)
        print('+++++++++++++++++++++++++++')

    print(count_diff_large)
    print(count_diff_minus)


def check_neuron(model_path_1, model_path_2):
    from transformer_lens import HookedTransformer
    tokenizer = AutoTokenizer.from_pretrained(model_path_1)

    hf_model_1 = AutoModelForCausalLM.from_pretrained(model_path_1)
    model_1 = HookedTransformer.from_pretrained(r'meta-llama/Llama-3.2-1B', hf_model=hf_model_1, device="cpu",
                                                fold_ln=False,
                                                center_writing_weights=False, center_unembed=True,
                                                tokenizer=tokenizer)

    hf_model_2 = AutoModelForCausalLM.from_pretrained(model_path_2)
    model_2 = HookedTransformer.from_pretrained(r'meta-llama/Llama-3.2-1B', hf_model=hf_model_2, device="cpu",
                                                fold_ln=False,
                                                center_writing_weights=False, center_unembed=True,
                                                tokenizer=tokenizer)

    w_1 = model_1.W_E
    w_2 = model_2.W_E

    print((w_1 == w_2).all())

    w_1 = model_1.W_K
    w_2 = model_2.W_K

    print((w_1 == w_2).all())

    w_1 = model_1.W_O
    w_2 = model_2.W_O

    print((w_1 == w_2).all())

    w_1 = model_1.W_Q
    w_2 = model_2.W_Q

    print((w_1 == w_2).all())

    w_1 = model_1.W_V
    w_2 = model_2.W_V

    print((w_1 == w_2).all())

    w_1 = model_1.W_gate
    w_2 = model_2.W_gate

    print((w_1 == w_2).all())

    # w_1 = model_1.W_pos
    # w_2 = model_2.W_pos
    #
    # print((w_1 == w_2).all())

    print('------------------')

    w_1 = model_1.W_in
    w_2 = model_2.W_in

    print((w_1 == w_2).all())

    w_1 = model_1.W_out
    w_2 = model_2.W_out

    print((w_1 == w_2).all())

    # W_in_1 = model_1.W_in
    # W_in_1 = W_in_1.swapaxes(1, 2).detach().to(torch.float).cpu().numpy()
    # print(W_in_1.shape)  # (11008, 4096)
    #
    # W_gate_1 = model_1.W_gate
    # W_gate_1 = W_gate_1.swapaxes(1, 2).detach().to(torch.float).cpu().numpy()
    # print(W_gate_1.shape)
    #
    # W_out_1 = model_1.W_out.detach().to(torch.float).cpu().numpy()
    # print(W_out_1.shape)
    #
    # W_in_2 = model_2.W_in
    # W_in_2 = W_in_2.swapaxes(1, 2).detach().to(torch.float).cpu().numpy()
    #
    # W_gate_2 = model_2.W_gate
    # W_gate_2 = W_gate_2.swapaxes(1, 2).detach().to(torch.float).cpu().numpy()
    #
    # W_out_2 = model_2.W_out.detach().to(torch.float).cpu().numpy()
    #
    # igo_list = ['in', 'out', 'gate']
    #
    # for igo in igo_list:
    #     if igo == 'in':
    #         path_tail = '/cos_in.csv'
    #         W_1 = W_in_1
    #         W_2 = W_in_2
    #     elif igo == 'out':
    #         path_tail = '/cos_out.csv'
    #         W_1 = W_out_1
    #         W_2 = W_out_2
    #     elif igo == 'gate':
    #         path_tail = '/cos_gate.csv'
    #         W_1 = W_gate_1
    #         W_2 = W_gate_2
    #
    #     neuron_corr = {}
    #     for layer_idx, w_layers in enumerate(zip(W_1, W_2)):
    #         for neuron_idx, w in enumerate(zip(w_layers[0], w_layers[1])):
    #             if not (w[0] == w[1]).all():
    #                 print(str(layer_idx) + '_' + str(neuron_idx) + '_' + igo)
    #
    #             # score = cosine_similarity([w[0]], [w[1]])
    #             # neuron_corr[(layer_idx, neuron_idx)] = score[0][0]


def run_1():
    path = '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/alpaca/base_full/39000step'
    neuron_number = 150000
    # cal_model_weight_cos(model_path_1=r'/data2/haoyun/models/llm/Llama-2-7b-chat-hf',
    #                      model_path_2=r'/data5/haoyun.xu/models/final/alpaca/base_full/checkpoint-39000',
    #                      save_dir=path)

    # cal_model_weight_cos(model_path_1=r'/mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf',
    #                      model_path_2=r'/mnt/nfs/algo/intern/haoyunx11/models/sft/final/mt/lora/enzh/rank_64/checkpoint-9600',
    #                      save_dir=path, is_lora=True)

    threhold = get_threhold_score(
        path_1=path + '/cos_in.csv',
        path_2=path + '/cos_out.csv',
        neuron_number=neuron_number)

    get_neuron(
        path_dir=path,
        threhold=threhold,
        neuron_number=neuron_number)

    # make_mask_with_dict_and_save(
    #     path=path + '/' + str(neuron_number) + '/neuron_dict_' + str(threhold) + '.json',
    #     save_path=path + '/' + str(neuron_number) + "/mask_cos.pt")

    # make_mask_with_dict_and_save_for_offload(path=path + '/neuron_dict_' + str(threhold) + '.json',
    #                                          save_path=path + "/mask_cos_offload.pt")


def run_2():
    path = '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_enzh_org/3200step/add_gate'
    neuron_number = 100000
    # cal_model_weight_cos(model_path_1=r'/mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf',
    #                      model_path_2=r'/mnt/nfs/algo/intern/haoyunx11/models/sft/final/mt/org/enzh/checkpoint-3200',
    #                      save_dir=path)

    threhold = get_threhold_score_add_gate(
        path_1=path + '/cos_in.csv',
        path_2=path + '/cos_out.csv',
        path_3=path + '/cos_gate.csv',
        neuron_number=neuron_number)

    get_neuron_add_gate(
        path_dir=path,
        threhold=threhold,
        neuron_number=neuron_number)

    make_mask_with_dict_and_save_add_gate(
        path=path + '/' + str(neuron_number) + '/neuron_dict_' + str(threhold) + '.json',
        save_path=path + '/' + str(neuron_number) + "/mask_cos.pt")


def run_get_llama3_2_1B_neuron():
    path = '/mnt/data1/saved/NeFT/llama3_2_1B/alpaca/6500step'
    neuron_number = 6000

    cal_model_weight_cos(model_path_1=r'/mnt/data1/models/Llama-3.2-1B',
                     model_path_2=r'/mnt/data1/output/model/llama3_2_1B_alpaca/checkpoint-6500',
                     save_dir=path)

    threhold = get_threhold_score(
        path_1=path + '/cos_in.csv',
        path_2=path + '/cos_out.csv',
        neuron_number=neuron_number)

    get_neuron(
        path_dir=path,
        threhold=threhold,
        neuron_number=neuron_number)

    make_mask_with_dict_and_save_llama3_2_1B_neuron(
        path=path + '/' + str(neuron_number) + '/neuron_dict_' + str(threhold) + '.json',
        save_path=path + '/' + str(neuron_number) + "/mask_cos.pt")


def run_get_Mistral_7B_v1_neuron():
    path = '/mnt/data1/saved/NeFT/Mistral_7B_v1_2/alpaca/6500step'
    neuron_number = 150000

    cal_model_weight_cos_Mistral_7B_v1(model_path_1=r'/mnt/data1/models/Mistral-7B-v0.1',
                     model_path_2=r'/mnt/data1/output/model/Mistral_7B_v1_alpaca_2/checkpoint-6500',
                     save_dir=path)

    threhold = get_threhold_score(
        path_1=path + '/cos_in.csv',
        path_2=path + '/cos_out.csv',
        neuron_number=neuron_number)

    get_neuron(
        path_dir=path,
        threhold=threhold,
        neuron_number=neuron_number)

    make_mask_with_dict_and_save_Mistral_7B_v1_neuron(
        path=path + '/' + str(neuron_number) + '/neuron_dict_' + str(threhold) + '.json',
        save_path=path + '/' + str(neuron_number) + "/mask_cos.pt")

def run_NeFT_check_neuron():
    path = '/mnt/data1/saved/check_neuron'

    cal_model_weight_cos(model_path_1=r'/mnt/data1/models/Llama-3.2-1B',
                     model_path_2=r'/mnt/data1/output/model/llama3_2_1B_alpaca_neuron_6000/checkpoint-6500',
                     save_dir=path)

def run_NeFT_check_neuron_Mistral_7B_v1():
    path = '/mnt/data1/saved/check_neuron_Mistral_7B_v1'

    cal_model_weight_cos_Mistral_7B_v1(model_path_1=r'/mnt/data1/models/Mistral-7B-v0.1',
                     model_path_2=r'/mnt/data1/output/model/Mistral_7B_v1_alpaca_neuron_150000/checkpoint-10',
                     save_dir=path)




if __name__ == '__main__':
    # make_mask_with_dict_and_save(
    #     path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_enzh_org/3200step/150000/neuron_dict_0.9985444.json',
    #     save_path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_enzh_org/3200step/150000/mask_cos.pt')

    # run_1()

    # cal_model_weight_cos_on_select_neuron(
    #     model_path_1=r'/data2/haoyun/models/llm/Llama-2-7b-chat-hf',
    #     model_path_2=r'/data2/haoyun/models/sft/final/mt/LoRA/rank_256/adapter_model',
    #     save_dir=r'/data2/haoyun/MA_project/cos/LoRA/rank_256',
    #     neuron_dict_path='',
    #     is_lora=True)

    # cal_model_weight_cos_on_select_neuron(
    #     model_path_1=r'/data2/haoyun/models/llm/Llama-2-7b-chat-hf',
    #     model_path_2=r'/mnt/nfs/algo/intern/haoyunx11/models/sft/final/mt/train_neuron/enzh/3200step_150000/checkpoint-3200',
    #     save_dir=r'/data2/haoyun/MA_project/cos/Neurons/150000',
    #     neuron_dict_path='/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_enzh_org/3200step/150000/neuron_dict_0.9985444.json',
    #     is_lora=False)

    # run_2()

    # check_neuron(model_path_1='/mnt/data1/models/Llama-3.2-1B_org',
    #              model_path_2='/mnt/data1/output/model/llama3_2_1B_alpaca_neuron_6000/checkpoint-6500')

    # get_Ratio()

    # count_igo()

    # compare_full_sft_and_neuron_train_cos_score_on_sn(
    #     cos_path1=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_enzh_org/3200step/cos_in.csv',
    #     cos_path2=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_enzh_org/train_neuron/3200step_100000/cos_in.csv',
    #     neuron_dict_path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_enzh_org/3200step/100000/neuron_dict_0.99847424.json')

# python cal_neuron_sim.py >output.out 2>&1 &

    # run_get_llama3_2_1B_neuron()
    run_get_Mistral_7B_v1_neuron()

    # run_NeFT_check_neuron()

    # run_NeFT_check_neuron_Mistral_7B_v1()

"""
7B: 11008 * 32

1B: 8192 * 16

"""
