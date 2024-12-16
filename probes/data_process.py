import csv

import json
import jsonlines


def get_neuron_from_corr_file(
        path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/final/top_neurons/Llama-2-7b-chat-hf/en/xnli.en_standard_prompt_pearsonr_.csv'):
    f = open(path, 'r', encoding='utf-8')
    rows = csv.reader(f)

    neuron_set = set()
    for idx, row in enumerate(rows):
        if idx == 0:
            continue

        neuron_layer = row[0]
        neuron = row[1]
        neuron_weight = row[5]

        neuron_name = str(neuron_layer) + '_' + str(neuron) + '_' + str(neuron_weight)
        neuron_set.add(neuron_name)

    neuron_list = list(neuron_set)

    print(neuron_list)

    print(len(neuron_list))


def make_neuron_dict(path_1='/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/in.json',
                     path_2='/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/out.json', w_path=''):
    with open(path_1, 'r', encoding='utf-8') as f:
        line_1 = json.load(f)
    with open(path_2, 'r', encoding='utf-8') as f:
        line_2 = json.load(f)

    top_layers_in = line_1['top_layers']
    top_layers_in.reverse()
    top_neurons_in = line_1['top_neurons']
    top_neurons_in.reverse()
    scores_in = line_1['scores']
    scores_in.reverse()

    top_layers_out = line_2['top_layers']
    top_layers_out.reverse()
    top_neurons_out = line_2['top_neurons']
    top_neurons_out.reverse()
    scores_out = line_2['scores']
    scores_out.reverse()

    # scores_all = []
    # scores_all.extend(scores_in)
    # scores_all.extend(scores_out)
    # print(len(scores_all))
    #
    # scores_all_abs = [abs(float(i)) for i in scores_all]
    #
    # scores_all_abs.sort(reverse=True)
    # print(scores_all_abs[99999])    # 0.04531020298600197

    neuron_name_dict = {}

    for line in zip(top_layers_in, top_neurons_in, scores_in):
        neuron_name = str(line[0]) + '_' + str(line[1]) + '_' + 'in'
        if neuron_name in neuron_name_dict.keys():
            continue
        neuron_name_dict[neuron_name] = abs(float(line[2]))

    for line in zip(top_layers_out, top_neurons_out, scores_out):
        neuron_name = str(line[0]) + '_' + str(line[1]) + '_' + 'out'
        if neuron_name in neuron_name_dict.keys():
            continue
        neuron_name_dict[neuron_name] = abs(float(line[2]))

    neuron_name_dict_sorted = sorted(neuron_name_dict.items(), key=lambda d: d[1], reverse=True)
    print(neuron_name_dict_sorted)

    neuron_dict = {}
    count = 0
    for line in neuron_name_dict_sorted:
        if count >= 100000:
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

    with open(
            "/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/xnli_neurons_sim/final/de/probes/org/100000" + "/neuron_dict" + ".json",
            "w") as f:
        json.dump(neuron_dict, f)
    print("加载入文件完成...")


def cut_data(
        path_in=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/data/xnli/final/de_standard_prompt_train.jsonl',
        path_out=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/data/xnli/final/de_standard_prompt_train_6k.jsonl'):

    lines = jsonlines.open(path_in)
    writer = jsonlines.open(path_out, 'w')
    count = 0
    for line in lines:
        if count >= 6000:
            break

        writer.write(line)
        count += 1



if __name__ == '__main__':
#   get_neuron_from_corr_file()
#   make_neuron_dict()
    cut_data()