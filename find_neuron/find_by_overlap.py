import json


def cross_language_overlap(path_list):
    neuron_dict_list = []

    overlap_neuron_dict = {}
    save_path = r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/overlap/x-zh/200000_find_x/overlap_neuron_dict.json'
    for path in path_list:
        f = open(path, 'r', encoding='utf-8')
        neuron_dict = json.load(f)
        print(len(neuron_dict.keys()))
        neuron_dict_list.append(neuron_dict)

    total_length = 0
    total_length_ = 0
    for items in zip(neuron_dict_list[0].items(), neuron_dict_list[1].items(), neuron_dict_list[2].items(),
                     neuron_dict_list[3].items()):

        print(items[0][0])

        overlap = list(set(items[0][1]).intersection(items[1][1], items[2][1]))

        if items[0][0] not in overlap_neuron_dict.keys():
            overlap_neuron_dict[items[0][0]] = overlap
        else:
            print(items[0][0])
            print('***************')
            break

        overlap_ = list(set(overlap).intersection(items[3][1]))

        # overlap = list(set(items[0][1]).intersection(items[1][1]))
        # overlap = list(set(items[0][1]).intersection(items[2][1]))
        # overlap = list(set(items[1][1]).intersection(items[2][1]))

        length = len(overlap)
        length_ = len(overlap_)
        # print(overlap)
        print(length)
        print(length_)
        total_length += length
        total_length_ += length_

    print(total_length)
    print(total_length_)

    with open(save_path, "w") as f:
        json.dump(overlap_neuron_dict, f)
    print("加载入文件完成...")


def two_neuron_dict_union(path_list):
    neuron_dict_list = []

    union_neuron_dict = {}
    save_path = r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/methods+/mt/enzh/union/pearson100000_cos_near100000/union_neuron_dict.json'
    for path in path_list:
        f = open(path, 'r', encoding='utf-8')
        neuron_dict = json.load(f)
        print(len(neuron_dict.keys()))
        neuron_dict_list.append(neuron_dict)

    total_length = 0
    total_length_ = 0
    for items in zip(neuron_dict_list[0].items(), neuron_dict_list[1].items()):

        print(items[0][0])

        union = list(set(items[0][1] + items[1][1]))
        length = len(union)
        total_length += length

        if items[0][0] not in union_neuron_dict.keys():
            union_neuron_dict[items[0][0]] = union
        else:
            print(items[0][0])
            print('***************')
            break

    print(total_length)
    print(total_length_)

    with open(save_path, "w") as f:
        json.dump(union_neuron_dict, f)
    print("加载入文件完成...")


def two_neuron_dict_overlap(path_list,
                            save_path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/methods_overlap/xnli/en/pearson100000_probe100000/overlap_neuron_dict.json'):
    neuron_dict_list = []
    overlap_neuron_dict = {}

    for path in path_list:
        f = open(path, 'r', encoding='utf-8')
        neuron_dict = json.load(f)
        # print(len(neuron_dict.keys()))
        neuron_dict_list.append(neuron_dict)

    total_length = 0
    count_in = 0
    for items in zip(neuron_dict_list[0].items(), neuron_dict_list[1].items()):

        if items[0][0] != items[1][0]:
            print('key not match')
            break

        # print(items[0][0])

        overlap = list(set(items[0][1]).intersection(items[1][1]))

        if items[0][0] not in overlap_neuron_dict.keys():
            overlap_neuron_dict[items[0][0]] = overlap
        else:
            # print(items[0][0])
            # print('***************')
            break

        # print(overlap)
        length = len(overlap)
        # print(length)

        if items[0][0].split('_')[1] == 'in':
            count_in += length

        total_length += length

    # print(total_length)
    # print(count_in)
    # print(total_length - count_in)
    with open(save_path, "w") as f:
        json.dump(overlap_neuron_dict, f)
    print("加载入文件完成...")

    return total_length


def two_neuron_dict_disjoint(path_list,
                             save_path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/methods_overlap/xnli/en/pearson100000_probe100000/overlap_neuron_dict.json'):
    neuron_dict_list = []
    disjoint_neuron_dict = {}

    for path in path_list:
        f = open(path, 'r', encoding='utf-8')
        neuron_dict = json.load(f)
        print(len(neuron_dict.keys()))
        neuron_dict_list.append(neuron_dict)

    total_length = 0

    for items in zip(neuron_dict_list[0].items(), neuron_dict_list[1].items()):

        print(items[0][0])

        disjoint = list(set(items[0][1]) - set(items[1][1]))

        if items[0][0] not in disjoint_neuron_dict.keys():
            disjoint_neuron_dict[items[0][0]] = disjoint
        else:
            print(items[0][0])
            print('***************')
            break

        length = len(disjoint)
        print(length)

        total_length += length

    print(total_length)

    with open(save_path, "w") as f:
        json.dump(disjoint_neuron_dict, f)
    print("加载入文件完成...")


def two_neuron_dict_complement_set(path_list,
                                   save_path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/methods_overlap/xnli/en/pearson100000_probe100000/overlap_neuron_dict.json'):
    neuron_dict_list = []
    overlap_neuron_dict = {}

    for path in path_list:
        f = open(path, 'r', encoding='utf-8')
        neuron_dict = json.load(f)
        print(len(neuron_dict.keys()))
        neuron_dict_list.append(neuron_dict)

    total_length = 0
    count_in = 0
    for items in zip(neuron_dict_list[0].items(), neuron_dict_list[1].items()):

        if items[0][0] != items[1][0]:
            print('key not match')
            break

        print(items[0][0])

        complement_set = list(set(items[0][1]) ^ set(items[1][1]))

        if items[0][0] not in overlap_neuron_dict.keys():
            overlap_neuron_dict[items[0][0]] = complement_set
        else:
            print(items[0][0])
            print('***************')
            break

        length = len(complement_set)
        print(length)

        if items[0][0].split('_')[1] == 'in':
            count_in += length

        total_length += length

    print(total_length)
    # print(count_in)
    # print(total_length - count_in)
    with open(save_path, "w") as f:
        json.dump(overlap_neuron_dict, f)
    print("加载入文件完成...")


def overlap_by_slice():
    path_dir = '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/rank_diff/cos-100000_cos-50000/max_sorted_add_rank_diff'

    temp_path = '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/temp.json'
    slice_cut = [{'start': 0, 'end': 0.01}, {'start': 0.01, 'end': 0.02}, {'start': 0.02, 'end': 0.03},
                 {'start': 0.03, 'end': 0.1}, {'start': 0.1, 'end': 0.2}, {'start': 0.2, 'end': 0.3},
                 {'start': 0.3, 'end': 0.4}, {'start': 0.4, 'end': 0.5}, {'start': 0.5, 'end': 0.6},
                 {'start': 0.6, 'end': 0.7}, {'start': 0.7, 'end': 0.8}, {'start': 0.8, 'end': 0.9},
                 {'start': 0.9, 'end': 0.97}, {'start': 0.97, 'end': 0.98}, {'start': 0.98, 'end': 0.99},
                 {'start': 0.99, 'end': 1}]


    for s in slice_cut:
        path = path_dir + '/' + str(s['start']) + '-' + str(s['end']) + '_neuron_dict.json'

        path_list = [
            path,
            '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_enzh_org/3200step/100000/neuron_dict_0.99847424.json',
        ]

        total_length = two_neuron_dict_overlap(path_list,
                                save_path=temp_path)

        path_list = [
            temp_path,
            '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/rank_diff/cos-100000_cos-50000/abs_rank_diff_larger_10W.json',
        ]

        total_length = two_neuron_dict_overlap(path_list, save_path=temp_path)

        print(s)
        print(total_length)
        print('*************************')


if __name__ == '__main__':
    # x-zh
    # path_list = [
    #     '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_enzh_org/3200step/100000/neuron_dict_0.99847424.json',
    #     '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_frzh_org/3200step/100000/neuron_dict_0.9984873.json',
    #     '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_hizh_org/750step/100000/neuron_dict_0.9996239.json',
    #     '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_hizh_org/750step/50000/neuron_dict_0.99959487.json'
    # ]

    # path_list = [
    #     '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_enzh_org/3200step/150000/neuron_dict_0.9985444.json',
    #     '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_frzh_org/3200step/150000/neuron_dict_0.99855405.json',
    #     '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_hizh_org/750step/150000/neuron_dict_0.99964154.json',
    #     '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_hizh_org/750step/50000/neuron_dict_0.99959487.json'
    # ]

    # path_list = [
    #     '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_enzh_org/3200step/200000/neuron_dict_0.9985985.json',
    #     '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_frzh_org/3200step/200000/neuron_dict_0.9986056.json',
    #     '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_hizh_org/750step/200000/neuron_dict_0.99965477.json',
    #     '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_hizh_org/750step/50000/neuron_dict_0.99959487.json'
    # ]

    # step overlap
    # path_list = [
    #     '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_enzh_org/3200step/100000/neuron_dict_0.99847424.json',
    #     '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_frzh_org/3200step/100000/neuron_dict_0.9984873.json',
    #     '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_hizh_org/750step/100000/neuron_dict_0.9996239.json',
    #     '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_hizh_org/750step/50000/neuron_dict_0.99959487.json'
    # ]

    # cross_language_overlap(path_list)

    # path_list = [
    #     '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_hifr_org/750step/100000/neuron_dict_0.99960494.json',
    #     '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/summary_neurons_sim/final/hifr/21step/100000/neuron_dict_0.99998355.json'
    # ]

    # frzh x-zh_100000_find_49305_union_50000
    # path_list = [
    #     '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_enzh_org/3200step/100000/neuron_dict_0.99847424.json',
    #     '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/enzh_org-enzh_3200step_full/100000/neuron_dict.json'
    # ]

    # two_neuron_dict_union(path_list)

    # xnli
    # en pearson 100000 vs probe 100000
    # path_list = [
    #     '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/xnli/en/en_org-en_1000step_full/100000/neuron_dict.json',
    #     '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/xnli_neurons_sim/final/probes/org/100000/neuron_dict_.json'
    # ]

    # mt
    # enzh pearson 100000 vs cos 100000
    # path_list = [
    #     '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/enzh_org-enzh_3200step_full/100000/neuron_dict.json',
    #     '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_enzh_org/3200step/100000/neuron_dict_0.99847424.json'
    # ]

    path_list = [
        # '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/enzh_org-enzh_lora_rank_64/pearson/reversed/500x2/neuron_dict.json',
        # '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/frzh/from_enzh/frzh_org-enzh_sn_3200_50000/pearson/50000x2/neuron_dict.json',
        # '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/enzh_org-enzh_sn_3200_100000/pearson/reversed/10000x2/neuron_dict.json',
        # '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/enzh_org-enzh_in_out/pearson/reversed/50000x2/neuron_dict.json',
        # '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_enzh_org/3200step/100000/neuron_dict_0.99847424.json',
        # '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/rank_diff/in-out_cos-50000/abs_rank_diff_larger_10W_rise.json',

        # '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/rank_diff/cos-150000_cos-50000/abs_rank_diff_larger_10W.json',
        # '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/rank_diff/cos-150000_cos-50000/abs_rank_diff_larger_10W_fall.json',
        # '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/rank_diff/cos-150000_cos-50000/5w/abs_rank_diff_larger_5W_fall.json',
        # '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/rank_diff/cos-10w-150000_cos-50000/overlap_abs-rank-diff-larger-10W_cos-10w-150000_neuron_dict.json',
        # '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/rank_diff/cos-200000_cos-50000/abs_rank_diff_larger_10W.json',

        # '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/rank_diff/cos-200000_cos-50000/abs_rank_diff_larger_10W.json',
        # '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/rank_diff/cos-175000_cos-50000/abs_rank_diff_larger_10W.json',
        # 
        # '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_enzh_org/3200step/reversed/100000/neuron_dict_0.9989356.json',
        # '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/enzh_org-enzh_3200step_full/pearson/50000x2/neuron_dict.json'

        # '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/temp_1.json',
        # '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/temp_2.json',
        # '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/temp_3.json',




        # '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/temp.json',
        # '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/rank_diff/cos-50000+reversed-50000_cos-50000/abs_rank_diff_larger_10W_rise.json'
        # '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/rank_diff/cos-50000+reversed-50000_cos-50000/disjoint_abs-rank-diff-larger-10W_cos-50000+reversed-50000_neuron_dict.json'
        # '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/rank_diff/cos-10w-150000_cos-50000/abs_rank_diff_larger_10W_rise.json'
        # '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/summary_neurons_sim/final/union/enzh/150000x2/overlap_union_neuron_dict.json',
        # '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/summary_neurons_sim/final/union/enzh/150000x2/overlap_union_neuron_dict.json'

        '/data2/haoyun/MA_project/LoRA_neurons/rank_diff/rank_256-rank_128/abs_rank_diff_larger_10W.json',
        '/data2/haoyun/MA_project/LoRA_neurons/rank_diff/rank_256-rank_128/abs_rank_diff_larger_10W.json'

    ]

    number = two_neuron_dict_overlap(path_list,
                            save_path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/temp_1.json')
    print(number)
    # two_neuron_dict_complement_set(path_list,
    #                         save_path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_enzh_org/3200step/other/50000+reversed_125000/neuron_dict.json')

    # two_neuron_dict_disjoint(path_list,
    #                          save_path='/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/rank_diff/cos-150000_cos-50000/disjoint_abs-rank-diff-larger-10W_cos-150000_neuron_dict.json')

    # overlap_by_slice()