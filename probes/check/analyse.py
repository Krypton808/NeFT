import pandas as pd


def order_dict(dict_info, order_info):
    order_ = None
    flag = 1
    for key, value in reversed(order_info.items()):
        reverse = value == 0
        if flag == 1:
            order_ = sorted(dict_info, key=lambda x: x[key], reverse=reverse)

            flag = 0
        else:
            order_ = sorted(order_, key=lambda x: x[key], reverse=reverse)
    return order_


def sort_neuron_list(neuron_name_list):
    dict_data_list = []
    OrderedDict = {'neuron_layer': 1, 'neuron': 1, 'neuron_weight': 1}

    for idx, name in enumerate(neuron_name_list):
        name = name.split('_')

        dict_data = {}
        dict_data['neuron_layer'] = int(name[0])
        dict_data['neuron'] = int(name[1])
        dict_data['neuron_weight'] = name[2]

        dict_data_list.append(dict_data)

    ret_list = order_dict(dict_data_list, OrderedDict)
    # print(ret_list)

    return ret_list


def getoverlap(*args):
    lists = []
    for path in args:
        pd_data = pd.read_csv(path)
        neuron_name_list = []  # str(neuron_layer + neuron + neuron_weight(in/out))
        abs_corr_list = []
        neuron_name_temp_list = []
        neuron_name_temp = ''
        abs_corr_temp_list = []
        dict_data = {}

        for data in pd_data.iterrows():
            data = data[1]
            neuron_layer = data['neuron_layer']
            neuron = data['neuron']
            neuron_weight = data['neuron_weight']
            abs_corr = data['abs_corr']
            neuron_name = str(neuron_layer) + '_' + str(neuron) + '_' + neuron_weight
            if neuron_name_temp != neuron_name:
                neuron_name_list.append(neuron_name)
                if abs_corr_temp_list != []:
                    abs_corr_list.append(abs_corr_temp_list)
                    dict_data[neuron_name_temp] = abs_corr_temp_list

                neuron_name_temp = neuron_name
                abs_corr_temp_list = []
            abs_corr_temp_list.append(abs_corr)

        # 添加最后一个
        abs_corr_list.append(abs_corr_temp_list)
        dict_data[neuron_name] = abs_corr_temp_list

        draw_data_list = []

        sorted_dict_list = sort_neuron_list(dict_data.keys())

        label_list = []
        for sorted_dict in sorted_dict_list:
            neuron_name = str(sorted_dict['neuron_layer']) + '_' + str(sorted_dict['neuron']) + '_' + sorted_dict[
                'neuron_weight']
            draw_data_list.append(dict_data[neuron_name])
            label_list.append(neuron_name)

        print(label_list)
        lists.append(label_list)

    r = set(lists[0])
    for i in lists[1:]:
        r.intersection_update(i)

    r = list(r)

    r = sort_neuron_list(r)
    label_list = []
    for sorted_dict in r:
        neuron_name = str(sorted_dict['neuron_layer']) + '_' + str(sorted_dict['neuron']) + '_' + sorted_dict[
            'neuron_weight']
        label_list.append(neuron_name)

    print('====================================')
    print(label_list)
    print(len(label_list))


if __name__ == '__main__':
    # ['1_20_out', '1_1331_out', '1_1936_in', '1_2069_in', '1_2069_out', '1_4600_in', '1_9015_out', '1_9350_in', '1_9805_out', '1_10397_out', '2_1325_in', '2_1325_out', '2_3699_in', '2_5201_out', '2_8637_in', '2_10040_in', '3_1370_in', '3_1370_out', '3_3687_out', '3_8948_out', '3_9352_in', '4_328_in', '4_328_out', '4_1522_in', '4_4336_in', '4_9116_in', '6_4469_in']
    # getoverlap(
    #     # en
    #     "/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/neuron_4_328_in_out_progressive_500steps/en/neuron_4_328_in_out_progressive_500steps.en_standard_prompt.csv",
    #     "/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/neuron_4_328_in_out_progressive_2000steps/en/neuron_4_328_in_out_progressive_2000steps.en_standard_prompt.csv",
    #     "/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_layer4_2500steps/en/xnli_en_sft_layer4_2500steps.en_standard_prompt.csv",
    #     "/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_layer30_2500steps/en/xnli_en_sft_layer30_2500steps.en_standard_prompt.csv",
    #     "/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_all_layers_500steps/en/xnli_en_sft_all_layers_500steps.en_standard_prompt.csv",
    #     "/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/en/xnli.en_standard_prompt.csv",
    #
    #     # de
    #     "/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_all_layers_500steps/de/xnli_en_sft_all_layers_500steps.de_standard_prompt.csv",
    #     "/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_layer4_2500steps/de/xnli_en_sft_layer4_2500steps.de_standard_prompt.csv",
    #     "/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_layer30_2500steps/de/xnli_en_sft_layer30_2500steps.de_standard_prompt.csv",
    #     "/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/de/xnli.de_standard_prompt.csv",
    #
    #     # zh
    #     "/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_all_layers_500steps/zh/xnli_en_sft_all_layers_500steps.zh_standard_prompt.csv",
    #     "/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_layer4_2500steps/zh/xnli_en_sft_layer4_2500steps.zh_standard_prompt.csv",
    #     "/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_layer30_2500steps/zh/xnli_en_sft_layer30_2500steps.zh_standard_prompt.csv",
    #     "/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/zh/xnli.zh_standard_prompt.csv")

    # ['0_623_out', '0_2317_out', '0_7374_in', '0_7374_out', '1_20_out', '1_1331_in', '1_1331_out', '1_1936_in',
    #  '1_2069_in', '1_2069_out', '1_2318_in', '1_4600_in', '1_5805_in', '1_7890_in', '1_8731_in', '1_9015_out',
    #  '1_9350_in', '1_9561_out', '1_9805_out', '1_10102_out', '1_10397_out', '2_1325_in', '2_1325_out', '2_1800_in',
    #  '2_3699_in', '2_5201_out', '2_8637_in', '2_10040_in', '3_1370_in', '3_1370_out', '3_1827_in', '3_3687_out',
    #  '3_7965_out', '3_8948_out', '3_9352_in', '3_9654_in', '4_328_in', '4_328_out', '4_1522_in', '4_2347_out',
    #  '4_4336_in', '4_9116_in', '6_4469_in']
    # getoverlap(
    #     # org standard
    #     "/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/en/xnli.en_standard_prompt.csv",
    #     "/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/de/xnli.de_standard_prompt.csv",
    #     "/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/zh/xnli.zh_standard_prompt.csv",
    #     "/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/fr/xnli.fr_standard_prompt.csv",
    #     "/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/ru/xnli.ru_standard_prompt.csv",
    #     "/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/es/xnli.es_standard_prompt.csv",
    #     "/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/bg/xnli.bg_standard_prompt.csv"
    # )

    getoverlap(
        # en
        "/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/neuron_4_328_in_out_progressive_500steps/en/neuron_4_328_in_out_progressive_500steps.en_standard_prompt.csv",
        "/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/neuron_4_328_in_out_progressive_2000steps/en/neuron_4_328_in_out_progressive_2000steps.en_standard_prompt.csv",
        "/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_layer4_2500steps/en/xnli_en_sft_layer4_2500steps.en_standard_prompt.csv",
        "/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_layer30_2500steps/en/xnli_en_sft_layer30_2500steps.en_standard_prompt.csv",
        "/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_all_layers_500steps/en/xnli_en_sft_all_layers_500steps.en_standard_prompt.csv",
        "/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/en/xnli.en_standard_prompt.csv",

        # de
        "/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_all_layers_500steps/de/xnli_en_sft_all_layers_500steps.de_standard_prompt.csv",
        "/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_layer4_2500steps/de/xnli_en_sft_layer4_2500steps.de_standard_prompt.csv",
        "/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_layer30_2500steps/de/xnli_en_sft_layer30_2500steps.de_standard_prompt.csv",
        "/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/de/xnli.de_standard_prompt.csv",

        # zh
        "/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_all_layers_500steps/zh/xnli_en_sft_all_layers_500steps.zh_standard_prompt.csv",
        "/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_layer4_2500steps/zh/xnli_en_sft_layer4_2500steps.zh_standard_prompt.csv",
        "/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_layer30_2500steps/zh/xnli_en_sft_layer30_2500steps.zh_standard_prompt.csv",
        "/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/zh/xnli.zh_standard_prompt.csv",

        # other
        "/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/fr/xnli.fr_standard_prompt.csv",
        "/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/ru/xnli.ru_standard_prompt.csv",
        "/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/es/xnli.es_standard_prompt.csv",
        "/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/bg/xnli.bg_standard_prompt.csv"

    )
