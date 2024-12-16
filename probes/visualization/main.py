import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# sns.heatmap(data=data,annot=True,fmt="d",cmap="RdBu_r")

def heatmap_test():
    # 随机生成一个200行10列的数据集
    data_new = np.random.randn(200, 10)
    corr = np.corrcoef(data_new, rowvar=False)
    print(corr.shape)
    # 以corr的形状生成一个全为0的矩阵
    mask = np.zeros_like(corr)
    # 将mask的对角线及以上设置为True
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        sns.heatmap(corr, mask=mask, vmax=0.5, annot=True, cmap="RdBu_r")

    plt.show()


def pandas_heatmap_test():
    arr_2d = np.arange(-8, 8).reshape((4, 4))
    print(arr_2d)
    df = pd.DataFrame(data=arr_2d, index=['a', 'b', 'c', 'd'], columns=['A', 'B', 'C', 'D'])
    print(df)


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


def order_dict_two(dict_info, order_info):
    """
    用于排序的函数,输入dict，指定排序条件，然后输出排序结果
    :param dict_info: 信息字典列表
    :param order_info: 排序条件，越前面的条件越重要。如{条件1:1},1表示升序（从小到大），0表示倒序（从大到小）
    :return:
    """
    order_lambda = []
    for key, value in order_info.items():
        if value:
            order_lambda.append('x["%s"]' % key)
        else:
            order_lambda.append('-x["%s"]' % key)
    exc = 'sorted(dict_info, key=lambda x: (%s))' % (','.join(order_lambda))
    order_ = eval(exc)
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


def neuron_heatmap(
        path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/neuron_4_328_in_out_progressive_500steps/en/neuron_4_328_in_out_progressive_500steps.en_standard_prompt.csv',
        vmax=0.7):
    sns.set_context({'figure.figsize': [30, 30]})
    title = path.split('/')[-1].strip().replace('.csv', '')
    fig_path = title + '.png'

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

    df = pd.DataFrame(data=draw_data_list, index=label_list)
    fig = sns.heatmap(df, vmax=vmax, annot=True, cmap="RdBu_r").set_title(title)
    heatmap = fig.get_figure()
    # heatmap.savefig(fig_path, dpi=400)

    # print(abs_corr_list)
    # print(len(abs_corr_list))
    # print(neuron_name_list)
    # print(len(neuron_name_list))
    # print(dict_data)
    # print(dict_data.keys())

    plt.show()


def neuron_heatmap_show_specify_neuron(path_list, specify_neuron_list, vmax=0.7):
    sns.set_context({'figure.figsize': [46, 20]})

    specify_neuron_draw_data_list = []
    specify_neuron_label_list = []

    draw_data_list_all = []
    label_list_all = []
    title_list = []

    for path in path_list:

        title = path.split('/')[-1].strip().replace('.csv', '')
        fig_path = specify_neuron_list[0] + '.png'

        pd_data = pd.read_csv(path)
        neuron_name_list = []  # str(neuron_layer + neuron + neuron_weight(in/out))
        abs_corr_list = []
        neuron_name_temp_list = []
        neuron_name_temp = ''
        abs_corr_temp_list = []
        dict_data = {}

        title_list.append(title)

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

        label_list_all.append(label_list)
        draw_data_list_all.append(draw_data_list)

    for specify_neuron in specify_neuron_list:
        for line in zip(label_list_all, draw_data_list_all, title_list):
            if specify_neuron not in line[0]:
                continue
            index = line[0].index(specify_neuron)
            name = line[2] + '_' + line[0][index]

            specify_neuron_label_list.append(name)
            specify_neuron_draw_data_list.append(line[1][index])

    # print(specify_neuron_label_list)
    # print(len(specify_neuron_label_list))
    # print(len(specify_neuron_draw_data_list))
    # print(len(specify_neuron_draw_data_list[0]))
    #
    print(specify_neuron_draw_data_list)


    df = pd.DataFrame(data=specify_neuron_draw_data_list, index=specify_neuron_label_list)
    fig = sns.heatmap(df, vmax=vmax, annot=False, cmap="RdBu_r").set_title(specify_neuron_list[0])
    heatmap = fig.get_figure()
    # heatmap.savefig(fig_path, dpi=400)
    # print(abs_corr_list)
    # print(len(abs_corr_list))
    # print(neuron_name_list)
    # print(len(neuron_name_list))
    # print(dict_data)
    # print(dict_data.keys())
    plt.show()


if __name__ == '__main__':
    # en
    # neuron_heatmap(vmax=0.4)
    #
    # neuron_heatmap(
    #     path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/neuron_4_328_in_out_progressive_2000steps/en/neuron_4_328_in_out_progressive_2000steps.en_standard_prompt.csv',
    #     vmax=0.4)
    #
    # neuron_heatmap(
    #     path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_layer4_2500steps/en/xnli_en_sft_layer4_2500steps.en_standard_prompt.csv',
    #     vmax=0.4)
    #
    # neuron_heatmap(
    #     path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_layer30_2500steps/en/xnli_en_sft_layer30_2500steps.en_standard_prompt.csv',
    #     vmax=0.4)
    #
    # neuron_heatmap(
    #     path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_all_layers_500steps/en/xnli_en_sft_all_layers_500steps.en_standard_prompt.csv',
    #     vmax=0.4)
    #
    # neuron_heatmap(
    #     path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/en/xnli.en_standard_prompt.csv',
    #     vmax=0.4)

    # de
    # neuron_heatmap(
    #     path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_layer4_2500steps/de/xnli_en_sft_layer4_2500steps.de_standard_prompt.csv',
    #     vmax=0.4)
    #
    # neuron_heatmap(
    #     path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_all_layers_500steps/de/xnli_en_sft_all_layers_500steps.de_standard_prompt.csv',
    #     vmax=0.4)

    # neuron_heatmap(
    #     path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_layer30_2500steps/de/xnli_en_sft_layer30_2500steps.de_standard_prompt.csv',
    #     vmax=0.4)

    # neuron_heatmap(
    #     path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/de/xnli.de_standard_prompt.csv',
    #     vmax=0.4)

    # zh
    neuron_heatmap(
        path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_all_layers_500steps/zh/xnli_en_sft_all_layers_500steps.zh_standard_prompt.csv',
        vmax=0.2)

    neuron_heatmap(
        path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_layer4_2500steps/zh/xnli_en_sft_layer4_2500steps.zh_standard_prompt.csv',
        vmax=0.2)

    neuron_heatmap(
        path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_layer30_2500steps/zh/xnli_en_sft_layer30_2500steps.zh_standard_prompt.csv',
        vmax=0.2)

    neuron_heatmap(
        path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/zh/xnli.zh_standard_prompt.csv',
        vmax=0.2)

    # R2
    # neuron_heatmap(
    #     path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/neuron_4_328_in_out_progressive_500steps/en/neuron_4_328_in_out_progressive_500steps.en_standard_prompt_r2_sklearn.csv',
    #     vmax=100)
    #
    # neuron_heatmap(
    #     path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/neuron_4_328_in_out_progressive_2000steps/en/neuron_4_328_in_out_progressive_2000steps.en_standard_prompt_r2_sklearn.csv',
    #     vmax=100)
    #
    # neuron_heatmap(
    #     path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_layer4_2500steps/en/xnli_en_sft_layer4_2500steps.en_standard_prompt_r2_sklearn.csv',
    #     vmax=100)
    #
    # neuron_heatmap(
    #     path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_layer30_2500steps/en/xnli_en_sft_layer30_2500steps.en_standard_prompt_r2_sklearn.csv',
    #     vmax=100)
    #
    # neuron_heatmap(
    #     path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_all_layers_500steps/en/xnli_en_sft_all_layers_500steps.en_standard_prompt_r2_sklearn.csv',
    #     vmax=100)
    #
    # neuron_heatmap(
    #     path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/en/xnli.en_standard_prompt_r2_sklearn.csv',
    #     vmax=100)

    # org
    # neuron_heatmap(
    #     path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/en/xnli.en_standard_prompt_r2_sklearn.csv',
    #     vmax=100)
    #
    # neuron_heatmap(
    #     path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/de/xnli.de_standard_prompt_r2_sklearn.csv',
    #     vmax=100)
    #
    # neuron_heatmap(
    #     path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/ru/xnli.ru_standard_prompt_r2_sklearn.csv',
    #     vmax=100)
    #
    # neuron_heatmap(
    #     path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/bg/xnli.bg_standard_prompt_r2_sklearn.csv',
    #     vmax=100)
    #
    # neuron_heatmap(
    #     path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/fr/xnli.fr_standard_prompt_r2_sklearn.csv',
    #     vmax=100)
    #
    # neuron_heatmap(
    #     path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/es/xnli.es_standard_prompt_r2_sklearn.csv',
    #     vmax=100)
    #
    # neuron_heatmap(
    #     path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/zh/xnli.zh_standard_prompt_r2_sklearn.csv',
    #     vmax=100)


# path_list = [
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
    #     "/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/zh/xnli.zh_standard_prompt.csv"
    # ]
    #
    # specify_neuron_list = ['1_20_out', '1_1331_out', '1_1936_in', '1_2069_in', '1_2069_out', '1_4600_in', '1_9015_out',
    #                        '1_9350_in', '1_9805_out', '1_10397_out', '2_1325_in', '2_1325_out', '2_3699_in',
    #                        '2_5201_out', '2_8637_in', '2_10040_in', '3_1370_in', '3_1370_out', '3_3687_out',
    #                        '3_8948_out', '3_9352_in', '4_328_in', '4_328_out', '4_1522_in', '4_4336_in', '4_9116_in',
    #                        '6_4469_in']
    #
    # specify_neuron_list_2 = ['1_2069_in', '1_4600_in', '3_1370_out', '3_3687_out', '3_8948_out', '4_328_in',
    #                          '4_328_out']
    #
    # specify_neuron_list_3 = ['11_647_out']
    #
    # for specify_neuron in specify_neuron_list_3:
    #     input = [specify_neuron]
    #
    #     neuron_heatmap_show_specify_neuron(path_list, input, vmax=0.3)
