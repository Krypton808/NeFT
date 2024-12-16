import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import json
import math
import matplotlib.gridspec as gridspec
from adjustText import adjust_text
from proplot import rc

from matplotlib.pyplot import MultipleLocator


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
    for key, value in Reversed(order_info.items()):
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
        dict_data['neuron_layer'] = int(float(name[0]))
        dict_data['neuron'] = int(float(name[1]))
        dict_data['neuron_weight'] = name[2] + '_' + name[3]

        dict_data_list.append(dict_data)

    ret_list = order_dict(dict_data_list, OrderedDict)
    # print(ret_list)

    return ret_list


def sort_neuron_list_cos(neuron_name_list):
    dict_data_list = []
    OrderedDict = {'neuron': 1}

    for idx, name in enumerate(neuron_name_list):
        name = name.split('_')

        dict_data = {}
        dict_data['neuron'] = int(float(name[0]))
        dict_data['neuron_weight'] = 'out'

        dict_data_list.append(dict_data)

    ret_list = order_dict(dict_data_list, OrderedDict)
    # print(ret_list)

    return ret_list


def neuron_heatmap(
        path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/neuron_4_328_in_out_progressive_500steps/en/neuron_4_328_in_out_progressive_500steps.en_standard_prompt.csv',
        vmax=1, threshold=0.5):
    sns.set_context({'figure.figsize': [30, 30]})
    title = path.split('/')[-1].strip().replace('.csv', '')
    fig_path = title + '.png'

    pd_data = pd.read_csv(path)
    neuron_name_list = []  # str(neuron_layer + neuron + neuron_weight(in/out))
    abs_corr_list = []
    neuron_name_temp_list = []
    dict_data = {}

    for data in pd_data.iterrows():
        data = data[1]
        neuron_layer = data['neuron_layer']
        neuron = data['neuron']
        activation_layer = data['activation_layer']
        neuron_weight = data['neuron_weight']
        abs_corr = data['abs_corr']
        if abs_corr < 0.05:
            continue

        if neuron_layer > activation_layer:
            continue

        neuron_name = str(neuron_layer) + '_' + str(neuron) + '_' + neuron_weight

        if neuron_name not in dict_data.keys():
            dict_data[neuron_name] = [0] * 32
            dict_data[neuron_name][activation_layer] = abs_corr

        else:
            dict_data[neuron_name][activation_layer] = abs_corr

    draw_data_list = []

    sorted_dict_list = sort_neuron_list(dict_data.keys())

    label_list = []
    for sorted_dict in sorted_dict_list:
        neuron_name = str(sorted_dict['neuron_layer']) + '_' + str(sorted_dict['neuron']) + '_' + sorted_dict[
            'neuron_weight']
        # print(dict_data.keys())
        draw_data_list.append(dict_data[neuron_name])
        label_list.append(neuron_name)

    print(label_list)

    df = pd.DataFrame(data=draw_data_list, index=label_list)
    fig = sns.heatmap(df, vmax=vmax, annot=True, cmap="RdBu_r").set_title(title)
    heatmap = fig.get_figure()
    heatmap.savefig(fig_path, dpi=400)

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


def neuron_heatmap_cos(
        path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/org_allsft/2500steps/cos_out.csv',
        vmax=1, vmin=0.9993, threshold=0.5):
    sns.set_context({'figure.figsize': [30, 30]})
    title = path.split('/')[-1].strip().replace('.csv', '')
    fig_path = title + '.png'

    pd_data = pd.read_csv(path)
    neuron_name_list = []  # str(neuron_layer + neuron + neuron_weight(in/out))
    abs_corr_list = []
    neuron_name_temp_list = []
    dict_data = {}

    for data in pd_data.iterrows():
        data = data[1]
        neuron_layer = int(float(data['layer_idx']))
        # if neuron_layer != 0:
        #     break

        neuron = int(float(data['neuron_idx']))
        neuron_weight = 'out'
        abs_corr = data['abs_corr']

        if abs_corr >= 0.9995:
            continue

        neuron_name = str(neuron) + '_' + neuron_weight

        if neuron_name not in dict_data.keys():
            dict_data[neuron_name] = [1] * 32
            dict_data[neuron_name][neuron_layer] = abs_corr

        else:
            dict_data[neuron_name][neuron_layer] = abs_corr

        #
        # neuron_name = str(neuron_layer) + '_' + str(neuron) + '_' + neuron_weight
        #
        # dict_data[neuron_name] = abs_corr

    draw_data_list = []

    sorted_dict_list = sort_neuron_list_cos(dict_data.keys())

    label_list = []
    for sorted_dict in sorted_dict_list:
        neuron_name = str(sorted_dict['neuron']) + '_' + sorted_dict['neuron_weight']
        # print(dict_data.keys())
        draw_data_list.append(dict_data[neuron_name])
        label_list.append(neuron_name)

    print(label_list)

    df = pd.DataFrame(data=draw_data_list, index=label_list)
    fig = sns.heatmap(df, vmax=vmax, vmin=vmin, annot=True, cmap="RdBu").set_title(title)
    heatmap = fig.get_figure()
    heatmap.savefig(fig_path, dpi=400)

    plt.show()


# scatter neuron overlap
def neuron_overlap(
        path_1=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/org_allsft/500steps/neuron_dict.json',
        path_2=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/org_allsft/2500steps/neuron_dict.json'):
    sns.set_context({'figure.figsize': [10, 100]})
    title = '1600step-3200step_enzh_train_neuron_only_overlap'
    fig_path = title + '.png'

    f_1 = open(path_1, 'r', encoding='utf-8')
    data_1 = json.load(f_1)

    f_2 = open(path_2, 'r', encoding='utf-8')
    data_2 = json.load(f_2)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(xmax=33, xmin=-1)
    plt.ylim(ymax=11009, ymin=-1)

    colors1 = '#00CED1'  # 点的颜色
    colors2 = '#DC143C'

    area = np.pi * 1 ** 2  # 点面积

    for line in zip(data_1.items(), data_2.items()):
        k_1 = line[0][0]
        k_2 = line[1][0]

        if 'gate' in k_1:
            continue

        y1 = line[0][1]
        y1 = [y for y in y1]

        y2 = line[1][1]
        y2 = [y for y in y2]

        y3 = list(set(y1) & set(y2))

        for y in y3:
            y1.remove(y)
            y2.remove(y)

        x1 = [int(line[0][0].split('_')[0])] * len(y1)
        x2 = [int(line[1][0].split('_')[0])] * len(y2)
        x3 = [int(line[1][0].split('_')[0])] * len(y3)

        print(x3)
        print(y3)

        # plt.scatter(x1, y1, s=area, c='g', alpha=0.4)
        # plt.scatter(x2, y2, s=area, c='b', alpha=0.4)
        plt.scatter(x3, y3, s=area, c='r', alpha=0.4)
        plt.title(title)

        # break

    # plt.savefig(fig_path, dpi=400)
    plt.show()


# overlap neuron dict and show neuron number on each layer
def neuron_overlap_bar(path_list, pic_name):
    x = range(32)
    data = [0] * 32

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

        length = len(overlap)
        # print(length)

        if items[0][0].split('_')[1] == 'in':
            count_in += length

        data[int(items[0][0].split('_')[0])] += length

        total_length += length

    print('total length: ')
    print(total_length)
    # print(count_in)
    # print(total_length - count_in)

    plt.grid(ls="--", alpha=0.5)
    plt.bar(x, data)
    # plt.xlabel('x', fontdict={'fontsize': 12})
    # plt.ylabel('y', fontdict={'fontsize': 12})
    # plt.title(pic_name)
    pic_name = pic_name.replace('\n', '')
    plt.savefig(pic_name + '.pdf', dpi=400)
    plt.show()
    return total_length


# 每一个 区间中 abs_rank_diff_mean
def abs_rank_diff_mean_bar_chart(
        path_dir='/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/rank_diff/cos-100000_cos-5w/max_sorted_add_rank_diff',
        pic_name=''):
    path_in = path_dir + '/max_sorted_add_rank_diff_in.csv'
    path_out = path_dir + '/max_sorted_add_rank_diff_out.csv'
    # all length: 352256
    sns.set_context({'figure.figsize': [8, 6]})

    # end 有小数向前，start 有小数向后，
    # [0: 3522], [3523: 7045], [7046: 10567], [10568: 105676], [105677: 211353], [211354: 317030], [317031: 341688], [341689: 345210], [345211: 348733], [348734: 352256]
    slice_cut = [{'start': 0, 'end': 0.01}, {'start': 0.01, 'end': 0.02}, {'start': 0.02, 'end': 0.03},
                 {'start': 0.03, 'end': 0.1}, {'start': 0.1, 'end': 0.2}, {'start': 0.2, 'end': 0.3},
                 {'start': 0.3, 'end': 0.4}, {'start': 0.4, 'end': 0.5}, {'start': 0.5, 'end': 0.6},
                 {'start': 0.6, 'end': 0.7}, {'start': 0.7, 'end': 0.8}, {'start': 0.8, 'end': 0.9},
                 {'start': 0.9, 'end': 0.97}, {'start': 0.97, 'end': 0.98}, {'start': 0.98, 'end': 0.99},
                 {'start': 0.99, 'end': 1}]

    for igo in ['in', 'out']:
        pic_name_ = pic_name + '_' + igo
        data = [0]
        x = [str(0)]
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
            x.append(str(s['end']))
            df_s = df[start_idx: end_idx]

            abs_rank_diff_mean = df_s['abs_rank_diff'].mean()
            print(abs_rank_diff_mean)
            data.append(abs_rank_diff_mean)
            print('**********************')

        plt.grid(ls="--", alpha=0.5)
        plt.bar(x, data, width=-1, align='edge')
        plt.title(pic_name_)
        pic_name_ = pic_name_.replace('\n', '')
        plt.savefig(pic_name_ + '.png', dpi=400)

        plt.show()


def abs_rank_diff_mean_bar_chart_gather(
        path_dir='/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/rank_diff/cos-100000_cos-5w/max_sorted_add_rank_diff',
        pic_name=''):
    plt.rc('font', family='Times New Roman')
    path_in = path_dir + '/max_sorted_add_rank_diff_in.csv'
    path_out = path_dir + '/max_sorted_add_rank_diff_out.csv'
    # all length: 352256
    # sns.set_context({'figure.figsize': [8.2, 6]})
    sns.set_context({'figure.figsize': [6, 5]})
    # plt.rcParams['font.sans-serif'] = ['Times New Roman']

    # end 有小数向前，start 有小数向后，
    # [0: 3522], [3523: 7045], [7046: 10567], [10568: 105676], [105677: 211353], [211354: 317030], [317031: 341688], [341689: 345210], [345211: 348733], [348734: 352256]
    # slice_cut = [{'start': 0, 'end': 0.01}, {'start': 0.01, 'end': 0.02}, {'start': 0.02, 'end': 0.03},
    #              {'start': 0.03, 'end': 0.1}, {'start': 0.1, 'end': 0.2}, {'start': 0.2, 'end': 0.3},
    #              {'start': 0.3, 'end': 0.4}, {'start': 0.4, 'end': 0.5}, {'start': 0.5, 'end': 0.6},
    #              {'start': 0.6, 'end': 0.7}, {'start': 0.7, 'end': 0.8}, {'start': 0.8, 'end': 0.9},
    #              {'start': 0.9, 'end': 0.97}, {'start': 0.97, 'end': 0.98}, {'start': 0.98, 'end': 0.99},
    #              {'start': 0.99, 'end': 1}]

    slice_cut = [{'start': 0, 'end': 0.01}, {'start': 0.01, 'end': 0.03},
                 {'start': 0.03, 'end': 0.1}, {'start': 0.1, 'end': 0.3}, {'start': 0.3, 'end': 0.5},
                 {'start': 0.5, 'end': 0.7}, {'start': 0.7, 'end': 0.9}, {'start': 0.9, 'end': 0.97},
                 {'start': 0.97, 'end': 0.99}, {'start': 0.99, 'end': 1}]

    data = [0]
    x = [str(0)]

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
        x.append(str(s['end']))
        df_s = df3[start_idx: end_idx]
        # print(df_s)
        abs_rank_diff_mean = df_s['abs_rank_diff'].mean()
        print(abs_rank_diff_mean)
        data.append(abs_rank_diff_mean)
        print('**********************')
    plt.grid(ls="--", alpha=0.5)
    plt.bar(x, data, width=-1, align='edge', label='${\mathrm{Avg}(\Delta\mathbf{Rank}})$\nin each proportion')
    plt.axhline(abs_rank_diff_mean_all, c='orange', label='${\mathrm{Avg}(\Delta\mathbf{Rank}})$\non the whole')

    # plt.title(pic_name, loc='center', fontsize='20', fontweight='bold')
    plt.legend(fontsize=15)
    pic_name = pic_name.replace('\n', '')
    # plt.tick_params(labelsize=15)
    plt.tick_params('x', labelsize=15, direction='out')
    plt.tick_params('y', labelsize=15, direction='in')
    plt.minorticks_off()
    plt.tight_layout()

    # plt.subplots_adjust(left=0.08, right=0.99, top=0.99, bottom=0.05)
    # plt.savefig(pic_name + '.pdf')
    plt.show()


def abs_rank_diff_layer_number_bar_chart_series(
        path_dir='/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/rank_diff/cos-100000_cos-5w/max_sorted_add_rank_diff',
        pic_name=''):
    path_in = path_dir + '/max_sorted_add_rank_diff_in.csv'
    path_out = path_dir + '/max_sorted_add_rank_diff_out.csv'
    # all length: 352256
    sns.set_context({'figure.figsize': [38, 15]})

    # end 有小数向前，start 有小数向后，
    # [0: 3522], [3523: 7045], [7046: 10567], [10568: 105676], [105677: 211353], [211354: 317030], [317031: 341688], [341689: 345210], [345211: 348733], [348734: 352256]
    slice_cut = [{'start': 0, 'end': 0.01}, {'start': 0.01, 'end': 0.02}, {'start': 0.02, 'end': 0.03},
                 {'start': 0.03, 'end': 0.1}, {'start': 0.1, 'end': 0.2}, {'start': 0.2, 'end': 0.3},
                 {'start': 0.3, 'end': 0.4}, {'start': 0.4, 'end': 0.5}, {'start': 0.5, 'end': 0.6},
                 {'start': 0.6, 'end': 0.7}, {'start': 0.7, 'end': 0.8}, {'start': 0.8, 'end': 0.9},
                 {'start': 0.9, 'end': 0.97}, {'start': 0.97, 'end': 0.98}, {'start': 0.98, 'end': 0.99},
                 {'start': 0.99, 'end': 1}]

    for igo in ['in']:
        percentage_dict_list = []
        pic_name_ = pic_name + '_' + igo

        x = [str(i) for i in range(32)]
        if igo == 'in':
            path = path_in
        elif igo == 'out':
            path = path_out

        df = pd.read_csv(path)
        row_number = df.shape[0]
        # print(row_number)
        data_list = []
        title_list = []
        for s in slice_cut:
            title_list.append(str(s['start']) + '~' + str(s['end']))

            start_idx = math.ceil(row_number * s['start'])
            end_idx = math.floor(row_number * s['end'])
            df_s = df[start_idx: end_idx]
            length = len(df_s)
            # print(length)

            layer_counts = df_s['neuron_layer'].value_counts()
            # print(layer_counts)
            layer_counts = sorted(layer_counts.items(), key=lambda d: d[0], reverse=False)
            data_dict = {}
            for item in layer_counts:
                layer = item[0]
                number = item[1]

                data_dict[str(layer)] = number / length

                # data.append(number / length)

            data_list.append(data_dict)

        fig, axes = plt.subplots(4, 4)
        for idx, v in enumerate(zip(axes.flatten(), data_list)):
            ax = v[0]
            data = []
            data_dict = v[1]
            for i in range(32):
                if str(i) in data_dict.keys():
                    data.append(data_dict[str(i)])
                else:
                    data.append(0)

            # temp_data = data_list[idx]
            # print(temp_data)
            # print(data)
            # print(max(data))

            if 0.1 < max(data) < 0.2:
                ax.set_ylim((0, 0.2))
                ax.tick_params("y", which="minor", width=0.05, colors="0.25")

            elif 0.2 < max(data) < 0.4:
                ax.set_ylim((0, 0.25))
                ax.tick_params("y", which="minor", width=0.05, colors="0.25")
            elif max(data) > 0.4:
                ax.set_ylim((0, 0.55))
                ax.tick_params("y", which="minor", width=0.2, colors="0.25")

            else:
                ax.set_ylim((0, 0.065))
                ax.tick_params("y", which="minor", width=0.02, colors="0.25")

            ax.set_title(title_list[idx])
            plt.grid(ls="--", alpha=0.5)
            # print(x)
            # print(data)
            ax.bar(x, data, width=0.8)

        # plt.title(pic_name_)
        pic_name_ = pic_name_.replace('\n', '')
        plt.savefig(pic_name_ + '.png')

        plt.show()


def show_suppressed_neuron(
        path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/enzh_org-enzh_3200step_full/abs_diff.csv'):
    # 'neuron_layer', 'neuron', 'activation_layer', 'neuron_weight', 'abs_corr_1', 'abs_corr_2', 'diff', 'abs_diff'

    title = 'enzh_org-enzh_3200step_train_neuron-pearson_abs_diff-threhold_0.01-show_suppressed_neuron'
    fig_path = title + '.png'
    sns.set_context({'figure.figsize': [10, 15]})
    pd_data = pd.read_csv(path)
    # print(pd_data)

    neuron_list_dict = {}

    area = np.pi * 1 ** 2

    x_list_b = []
    y_list_b = []

    x_list_r = []
    y_list_r = []

    for data in pd_data.iterrows():
        data = data[1]

        neuron_layer = data[0]
        neuron = data[1]
        activation_layer = data[2]
        neuron_weight = data[3]
        abs_corr_1 = data[4]
        abs_corr_2 = data[5]
        diff = data[6]
        abs_diff = data[7]
        neuron_name = str(neuron_layer) + '_' + str(neuron) + '_' + str(activation_layer)
        if abs_diff < 0.01:
            # print(neuron_name)
            continue

        # print(neuron_name)

        if diff >= 0:
            flag = 1
        else:
            flag = -1

        if neuron_name not in neuron_list_dict.keys():
            neuron_list_dict[neuron_name] = flag
            # x_list.append(int(neuron_layer))
            # y_list.append(int(neuron))

            if flag == 1:
                x_list_b.append(neuron_layer)
                y_list_b.append(neuron)

            else:
                x_list_r.append(neuron_layer)
                y_list_r.append(neuron)

        else:
            if neuron_list_dict[neuron_name] != flag:
                print(neuron_name)
                print(neuron_list_dict[neuron_name])
                print(flag)
                break

    plt.scatter(x_list_b, y_list_b, s=area, c='b', alpha=0.4)
    plt.scatter(x_list_r, y_list_r, s=area, c='r', alpha=0.4)

    plt.title(title)

    plt.savefig(fig_path, dpi=400)
    plt.show()


def run_1():
    model_1 = 'cos-100000'
    model_2 = 'cos-50000'
    # abs_rank_diff_larger_10W_prefix = '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/rank_diff' + '/' + model_1 + '_' + model_2
    abs_rank_diff_larger_10W_prefix = '/data2/haoyun/MA_project/neurons/rank_diff/enzh-enzh/frzh_100000_find_49305-enzh_100000_find_49305'

    abs_rank_diff_larger_10W = abs_rank_diff_larger_10W_prefix + '/abs_rank_diff_larger_10W.json'
    abs_rank_diff_larger_10W_rise = abs_rank_diff_larger_10W_prefix + '/abs_rank_diff_larger_10W_rise.json'
    abs_rank_diff_larger_10W_fall = abs_rank_diff_larger_10W_prefix + '/abs_rank_diff_larger_10W_fall.json'

    # overlap_abs_rank_diff_larger_10W_and_model_1_neuron_dict = abs_rank_diff_larger_10W_prefix + '/overlap_abs-rank-diff-larger-10W_' + model_1 + '_neuron_dict.json'
    # disjoint_abs_rank_diff_larger_10W_and_model_1_neuron_dict = abs_rank_diff_larger_10W_prefix + '/disjoint_abs-rank-diff-larger-10W_' + model_1 + '_neuron_dict.json'

    max_sorted_add_rank_diff = abs_rank_diff_larger_10W_prefix + '/max_sorted_add_rank_diff'

    # total_number = neuron_overlap_bar([abs_rank_diff_larger_10W] * 2,
    #                                   pic_name='Neurons that rank difference larger than 10w\n Between Cos-10w And Cos-5w')
    # rise_number = neuron_overlap_bar([abs_rank_diff_larger_10W_rise] * 2,
    #                                  pic_name='Neurons that rise in rank by more than 10w\n Between Cos-10w And Cos-5w')
    # print(rise_number / total_number)
    #
    # fall_number = neuron_overlap_bar([abs_rank_diff_larger_10W_fall] * 2,
    #                                  pic_name='Neurons that drop in rank by more than 10w\n Between Cos-10w And Cos-5w')
    # print(fall_number / total_number)
    #
    # overlap_number = neuron_overlap_bar([overlap_abs_rank_diff_larger_10W_and_model_1_neuron_dict] * 2,
    #                                     pic_name='Neurons that respond strongly to weight updates\n Between Cos-10w And Cos-5w')
    # print(overlap_number / total_number)
    #
    # disjoint_number = neuron_overlap_bar([disjoint_abs_rank_diff_larger_10W_and_model_1_neuron_dict] * 2,
    #                                      pic_name='Neurons that are indirectly affected and respond strongly\n Between Cos-10w And Cos-5w')
    # print(disjoint_number / total_number)

    # abs_rank_diff_mean_bar_chart(path_dir=max_sorted_add_rank_diff, pic_name='max_sorted_add_rank_diff')
    abs_rank_diff_mean_bar_chart_gather(path_dir=max_sorted_add_rank_diff,
                                        pic_name='Average Rank Difference In Each Proportion\n Between Cos-10w And Cos-5w')
    # abs_rank_diff_layer_number_bar_chart_series(max_sorted_add_rank_diff,
    #                                             pic_name='layer_number_in_every_percentage_range\n Between Cos-10w And Cos-5w')


# rank diff 5w
def run_2():
    model_1 = 'cos-150000'
    model_2 = 'cos-5w'
    abs_rank_diff_larger_5W_prefix = '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/rank_diff' + '/' + model_1 + '_' + model_2 + '/5w'
    abs_rank_diff_larger_5W = abs_rank_diff_larger_5W_prefix + '/abs_rank_diff_larger_5W.json'
    abs_rank_diff_larger_5W_rise = abs_rank_diff_larger_5W_prefix + '/abs_rank_diff_larger_5W_rise.json'
    abs_rank_diff_larger_5W_fall = abs_rank_diff_larger_5W_prefix + '/abs_rank_diff_larger_5W_fall.json'
    overlap_abs_rank_diff_larger_5W_and_model_1_neuron_dict = abs_rank_diff_larger_5W_prefix + '/overlap_abs-rank-diff-larger-5W_' + model_1 + '_neuron_dict.json'
    disjoint_abs_rank_diff_larger_5W_and_model_1_neuron_dict = abs_rank_diff_larger_5W_prefix + '/disjoint_abs-rank-diff-larger-5W_' + model_1 + '_neuron_dict.json'

    neuron_overlap_bar([abs_rank_diff_larger_5W] * 2,
                       pic_name='Neurons that rank difference larger than 5w\n(cos-10w)')
    neuron_overlap_bar([abs_rank_diff_larger_5W_rise] * 2,
                       pic_name='Neurons that rise in rank by more than 5w\n(cos-10w)')
    neuron_overlap_bar([abs_rank_diff_larger_5W_fall] * 2,
                       pic_name='Neurons that drop in rank by more than 5w\n(cos-10w)')

    neuron_overlap_bar([overlap_abs_rank_diff_larger_5W_and_model_1_neuron_dict] * 2,
                       pic_name='Neurons that respond strongly to weight updates')
    neuron_overlap_bar([disjoint_abs_rank_diff_larger_5W_and_model_1_neuron_dict] * 2,
                       pic_name='Neurons that are indirectly affected and respond strongly')


# 折线图 + 柱状图
def run_3():
    sns.set_context({'figure.figsize': [15, 15]})

    # 柱状图
    x = ['50000+Reversed_5w', '50000+Reversed_7.5w', '50000+Reversed_10w',
         '50000+Reversed_12.5w', '50000+Reversed_15w']
    rank_diff_mean = [36529.870, 36090.601, 38135.431, 37769.074, 39948.416]

    # plt.bar(x=x, height=rank_diff_mean, width=0.3,color=['r','g','b','r','g','b'])
    plt.bar(x=x, height=rank_diff_mean, width=0.6, label='Rank Difference')

    plt.xticks(range(len(x)), x)
    plt.ylim((30000, 40000))
    plt.ylabel('Absolute Rank Difference')

    ax2 = plt.twinx()
    ax2.set_ylabel("Bleu Score Difference")
    y = [0.753, 0.751, 0.964, 1.117, 1.3434]
    # 设置坐标轴范围
    ax2.set_ylim([0, 1.5])
    plt.plot(x, y, '-', c='y', marker='.', label="Bleu", linewidth='5')

    plt.show()


def run_4():
    sns.set_context({'figure.figsize': [15, 15]})

    # 柱状图
    x = ['cos-10w', 'cos-12.5w', 'cos-15w',
         'cos-17.5w', 'cos-20w']
    rank_diff_mean = [17172.437, 18860.412, 22295.459, 27005.246, 26135.555]

    # plt.bar(x=x, height=rank_diff_mean, width=0.3,color=['r','g','b','r','g','b'])
    plt.bar(x=x, height=rank_diff_mean, width=0.6, label='Rank Difference')

    plt.xticks(range(len(x)), x)
    plt.ylim((15000, 28000))
    plt.ylabel('Average Rank Difference')

    ax2 = plt.twinx()
    ax2.set_ylabel("Bleu Score Difference")
    y = [0.333, 0.370, 0.356, 0.486, 0.056]
    # 设置坐标轴范围
    ax2.set_ylim([0, 1.5])
    plt.plot(x, y, '-', c='y', marker='.', label="Bleu", linewidth='5')

    plt.show()


def run_5():
    rc["tick.labelweight"] = "bold"
    sns.set_context({'figure.figsize': [24, 28]})

    title = 'Comparison Of Rank Difference With cos-5w'
    fig_path = title + '.png'

    # 柱状图
    x = ['cos-10w', 'NeFT${_{3\%}}$+Reversed-5w', 'cos-12.5w', 'NeFT${_{3\%}}$+Reversed-7.5w',
         'cos-15w', 'NeFT${_{3\%}}$+Reversed-10w', 'cos-17.5w', 'NeFT${_{3\%}}$+Reversed-12.5w',
         'cos-20w', 'NeFT${_{3\%}}$+Reversed-15w']

    # x = ['cos-10w', 'NeFT${_{3\%}}$+Reversed_5w', 'cos-12.5w', 'NeFT${_{3\%}}$+Reversed_7.5w',
    #      'cos-15w', 'NeFT${_{3\%}}$+Reversed_10w', 'cos-17.5w', 'NeFT${_{3\%}}$+Reversed_12.5w',
    #      'cos-20w', 'NeFT${_{3\%}}$+Reversed_15w']

    rank_diff_mean = [17172.437, 36529.870, 18860.412, 36090.601, 22295.459, 38135.431, 27005.246, 37769.074, 26135.555,
                      39948.416]

    # plt.bar(x=x, height=rank_diff_mean, width=0.3, color=['r','g','b','r','g','b'])
    plt.bar(x=x, height=rank_diff_mean, width=0.6, label='Rank Difference', align='edge',
            # color=['g', 'y', 'g', 'y', 'g', 'y', 'g', 'y', 'g', 'y']
            color=['g', 'y', 'g', 'y', 'g', 'y', 'g', 'y', 'g', 'y'])

    # plt.xticks(range(len(x)), x)
    # plt.ylim((15000, 42000))
    # plt.ylabel('Average Rank Difference')
    #
    # ax2 = plt.twinx()
    # ax2.set_ylabel("Bleu Score")
    # y = [28.674, 27.5883, 28.711, 27.590, 28.697, 27.377, 28.827, 27.224, 28.397, 26.996]
    # # 设置坐标轴范围
    # ax2.set_ylim([26, 30])
    # plt.plot(x, y, '-', c='b', marker='.', label="Bleu", linewidth='5')

    plt.title(title, fontsize=45)

    plt.tick_params(labelsize=30)
    plt.xticks(rotation=39)
    plt.ylabel("Rank Difference", fontsize=45)
    plt.savefig(fig_path, dpi=400)

    plt.show()


def run_6():
    # rc["tick.labelweight"] = "bold"
    import matplotlib
    import random
    import matplotlib.pyplot as plt
    # plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['figure.figsize'] = [5, 6]

    # 中文乱码和坐标轴负号处理。

    # 城市数据。
    name = ['NeFT${_{6\%}}$\nNeFT${_{3\%}}$', 'NeFT${_{3\%}}$+Reversed${_{3\%}}$\nNeFT${_{3\%}}$',
            'NeFT${_{7\%}}$\nNeFT${_{3\%}}$', 'NeFT${_{3\%}}$+Reversed${_{4\%}}$\nNeFT${_{3\%}}$',
            'NeFT${_{9\%}}$\nNeFT${_{3\%}}$', 'NeFT${_{3\%}}$+Reversed${_{6\%}}$\nNeFT${_{3\%}}$',
            'NeFT${_{10\%}}$\nNeFT${_{3\%}}$', 'NeFT${_{3\%}}$+Reversed${_{7\%}}$\nNeFT${_{3\%}}$',
            'NeFT${_{12\%}}$\nNeFT${_{3\%}}$', 'NeFT${_{3\%}}$+Reversed${_{9\%}}$\nNeFT${_{3\%}}$']

    # 数组反转。
    name.reverse()

    # 装载随机数据。
    data = [17172.437, 36529.870, 18860.412, 36090.601, 22295.459, 38135.431, 27005.246, 37769.074, 26135.555,
            39948.416]
    data.reverse()
    # 绘图。
    fig, ax = plt.subplots()
    b = ax.barh(range(len(name)), data, color=['y', 'g', 'y', 'g', 'y', 'g', 'y', 'g', 'y', 'g'],
                height=0.5)  # '#6699CC'

    # 为横向水平的柱图右侧添加数据标签。
    for rect in b:
        w = rect.get_width()
        ax.text(w - 3000, rect.get_y() + rect.get_height() / 2, '%d' %
                int(w), ha='left', va='center', fontsize='10.5')

    # 设置Y轴纵坐标上的刻度线标签。
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ax.minorticks_off()
    # ax.get_yaxis().set_visible(False)
    ax.set_yticklabels(name)
    plt.tick_params('x', labelsize=13, direction='out')
    plt.tick_params('y', labelsize=13, direction='in')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.grid(None)

    # 不要X横坐标上的label标签。
    # plt.xticks(())
    title = 'Comparison Of Average Rank Difference With cos-5w'
    fig_path = title + '.pdf'

    plt.tight_layout()
    plt.savefig(fig_path)

    plt.show()


def run_7():
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = [5, 5]
    name = ['NeFT${_{12\%}}$\nNeFT${_{6\%}}$', 'NeFT${_{12\%}}$\nNeFT${_{7\%}}$', 'NeFT${_{12\%}}$\nNeFT${_{9\%}}$',
            'NeFT${_{12\%}}$\nNeFT${_{10\%}}$',
            'NeFT${_{3\%}}$+Reversed${_{9\%}}$\nNeFT${_{3\%}}$+Reversed${_{3\%}}$',
            'NeFT${_{3\%}}$+Reversed${_{9\%}}$\nNeFT${_{3\%}}$+Reversed${_{4\%}}$',
            'NeFT${_{3\%}}$+Reversed${_{9\%}}$\nNeFT${_{3\%}}$+Reversed${_{6\%}}$',
            'NeFT${_{3\%}}$+Reversed${_{9\%}}$\nNeFT${_{3\%}}$+Reversed${_{7\%}}$']
    name.reverse()
    data = [23534.646, 21680.398, 21757.541, 22055.132, 15331.863, 14107.380, 9762.998, 9989.181]
    data.reverse()
    fig, ax = plt.subplots()
    b = ax.barh(range(len(name)), data, color=['y', 'y', 'y', 'y', 'g', 'g', 'g', 'g'], height=0.5)  # '#6699CC'

    for rect in b:
        w = rect.get_width()
        ax.text(w - 2000, rect.get_y() + rect.get_height() / 2, '%d' %
                int(w), ha='left', va='center', fontsize='10.5')

    ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7])
    ax.minorticks_off()
    ax.set_yticklabels(name)
    plt.tick_params('x', labelsize=13, direction='out')
    plt.tick_params('y', labelsize=13, direction='in')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.grid(None)

    # 不要X横坐标上的label标签。
    # plt.xticks(())
    title = 'Average Rank Difference'
    fig_path = title + '.pdf'
    plt.tight_layout()
    plt.savefig(fig_path)

    plt.show()


def run_6_and_7():
    plt.rc('font', family='Times New Roman')
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.rcParams['figure.figsize'] = [5.8, 5.8]
    fig = plt.figure(facecolor='white')
    ax1 = fig.add_subplot(1, 2, 1, facecolor='white')
    ax2 = fig.add_subplot(1, 2, 2, facecolor='white')

    # name = ['NeFT${_{6\%}}$\nNeFT${_{3\%}}$',
    #         'NeFT${_{7\%}}$\nNeFT${_{3\%}}$',
    #         'NeFT${_{9\%}}$\nNeFT${_{3\%}}$',
    #         'NeFT${_{10\%}}$\nNeFT${_{3\%}}$',
    #         'NeFT${_{12\%}}$\nNeFT${_{3\%}}$',
    #         'NeFT${_{3\%}}$+Reversed${_{3\%}}$\nNeFT${_{3\%}}$',
    #         'NeFT${_{3\%}}$+Reversed${_{4\%}}$\nNeFT${_{3\%}}$',
    #         'NeFT${_{3\%}}$+Reversed${_{6\%}}$\nNeFT${_{3\%}}$',
    #         'NeFT${_{3\%}}$+Reversed${_{7\%}}$\nNeFT${_{3\%}}$',
    #         'NeFT${_{3\%}}$+Reversed${_{9\%}}$\nNeFT${_{3\%}}$']

    name = ['NeFT${_{6\%}}$',
            'NeFT${_{7\%}}$',
            'NeFT${_{9\%}}$',
            'NeFT${_{10\%}}$',
            'NeFT${_{12\%}}$',
            'NeFT${_{3\%}}$+Reversed${_{3\%}}$',
            'NeFT${_{3\%}}$+Reversed${_{4\%}}$',
            'NeFT${_{3\%}}$+Reversed${_{6\%}}$',
            'NeFT${_{3\%}}$+Reversed${_{7\%}}$',
            'NeFT${_{3\%}}$+Reversed${_{9\%}}$']

    # 数组反转。
    name.reverse()

    # 装载随机数据。
    # data = [17172.437, 36529.870, 18860.412, 36090.601, 22295.459, 38135.431, 27005.246, 37769.074, 26135.555,
    #         39948.416]

    data = [17172.437, 18860.412, 22295.459, 27005.246, 26135.555, 36529.870, 36090.601, 38135.431, 37769.074,
            39948.416]
    data.reverse()
    # 绘图。
    # fig, ax = plt.subplots()
    # b = ax1.barh(range(len(name)), data, color=['y', 'g', 'y', 'g', 'y', 'g', 'y', 'g', 'y', 'g'],
    #              height=0.5)  # '#6699CC'
    b = ax1.barh(range(len(name)), data, color=['y', 'y', 'y', 'y', 'y', 'g', 'g', 'g', 'g', 'g'],
                 height=0.5)  # '#6699CC'

    # 为横向水平的柱图右侧添加数据标签。
    for rect in b:
        w = rect.get_width()
        ax1.text(w, rect.get_y() + rect.get_height() / 2, '%d' %
                 int(w), ha='left', va='center', fontsize='10.5')

    # 设置Y轴纵坐标上的刻度线标签。
    ax1.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ax1.minorticks_off()
    # ax.get_yaxis().set_visible(False)
    ax1.set_yticklabels(name)
    ax1.tick_params('x', labelsize=13, direction='out')
    ax1.tick_params('y', labelsize=13, direction='in')

    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.grid(None)
    # ax1.set_title('${\mathbf{[\:A.]}}$  ${\mathrm{Avg}(\Delta\mathbf{Rank}})$ with NeFT${_{3\%}}$', x=0, y=1.03, size=15)
    ax1.set_title('(a) Avg $\Delta$Rank with NeFT${_{3\%}}$', x=-0.58, y=-0.145, size=15)

    # name = ['NeFT${_{12\%}}$\nNeFT${_{6\%}}$', 'NeFT${_{12\%}}$\nNeFT${_{7\%}}$', 'NeFT${_{12\%}}$\nNeFT${_{9\%}}$',
    #         'NeFT${_{12\%}}$\nNeFT${_{10\%}}$',
    #         'NeFT${_{3\%}}$+Reversed${_{9\%}}$\nNeFT${_{3\%}}$+Reversed${_{3\%}}$',
    #         'NeFT${_{3\%}}$+Reversed${_{9\%}}$\nNeFT${_{3\%}}$+Reversed${_{4\%}}$',
    #         'NeFT${_{3\%}}$+Reversed${_{9\%}}$\nNeFT${_{3\%}}$+Reversed${_{6\%}}$',
    #         'NeFT${_{3\%}}$+Reversed${_{9\%}}$\nNeFT${_{3\%}}$+Reversed${_{7\%}}$']

    name = ['NeFT${_{6\%}}$', 'NeFT${_{7\%}}$', 'NeFT${_{9\%}}$',
            'NeFT${_{10\%}}$',
            'NeFT${_{3\%}}$+Reversed${_{3\%}}$',
            'NeFT${_{3\%}}$+Reversed${_{4\%}}$',
            'NeFT${_{3\%}}$+Reversed${_{6\%}}$',
            'NeFT${_{3\%}}$+Reversed${_{7\%}}$']

    name.reverse()
    data = [23534.646, 21680.398, 21757.541, 22055.132, 15331.863, 14107.380, 9762.998, 9989.181]
    data.reverse()
    b = ax2.barh(range(len(name)), data, color=['y', 'y', 'y', 'y', 'g', 'g', 'g', 'g'], height=0.5)  # '#6699CC'

    for rect in b:
        w = rect.get_width()
        ax2.text(w, rect.get_y() + rect.get_height() / 2, '%d' %
                 int(w), ha='left', va='center', fontsize='10.5')

    ax2.set_yticks([0, 1, 2, 3, 4, 5, 6, 7])
    ax2.minorticks_off()
    ax2.set_yticklabels(name)
    ax2.tick_params('x', labelsize=13, direction='out')
    ax2.tick_params('y', labelsize=13, direction='in')

    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.grid(None)
    # ax2.set_title('${\mathbf{[\:B.]}}$  ${\mathrm{Avg}(\Delta\mathbf{Rank}})$ with NeFT${_{12\%}}$ &\n${\mathrm{Avg}(\Delta\mathbf{Rank}})$ with NeFT${_{3\%}}$+Reversed${_{9\%}}$',
    #     x=0, y=1.03, size=15)
    ax2.set_title(
        '(b) Avg $\Delta$Rank with NeFT${_{12\%}}$ &\nAvg $\Delta$Rank with NeFT${_{3\%}}$+Reversed${_{9\%}}$',
        x=-0.48, y=-0.2, size=15)

    title = 'Average Rank gather'
    fig_path = title + '.pdf'
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.show()


def run_7_old():
    import matplotlib.pyplot as plt
    sns.set_context({'figure.figsize': [6, 6]})
    name = ['NeFT${_{12\%}}$\nNeFT${_{6\%}}$', 'NeFT${_{12\%}}$\nNeFT${_{7\%}}$', 'NeFT${_{12\%}}$\nNeFT${_{9\%}}$',
            'NeFT${_{12\%}}$\nNeFT${_{10\%}}$',
            'NeFT${_{3\%}}$+Reversed${_{9\%}}$\nNeFT${_{3\%}}$+Reversed${_{3\%}}$',
            'NeFT${_{3\%}}$+Reversed${_{9\%}}$\nNeFT${_{3\%}}$+Reversed${_{4\%}}$',
            'NeFT${_{3\%}}$+Reversed${_{9\%}}$\nNeFT${_{3\%}}$+Reversed${_{6\%}}$',
            'NeFT${_{3\%}}$+Reversed${_{9\%}}$\nNeFT${_{3\%}}$+Reversed${_{7\%}}$']
    name.reverse()
    data = [23534.646, 21680.398, 21757.541, 22055.132, 15331.863, 14107.380, 9762.998, 9989.181]
    data.reverse()
    fig, ax = plt.subplots()
    b = ax.barh(range(len(name)), data, color=['y', 'y', 'y', 'y', 'g', 'g', 'g', 'g'])  # '#6699CC'

    for rect in b:
        w = rect.get_width()
        ax.text(w - 2000, rect.get_y() + rect.get_height() / 2, '%d' %
                int(w), ha='left', va='center', fontsize='10.5')

    ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7])
    ax.minorticks_off()

    ax.set_yticklabels(name)
    plt.tick_params('x', labelsize=13, direction='out')
    plt.tick_params('y', labelsize=13, direction='in')

    # 不要X横坐标上的label标签。
    # plt.xticks(())
    title = 'Average Rank Difference'

    # plt.title(title, loc='center', fontsize='32',
    #           fontweight='bold')

    fig_path = title + '.pdf'
    # plt.subplots_adjust(left=0.21, right=0.99, top=0.99, bottom=0.05)
    # plt.savefig(fig_path, dpi=400)

    plt.show()


# win\tie\lose
def run_8():
    plt.rc('font', family='Times New Roman')
    category_names = ['NeFT Win', 'LoRA Win']
    # results = {
    #     'En --> Zh (20k)\nEn --> Zh': [3, 1],
    #     'En --> Zh (20k)\nFr --> Zh': [3, 1],
    #     'En --> Zh (20k)\nHi --> Zh': [3, 1],
    #
    #     'Fr --> Zh (20k)\nEn --> Zh': [4, 0],
    #     'Fr --> Zh (20k)\nFr --> Zh': [4, 0],
    #     'Fr --> Zh (20k)\nHi --> Zh': [4, 0],
    #
    #     'Hi --> Zh (4.5k)\nEn --> Zh': [0, 4],
    #     'Hi --> Zh (4.5k)\nFr --> Zh': [1, 3],
    #     'Hi --> Zh (4.5k)\nHi --> Zh': [1, 3]
    # }

    # results = {
    #     'En → Zh': [7, 5],
    #     'Fr → Zh': [8, 4],
    #     'Hi → Zh': [8, 4],
    # }

    results = {
        'En → Zh': [14, 10],
        'Fr → Zh': [14, 10],
        'Hi → Zh': [18, 6],
    }

    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('acton')(np.linspace(0.15, 0.85, data.shape[1]))
    fig, ax = plt.subplots(figsize=(3.5, 1.65))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())
    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.65,
                        label=colname, color=color)
        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        ax.bar_label(rects, label_type='center', color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left')
    plt.tick_params('y', direction='in', labelsize=13)
    plt.minorticks_off()
    plt.tight_layout()

    fig_path = 'NeFT_LoRA_comparison_win_tie_lose.pdf'
    plt.savefig(fig_path)
    plt.show()


def run_9():
    plt.rcParams['figure.figsize'] = [6, 1.5]
    # countries = ['NeFT${_{6\%}}$', 'NeFT${_{7\%}}$', 'NeFT${_{9\%}}$', 'NeFT${_{10\%}}$', 'NeFT${_{12\%}}$', 'NeFT${_{3\%}}$+Reversed${_{3\%}}$', 'NeFT${_{3\%}}$+Reversed${_{4\%}}$', 'NeFT${_{3\%}}$+Reversed${_{6\%}}$', 'NeFT${_{3\%}}$+Reversed${_{7\%}}$', 'NeFT${_{3\%}}$+Reversed${_{9\%}}$']
    # strongly_neuron = [2546, 4294, 8947, 18716, 17129, 54572, 52236, 59262, 57671, 65961]
    # suppressed_neuron = [1221, 2125, 4459, 10292, 8710, 26298, 25456, 28980, 27591, 31686]
    # indirectly_neuron = [1550, 2794, 5990, 13465, 11234, 44439, 39911, 42855, 39488, 42619]

    # NeFT_x% with NeFT_3%
    countries = ['NeFT${_{6\%}}$', 'NeFT${_{7\%}}$', 'NeFT${_{9\%}}$', 'NeFT${_{10\%}}$', 'NeFT${_{12\%}}$']
    strongly_neuron = [2546, 4294, 8947, 18716, 17129]
    suppressed_neuron = [1221, 2125, 4459, 10292, 8710]
    indirectly_neuron = [1550, 2794, 5990, 13465, 11234]

    # NeFT_3%+Reversed_x% with NeFT_3%
    # countries = ['NeFT${_{3\%}}$+Reversed${_{3\%}}$', 'NeFT${_{3\%}}$+Reversed${_{4\%}}$',
    #              'NeFT${_{3\%}}$+Reversed${_{6\%}}$', 'NeFT${_{3\%}}$+Reversed${_{7\%}}$',
    #              'NeFT${_{3\%}}$+Reversed${_{9\%}}$']
    # strongly_neuron = [54572, 52236, 59262, 57671, 65961]
    # suppressed_neuron = [26298, 25456, 28980, 27591, 31686]
    # indirectly_neuron = [44439, 39911, 42855, 39488, 42619]

    # NeFT_12% with NeFT_x%
    # countries = ['NeFT${_{6\%}}$', 'NeFT${_{7\%}}$', 'NeFT${_{9\%}}$', 'NeFT${_{10\%}}$']
    # strongly_neuron = [10396, 6930, 6452, 6473]
    # suppressed_neuron = [5670, 3971, 3306, 2797]
    # indirectly_neuron = [6680, 4309, 4276, 4577]

    # NeFT_3%+Reversed_9% with NeFT_3%+Reversed_x%
    # countries = ['NeFT${_{3\%}}$+Reversed${_{3\%}}$', 'NeFT${_{3\%}}$+Reversed${_{4\%}}$',
    #              'NeFT${_{3\%}}$+Reversed${_{6\%}}$', 'NeFT${_{3\%}}$+Reversed${_{7\%}}$']
    # strongly_neuron = [2514, 1611, 745, 413]
    # suppressed_neuron = [1069, 615, 244, 146]
    # indirectly_neuron = [245, 112, 13, 14]

    x = np.arange(len(countries))
    width = 0.2
    gold_x = x
    silver_x = x + width
    bronze_x = x + 2 * width

    plt.bar(gold_x, strongly_neuron, width=width, color="gold", label='Strongly')
    plt.bar(silver_x, suppressed_neuron, width=width, color="silver", label='Suppressed')
    plt.bar(bronze_x, indirectly_neuron, width=width, color="saddlebrown", label='Indirectly')

    plt.xticks(x + width, labels=countries, fontsize=13)
    # plt.xticks(x + width, labels=countries, fontsize=13, rotation=15)
    # plt.ylim((25000, 68000))
    plt.tick_params('y', direction='in', labelsize=13)
    plt.minorticks_off()

    for i in range(len(countries)):
        plt.text(gold_x[i], strongly_neuron[i] + 300, strongly_neuron[i], va="bottom", ha="center", fontsize=8)
        plt.text(silver_x[i], suppressed_neuron[i], suppressed_neuron[i], va="bottom", ha="center", fontsize=8)
        plt.text(bronze_x[i], indirectly_neuron[i] + 700, indirectly_neuron[i], va="bottom", ha="center", fontsize=8)
    plt.legend(loc='upper left')
    plt.tight_layout()

    fig_path = 'different_type_neuron_1.pdf'
    plt.savefig(fig_path)
    plt.show()


def run_10():
    plt.rcParams['figure.figsize'] = [6, 2]
    # countries = ['NeFT${_{6\%}}$', 'NeFT${_{7\%}}$', 'NeFT${_{9\%}}$', 'NeFT${_{10\%}}$', 'NeFT${_{12\%}}$', 'NeFT${_{3\%}}$+Reversed${_{3\%}}$', 'NeFT${_{3\%}}$+Reversed${_{4\%}}$', 'NeFT${_{3\%}}$+Reversed${_{6\%}}$', 'NeFT${_{3\%}}$+Reversed${_{7\%}}$', 'NeFT${_{3\%}}$+Reversed${_{9\%}}$']
    # strongly_neuron = [2546, 4294, 8947, 18716, 17129, 54572, 52236, 59262, 57671, 65961]
    # suppressed_neuron = [1221, 2125, 4459, 10292, 8710, 26298, 25456, 28980, 27591, 31686]
    # indirectly_neuron = [1550, 2794, 5990, 13465, 11234, 44439, 39911, 42855, 39488, 42619]

    # NeFT_x% with NeFT_3%
    # countries = ['NeFT${_{6\%}}$', 'NeFT${_{7\%}}$', 'NeFT${_{9\%}}$', 'NeFT${_{10\%}}$', 'NeFT${_{12\%}}$']
    # strongly_neuron = [2546, 4294, 8947, 18716, 17129]
    # suppressed_neuron = [1221, 2125, 4459, 10292, 8710]
    # indirectly_neuron = [1550, 2794, 5990, 13465, 11234]

    # NeFT_3%+Reversed_x% with NeFT_3%
    countries = ['NeFT${_{3\%}}$+Reversed${_{3\%}}$', 'NeFT${_{3\%}}$+Reversed${_{4\%}}$',
                 'NeFT${_{3\%}}$+Reversed${_{6\%}}$', 'NeFT${_{3\%}}$+Reversed${_{7\%}}$',
                 'NeFT${_{3\%}}$+Reversed${_{9\%}}$']
    strongly_neuron = [54572, 52236, 59262, 57671, 65961]
    suppressed_neuron = [26298, 25456, 28980, 27591, 31686]
    indirectly_neuron = [44439, 39911, 42855, 39488, 42619]

    # NeFT_12% with NeFT_x%
    # countries = ['NeFT${_{6\%}}$', 'NeFT${_{7\%}}$', 'NeFT${_{9\%}}$', 'NeFT${_{10\%}}$']
    # strongly_neuron = [10396, 6930, 6452, 6473]
    # suppressed_neuron = [5670, 3971, 3306, 2797]
    # indirectly_neuron = [6680, 4309, 4276, 4577]

    # NeFT_3%+Reversed_9% with NeFT_3%+Reversed_x%
    # countries = ['NeFT${_{3\%}}$+Reversed${_{3\%}}$', 'NeFT${_{3\%}}$+Reversed${_{4\%}}$',
    #              'NeFT${_{3\%}}$+Reversed${_{6\%}}$', 'NeFT${_{3\%}}$+Reversed${_{7\%}}$']
    # strongly_neuron = [2514, 1611, 745, 413]
    # suppressed_neuron = [1069, 615, 244, 146]
    # indirectly_neuron = [245, 112, 13, 14]

    x = np.arange(len(countries))
    width = 0.2
    gold_x = x
    silver_x = x + width
    bronze_x = x + 2 * width

    plt.bar(gold_x, strongly_neuron, width=width, color="gold", label='Strongly')
    plt.bar(silver_x, suppressed_neuron, width=width, color="silver", label='Suppressed')
    plt.bar(bronze_x, indirectly_neuron, width=width, color="saddlebrown", label='Indirectly')

    # plt.xticks(x + width, labels=countries, fontsize=13)
    plt.xticks(x + width, labels=countries, fontsize=13, rotation=15)
    # plt.ylim((20000, 68000))
    plt.tick_params('y', direction='in', labelsize=13)
    plt.minorticks_off()

    for i in range(len(countries)):
        plt.text(gold_x[i], strongly_neuron[i], strongly_neuron[i], va="bottom", ha="center", fontsize=8)
        plt.text(silver_x[i], suppressed_neuron[i], suppressed_neuron[i], va="bottom", ha="center", fontsize=8)
        plt.text(bronze_x[i], indirectly_neuron[i], indirectly_neuron[i], va="bottom", ha="center", fontsize=8)
    # plt.legend(loc='upper left')
    plt.tight_layout()

    fig_path = 'different_type_neuron_2.pdf'
    plt.savefig(fig_path)
    plt.show()


def run_11():
    plt.rcParams['figure.figsize'] = [6, 1.5]
    # countries = ['NeFT${_{6\%}}$', 'NeFT${_{7\%}}$', 'NeFT${_{9\%}}$', 'NeFT${_{10\%}}$', 'NeFT${_{12\%}}$', 'NeFT${_{3\%}}$+Reversed${_{3\%}}$', 'NeFT${_{3\%}}$+Reversed${_{4\%}}$', 'NeFT${_{3\%}}$+Reversed${_{6\%}}$', 'NeFT${_{3\%}}$+Reversed${_{7\%}}$', 'NeFT${_{3\%}}$+Reversed${_{9\%}}$']
    # strongly_neuron = [2546, 4294, 8947, 18716, 17129, 54572, 52236, 59262, 57671, 65961]
    # suppressed_neuron = [1221, 2125, 4459, 10292, 8710, 26298, 25456, 28980, 27591, 31686]
    # indirectly_neuron = [1550, 2794, 5990, 13465, 11234, 44439, 39911, 42855, 39488, 42619]

    # NeFT_x% with NeFT_3%
    # countries = ['NeFT${_{6\%}}$', 'NeFT${_{7\%}}$', 'NeFT${_{9\%}}$', 'NeFT${_{10\%}}$', 'NeFT${_{12\%}}$']
    # strongly_neuron = [2546, 4294, 8947, 18716, 17129]
    # suppressed_neuron = [1221, 2125, 4459, 10292, 8710]
    # indirectly_neuron = [1550, 2794, 5990, 13465, 11234]

    # NeFT_3%+Reversed_x% with NeFT_3%
    # countries = ['NeFT${_{3\%}}$+Reversed${_{3\%}}$', 'NeFT${_{3\%}}$+Reversed${_{4\%}}$',
    #              'NeFT${_{3\%}}$+Reversed${_{6\%}}$', 'NeFT${_{3\%}}$+Reversed${_{7\%}}$',
    #              'NeFT${_{3\%}}$+Reversed${_{9\%}}$']
    # strongly_neuron = [54572, 52236, 59262, 57671, 65961]
    # suppressed_neuron = [26298, 25456, 28980, 27591, 31686]
    # indirectly_neuron = [44439, 39911, 42855, 39488, 42619]

    # NeFT_12% with NeFT_x%
    countries = ['NeFT${_{6\%}}$', 'NeFT${_{7\%}}$', 'NeFT${_{9\%}}$', 'NeFT${_{10\%}}$']
    strongly_neuron = [10396, 6930, 6452, 6473]
    suppressed_neuron = [5670, 3971, 3306, 2797]
    indirectly_neuron = [6680, 4309, 4276, 4577]

    # NeFT_3%+Reversed_9% with NeFT_3%+Reversed_x%
    # countries = ['NeFT${_{3\%}}$+Reversed${_{3\%}}$', 'NeFT${_{3\%}}$+Reversed${_{4\%}}$',
    #              'NeFT${_{3\%}}$+Reversed${_{6\%}}$', 'NeFT${_{3\%}}$+Reversed${_{7\%}}$']
    # strongly_neuron = [2514, 1611, 745, 413]
    # suppressed_neuron = [1069, 615, 244, 146]
    # indirectly_neuron = [245, 112, 13, 14]

    x = np.arange(len(countries))
    width = 0.2
    gold_x = x
    silver_x = x + width
    bronze_x = x + 2 * width

    plt.bar(gold_x, strongly_neuron, width=width, color="gold", label='Strongly')
    plt.bar(silver_x, suppressed_neuron, width=width, color="silver", label='Suppressed')
    plt.bar(bronze_x, indirectly_neuron, width=width, color="saddlebrown", label='Indirectly')

    plt.xticks(x + width, labels=countries, fontsize=13)
    # plt.xticks(x + width, labels=countries, fontsize=13, rotation=15)
    # plt.ylim((20000, 68000))
    plt.tick_params('y', direction='in', labelsize=13)
    plt.minorticks_off()

    for i in range(len(countries)):
        plt.text(gold_x[i], strongly_neuron[i], strongly_neuron[i], va="bottom", ha="center", fontsize=8)
        plt.text(silver_x[i], suppressed_neuron[i], suppressed_neuron[i], va="bottom", ha="center", fontsize=8)
        plt.text(bronze_x[i], indirectly_neuron[i], indirectly_neuron[i], va="bottom", ha="center", fontsize=8)
    # plt.legend(loc='upper left')
    plt.tight_layout()

    fig_path = 'different_type_neuron_3.pdf'
    plt.savefig(fig_path)
    plt.show()


def run_12():
    plt.rcParams['figure.figsize'] = [6, 2]
    # countries = ['NeFT${_{6\%}}$', 'NeFT${_{7\%}}$', 'NeFT${_{9\%}}$', 'NeFT${_{10\%}}$', 'NeFT${_{12\%}}$', 'NeFT${_{3\%}}$+Reversed${_{3\%}}$', 'NeFT${_{3\%}}$+Reversed${_{4\%}}$', 'NeFT${_{3\%}}$+Reversed${_{6\%}}$', 'NeFT${_{3\%}}$+Reversed${_{7\%}}$', 'NeFT${_{3\%}}$+Reversed${_{9\%}}$']
    # strongly_neuron = [2546, 4294, 8947, 18716, 17129, 54572, 52236, 59262, 57671, 65961]
    # suppressed_neuron = [1221, 2125, 4459, 10292, 8710, 26298, 25456, 28980, 27591, 31686]
    # indirectly_neuron = [1550, 2794, 5990, 13465, 11234, 44439, 39911, 42855, 39488, 42619]

    # NeFT_x% with NeFT_3%
    # countries = ['NeFT${_{6\%}}$', 'NeFT${_{7\%}}$', 'NeFT${_{9\%}}$', 'NeFT${_{10\%}}$', 'NeFT${_{12\%}}$']
    # strongly_neuron = [2546, 4294, 8947, 18716, 17129]
    # suppressed_neuron = [1221, 2125, 4459, 10292, 8710]
    # indirectly_neuron = [1550, 2794, 5990, 13465, 11234]

    # NeFT_3%+Reversed_x% with NeFT_3%
    # countries = ['NeFT${_{3\%}}$+Reversed${_{3\%}}$', 'NeFT${_{3\%}}$+Reversed${_{4\%}}$',
    #              'NeFT${_{3\%}}$+Reversed${_{6\%}}$', 'NeFT${_{3\%}}$+Reversed${_{7\%}}$',
    #              'NeFT${_{3\%}}$+Reversed${_{9\%}}$']
    # strongly_neuron = [54572, 52236, 59262, 57671, 65961]
    # suppressed_neuron = [26298, 25456, 28980, 27591, 31686]
    # indirectly_neuron = [44439, 39911, 42855, 39488, 42619]

    # NeFT_12% with NeFT_x%
    # countries = ['NeFT${_{6\%}}$', 'NeFT${_{7\%}}$', 'NeFT${_{9\%}}$', 'NeFT${_{10\%}}$']
    # strongly_neuron = [10396, 6930, 6452, 6473]
    # suppressed_neuron = [5670, 3971, 3306, 2797]
    # indirectly_neuron = [6680, 4309, 4276, 4577]

    # NeFT_3%+Reversed_9% with NeFT_3%+Reversed_x%
    countries = ['NeFT${_{3\%}}$+Reversed${_{3\%}}$', 'NeFT${_{3\%}}$+Reversed${_{4\%}}$',
                 'NeFT${_{3\%}}$+Reversed${_{6\%}}$', 'NeFT${_{3\%}}$+Reversed${_{7\%}}$']
    strongly_neuron = [2514, 1611, 745, 413]
    suppressed_neuron = [1069, 615, 244, 146]
    indirectly_neuron = [245, 112, 13, 14]

    x = np.arange(len(countries))
    width = 0.2
    gold_x = x
    silver_x = x + width
    bronze_x = x + 2 * width

    plt.bar(gold_x, strongly_neuron, width=width, color="gold", label='Strongly')
    plt.bar(silver_x, suppressed_neuron, width=width, color="silver", label='Suppressed')
    plt.bar(bronze_x, indirectly_neuron, width=width, color="saddlebrown", label='Indirectly')

    # plt.xticks(x + width, labels=countries, fontsize=13)
    plt.xticks(x + width, labels=countries, fontsize=13, rotation=15)
    # plt.ylim((20000, 68000))
    plt.tick_params('y', direction='in', labelsize=13)
    plt.minorticks_off()

    for i in range(len(countries)):
        plt.text(gold_x[i], strongly_neuron[i], strongly_neuron[i], va="bottom", ha="center", fontsize=8)
        plt.text(silver_x[i], suppressed_neuron[i], suppressed_neuron[i], va="bottom", ha="center", fontsize=8)
        plt.text(bronze_x[i], indirectly_neuron[i], indirectly_neuron[i], va="bottom", ha="center", fontsize=8)
    # plt.legend(loc='upper left')
    plt.tight_layout()

    fig_path = 'different_type_neuron_4.pdf'
    plt.savefig(fig_path)
    plt.show()


def run_13():
    plt.rcParams['figure.figsize'] = [7, 7]
    fig = plt.figure(facecolor='white')
    ax1 = fig.add_subplot(4, 1, 1, facecolor='white')
    ax2 = fig.add_subplot(4, 1, 2, facecolor='white')
    ax3 = fig.add_subplot(4, 1, 3, facecolor='white')
    ax4 = fig.add_subplot(4, 1, 4, facecolor='white')

    # spec2 = gridspec.GridSpec(ncols=1, nrows=4)
    # ax1 = fig.add_subplot(spec2[0, 0])
    # ax2 = fig.add_subplot(spec2[1, 0])
    # ax3 = fig.add_subplot(spec2[2, 0])
    # ax4 = fig.add_subplot(spec2[3, 0])

    # NeFT_x% with NeFT_3%
    countries = ['NeFT${_{6\%}}$', 'NeFT${_{7\%}}$', 'NeFT${_{9\%}}$', 'NeFT${_{10\%}}$', 'NeFT${_{12\%}}$']
    strongly_neuron = [2546, 4294, 8947, 18716, 17129]
    suppressed_neuron = [1221, 2125, 4459, 10292, 8710]
    indirectly_neuron = [1550, 2794, 5990, 13465, 11234]

    x = np.arange(len(countries))
    width = 0.2
    gold_x = x
    silver_x = x + width
    bronze_x = x + 2 * width

    ax1.bar(gold_x, strongly_neuron, width=width, color="gold", label='Strongly')
    ax1.bar(silver_x, suppressed_neuron, width=width, color="silver", label='Suppressed')
    ax1.bar(bronze_x, indirectly_neuron, width=width, color="saddlebrown", label='Indirectly')

    ax1.set_xticks(x + width)
    ax1.set_xticklabels(labels=countries, fontsize=15)
    ax1.tick_params('y', direction='in', labelsize=15)
    ax1.minorticks_off()

    for i in range(len(countries)):
        ax1.text(gold_x[i], strongly_neuron[i] - 800, strongly_neuron[i], va="bottom", ha="center", fontsize=8)
        ax1.text(silver_x[i], suppressed_neuron[i], suppressed_neuron[i], va="bottom", ha="center", fontsize=8)
        ax1.text(bronze_x[i], indirectly_neuron[i] + 200, indirectly_neuron[i], va="bottom", ha="center", fontsize=8)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    ax1.set_title('NeFT${_{x\%}}$ ${\Delta\mathbf{Rank}}$ with NeFT${_{3\%}}$', size=15)

    # # NeFT_3%+Reversed_x% with NeFT_3%
    countries = ['NeFT${_{3\%}}$+Reversed${_{3\%}}$', 'NeFT${_{3\%}}$+Reversed${_{4\%}}$',
                 'NeFT${_{3\%}}$+Reversed${_{6\%}}$', 'NeFT${_{3\%}}$+Reversed${_{7\%}}$',
                 'NeFT${_{3\%}}$+Reversed${_{9\%}}$']
    strongly_neuron = [54572, 52236, 59262, 57671, 65961]
    suppressed_neuron = [26298, 25456, 28980, 27591, 31686]
    indirectly_neuron = [44439, 39911, 42855, 39488, 42619]

    x = np.arange(len(countries))
    width = 0.2
    gold_x = x
    silver_x = x + width
    bronze_x = x + 2 * width

    ax2.bar(gold_x, strongly_neuron, width=width, color="gold", label='Strongly')
    ax2.bar(silver_x, suppressed_neuron, width=width, color="silver", label='Suppressed')
    ax2.bar(bronze_x, indirectly_neuron, width=width, color="saddlebrown", label='Indirectly')

    ax2.set_xticks(x + width)
    ax2.set_xticklabels(labels=countries, fontsize=15, rotation=15)
    ax2.tick_params('y', direction='in', labelsize=15)
    # plt.ylim((20000, 68000))
    ax2.tick_params('y', direction='in', labelsize=15)
    ax2.minorticks_off()

    for i in range(len(countries)):
        ax2.text(gold_x[i], strongly_neuron[i], strongly_neuron[i], va="bottom", ha="center", fontsize=8)
        ax2.text(silver_x[i], suppressed_neuron[i], suppressed_neuron[i], va="bottom", ha="center", fontsize=8)
        ax2.text(bronze_x[i], indirectly_neuron[i], indirectly_neuron[i], va="bottom", ha="center", fontsize=8)

    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_title('NeFT${_{3\%}}$+Reversed${_{x\%}}$ ${\Delta\mathbf{Rank}}$ with NeFT${_{3\%}}$', size=15)

    # NeFT_12% with NeFT_x%
    countries = ['NeFT${_{6\%}}$', 'NeFT${_{7\%}}$', 'NeFT${_{9\%}}$', 'NeFT${_{10\%}}$']
    strongly_neuron = [10396, 6930, 6452, 6473]
    suppressed_neuron = [5670, 3971, 3306, 2797]
    indirectly_neuron = [6680, 4309, 4276, 4577]

    # NeFT_3%+Reversed_9% with NeFT_3%+Reversed_x%
    x = np.arange(len(countries))
    width = 0.2
    gold_x = x
    silver_x = x + width
    bronze_x = x + 2 * width

    strongly = ax3.bar(gold_x, strongly_neuron, width=width, color="gold", label='Strongly')
    suppressed = ax3.bar(silver_x, suppressed_neuron, width=width, color="silver", label='Suppressed')
    indirectly = ax3.bar(bronze_x, indirectly_neuron, width=width, color="saddlebrown", label='Indirectly')

    ax3.set_xticks(x + width)
    ax3.set_xticklabels(labels=countries, fontsize=15)
    ax3.tick_params('y', direction='in', labelsize=15)
    ax3.minorticks_off()

    for i in range(len(countries)):
        ax3.text(gold_x[i], strongly_neuron[i], strongly_neuron[i], va="bottom", ha="center", fontsize=8)
        ax3.text(silver_x[i], suppressed_neuron[i], suppressed_neuron[i], va="bottom", ha="center", fontsize=8)
        ax3.text(bronze_x[i], indirectly_neuron[i], indirectly_neuron[i], va="bottom", ha="center", fontsize=8)

    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.set_title('${\Delta\mathbf{Rank}}$ with NeFT${_{12\%}}$', size=15)

    # NeFT_3%+Reversed_9% with NeFT_3%+Reversed_x%
    countries = ['NeFT${_{3\%}}$+Reversed${_{3\%}}$', 'NeFT${_{3\%}}$+Reversed${_{4\%}}$',
                 'NeFT${_{3\%}}$+Reversed${_{6\%}}$', 'NeFT${_{3\%}}$+Reversed${_{7\%}}$']
    strongly_neuron = [2514, 1611, 745, 413]
    suppressed_neuron = [1069, 615, 244, 146]
    indirectly_neuron = [245, 112, 13, 14]

    x = np.arange(len(countries))
    width = 0.2
    gold_x = x
    silver_x = x + width
    bronze_x = x + 2 * width

    ax4.bar(gold_x, strongly_neuron, width=width, color="gold", label='Strongly')
    ax4.bar(silver_x, suppressed_neuron, width=width, color="silver", label='Suppressed')
    ax4.bar(bronze_x, indirectly_neuron, width=width, color="saddlebrown", label='Indirectly')

    ax4.set_xticks(x + width)
    ax4.set_xticklabels(labels=countries, fontsize=15, rotation=15)
    ax4.tick_params('y', direction='in', labelsize=15)
    ax4.minorticks_off()

    for i in range(len(countries)):
        ax4.text(gold_x[i], strongly_neuron[i], strongly_neuron[i], va="bottom", ha="center", fontsize=8)
        ax4.text(silver_x[i], suppressed_neuron[i], suppressed_neuron[i], va="bottom", ha="center", fontsize=8)
        ax4.text(bronze_x[i], indirectly_neuron[i], indirectly_neuron[i], va="bottom", ha="center", fontsize=8)
    ax4.spines['right'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    ax4.set_title('${\Delta\mathbf{Rank}}$ with NeFT${_{3\%}}$+Reversed${_{9\%}}$', size=15)

    # fig_path = 'different_type_neuron_1.pdf'
    # plt.savefig(fig_path)
    plt.tight_layout()
    fig.legend(handles=[strongly, suppressed, indirectly], labels=['Strongly', 'Suppressed', 'Indirectly'], loc='right')

    fig_path = 'different_type_neuron_all.pdf'
    plt.savefig(fig_path)
    plt.show()


def run_14():
    plt.rcParams['figure.figsize'] = [6, 6.5]
    plt.rc('font', family='Times New Roman')
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    fig = plt.figure(facecolor='white')
    ax1 = fig.add_subplot(3, 1, 1, facecolor='white')
    ax2 = fig.add_subplot(3, 1, 2, facecolor='white')
    # ax3 = fig.add_subplot(4, 1, 3, facecolor='white')
    ax4 = fig.add_subplot(3, 1, 3, facecolor='white')

    # NeFT_x% with NeFT_3%
    countries = ['NeFT${_{6\%}}$', 'NeFT${_{7\%}}$', 'NeFT${_{9\%}}$', 'NeFT${_{10\%}}$', 'NeFT${_{12\%}}$']
    strongly_neuron = [2546, 4294, 8947, 18716, 17129]
    suppressed_neuron = [1221, 2125, 4459, 10292, 8710]
    indirectly_neuron = [1550, 2794, 5990, 13465, 11234]

    strongly_neuron_2 = [10396, 6930, 6452, 6473, 0]
    suppressed_neuron_2 = [5670, 3971, 3306, 2797, 0]
    indirectly_neuron_2 = [6680, 4309, 4276, 4577, 0]

    x = np.arange(len(countries)) * 50
    width = 2
    gold_x = x - width
    silver_x = x
    bronze_x = x + width
    gold_x_2 = x + 2 * width
    silver_x_2 = x + 3 * width
    bronze_x_2 = x + 4 * width
    # = x + 5 * width

    ax1.bar(gold_x, strongly_neuron, width=width, color="gold", label='Strongly')
    ax1.bar(silver_x, suppressed_neuron, width=width, color="silver", label='Suppressed')
    ax1.bar(bronze_x, indirectly_neuron, width=width, color="saddlebrown", label='Indirectly')
    strongly_2 = ax1.bar(gold_x_2, strongly_neuron_2, width=width, color="gold", label='Strongly', hatch='////')
    suppressed_2 = ax1.bar(silver_x_2, suppressed_neuron_2, width=width, color="silver", label='Suppressed',
                           hatch='////')
    indirectly_2 = ax1.bar(bronze_x_2, indirectly_neuron_2, width=width, color="saddlebrown", label='Indirectly',
                           hatch='////')

    ax1.set_xticks(x + width)
    ax1.set_xticklabels(labels=countries, fontsize=15)
    ax1.tick_params('y', direction='in', labelsize=15)
    ax1.minorticks_off()

    for i in range(len(countries)):
        if i == 4:
            ax1.text(gold_x[i], strongly_neuron[i], strongly_neuron[i], va="bottom", ha="center", fontsize=8)
            ax1.text(silver_x[i], suppressed_neuron[i], suppressed_neuron[i], va="bottom", ha="center", fontsize=8)
            ax1.text(bronze_x[i], indirectly_neuron[i], indirectly_neuron[i], va="bottom", ha="center", fontsize=8)
        else:
            ax1.text(gold_x[i], strongly_neuron[i] - 300, strongly_neuron[i], va="bottom", ha="center", fontsize=8)
            ax1.text(silver_x[i], suppressed_neuron[i] - 800, suppressed_neuron[i], va="bottom", ha="center",
                     fontsize=8)
            ax1.text(bronze_x[i], indirectly_neuron[i] + 600, indirectly_neuron[i], va="bottom", ha="center",
                     fontsize=8)

            ax1.text(gold_x_2[i], strongly_neuron_2[i] + 1800, strongly_neuron_2[i], va="bottom", ha="center",
                     fontsize=8)
            ax1.text(silver_x_2[i], suppressed_neuron_2[i] - 200, suppressed_neuron_2[i], va="bottom", ha="center",
                     fontsize=8)
            ax1.text(bronze_x_2[i], indirectly_neuron_2[i] + 1200, indirectly_neuron_2[i], va="bottom", ha="center",
                     fontsize=8)

    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    ax1.set_title('(a) NeFT${_{x\%}}$ $\Delta$Rank with NeFT${_{3\%}}$ and NeFT${_{12\%}}$', size=15)

    # # NeFT_3%+Reversed_x% with NeFT_3%
    countries = ['NeFT${_{3\%}}$+Reversed${_{3\%}}$', 'NeFT${_{3\%}}$+Reversed${_{4\%}}$',
                 'NeFT${_{3\%}}$+Reversed${_{6\%}}$', 'NeFT${_{3\%}}$+Reversed${_{7\%}}$',
                 'NeFT${_{3\%}}$+Reversed${_{9\%}}$']
    strongly_neuron = [54572, 52236, 59262, 57671, 65961]
    suppressed_neuron = [26298, 25456, 28980, 27591, 31686]
    indirectly_neuron = [44439, 39911, 42855, 39488, 42619]

    x = np.arange(len(countries))
    width = 0.29
    gold_x = x
    silver_x = x + width
    bronze_x = x + 2 * width

    ax2.bar(gold_x, strongly_neuron, width=width, color="gold", label='Strongly')
    ax2.bar(silver_x, suppressed_neuron, width=width, color="silver", label='Suppressed')
    ax2.bar(bronze_x, indirectly_neuron, width=width, color="saddlebrown", label='Indirectly')

    ax2.set_xticks(x + width)
    ax2.set_xticklabels(labels=countries, fontsize=15, rotation=15)
    ax2.tick_params('y', direction='in', labelsize=15)
    # plt.ylim((20000, 68000))
    ax2.tick_params('y', direction='in', labelsize=15)
    ax2.minorticks_off()

    for i in range(len(countries)):
        ax2.text(gold_x[i], strongly_neuron[i], strongly_neuron[i], va="bottom", ha="center", fontsize=8)
        ax2.text(silver_x[i], suppressed_neuron[i], suppressed_neuron[i], va="bottom", ha="center", fontsize=8)
        ax2.text(bronze_x[i], indirectly_neuron[i], indirectly_neuron[i], va="bottom", ha="center", fontsize=8)

    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_title('(b) NeFT${_{3\%}}$+Reversed${_{x\%}}$ $\Delta$Rank with NeFT${_{3\%}}$', size=15)

    # # NeFT_12% with NeFT_x%
    # countries = ['NeFT${_{6\%}}$', 'NeFT${_{7\%}}$', 'NeFT${_{9\%}}$', 'NeFT${_{10\%}}$']
    # strongly_neuron = [10396, 6930, 6452, 6473]
    # suppressed_neuron = [5670, 3971, 3306, 2797]
    # indirectly_neuron = [6680, 4309, 4276, 4577]
    #
    # # NeFT_3%+Reversed_9% with NeFT_3%+Reversed_x%
    # x = np.arange(len(countries))
    # width = 0.2
    # gold_x = x
    # silver_x = x + width
    # bronze_x = x + 2 * width
    #
    # strongly = ax3.bar(gold_x, strongly_neuron, width=width, color="gold", label='Strongly')
    # suppressed = ax3.bar(silver_x, suppressed_neuron, width=width, color="silver", label='Suppressed')
    # indirectly = ax3.bar(bronze_x, indirectly_neuron, width=width, color="saddlebrown", label='Indirectly')
    #
    # ax3.set_xticks(x + width)
    # ax3.set_xticklabels(labels=countries, fontsize=15)
    # ax3.tick_params('y', direction='in', labelsize=15)
    # ax3.minorticks_off()
    #
    # for i in range(len(countries)):
    #     ax3.text(gold_x[i], strongly_neuron[i], strongly_neuron[i], va="bottom", ha="center", fontsize=8)
    #     ax3.text(silver_x[i], suppressed_neuron[i], suppressed_neuron[i], va="bottom", ha="center", fontsize=8)
    #     ax3.text(bronze_x[i], indirectly_neuron[i], indirectly_neuron[i], va="bottom", ha="center", fontsize=8)
    #
    # ax3.spines['right'].set_visible(False)
    # ax3.spines['top'].set_visible(False)
    # ax3.set_title('${\Delta\mathbf{Rank}}$ with NeFT${_{12\%}}$', size=15)

    # NeFT_3%+Reversed_9% with NeFT_3%+Reversed_x%
    countries = ['NeFT${_{3\%}}$+Reversed${_{3\%}}$', 'NeFT${_{3\%}}$+Reversed${_{4\%}}$',
                 'NeFT${_{3\%}}$+Reversed${_{6\%}}$', 'NeFT${_{3\%}}$+Reversed${_{7\%}}$']
    strongly_neuron = [2514, 1611, 745, 413]
    suppressed_neuron = [1069, 615, 244, 146]
    indirectly_neuron = [245, 112, 13, 14]

    x = np.arange(len(countries))
    width = 0.2
    gold_x = x
    silver_x = x + width
    bronze_x = x + 2 * width

    strongly = ax4.bar(gold_x, strongly_neuron, width=width, color="gold", label='Strongly')
    suppressed = ax4.bar(silver_x, suppressed_neuron, width=width, color="silver", label='Suppressed')
    indirectly = ax4.bar(bronze_x, indirectly_neuron, width=width, color="saddlebrown", label='Indirectly')

    ax4.set_xticks(x + width)
    ax4.set_xticklabels(labels=countries, fontsize=15, rotation=15)
    ax4.tick_params('y', direction='in', labelsize=15)
    ax4.minorticks_off()

    for i in range(len(countries)):
        ax4.text(gold_x[i], strongly_neuron[i], strongly_neuron[i], va="bottom", ha="center", fontsize=8)
        ax4.text(silver_x[i], suppressed_neuron[i], suppressed_neuron[i], va="bottom", ha="center", fontsize=8)
        ax4.text(bronze_x[i], indirectly_neuron[i], indirectly_neuron[i], va="bottom", ha="center", fontsize=8)
    ax4.spines['right'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    ax4.set_title('(c) NeFT${_{3\%}}$+Reversed${_{x\%}}$ $\Delta$Rank with NeFT${_{3\%}}$+Reversed${_{9\%}}$', size=15)

    # fig_path = 'different_type_neuron_1.pdf'
    # plt.savefig(fig_path)
    # plt.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.05)
    plt.tight_layout()
    ax1.grid(None)
    ax2.grid(None)
    ax4.grid(None)
    # ax1.legend(handles=[strongly, suppressed, indirectly, strongly_2, suppressed_2, indirectly_2],
    #            labels=['Strongly', 'Suppressed', 'Indirectly',
    #                    'Strongly and\n$\Delta$Rank with NeFT${_{12\%}}$',
    #                    'Suppressed and\n$\Delta$Rank with NeFT${_{12\%}}$',
    #                    'Indirectly and\n$\Delta$Rank with NeFT${_{12\%}}$'], loc='upper center', ncol=3)

    labelss = ax1.legend(handles=[strongly, strongly_2, suppressed, suppressed_2, indirectly, indirectly_2],
                         labels=['Strongly', 'Strongly and\n$\Delta$Rank with NeFT${_{12\%}}$', 'Suppressed',
                                 'Suppressed and\n$\Delta$Rank with NeFT${_{12\%}}$', 'Indirectly',
                                 'Indirectly and\n$\Delta$Rank with NeFT${_{12\%}}$'], ncol=3,
                         bbox_to_anchor=(1, -0.33))

    # labelss.set_fontproperties('Times New Roman')

    fig_path = 'different_type_neuron_all_2.pdf'
    plt.savefig(fig_path)
    plt.show()


def run_15():
    plt.rcParams['figure.figsize'] = [5, 5]
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    x = ['3%', '4%', '6%', '7%', '9%']
    # y1 = [28.67, 28.71, 28.70, 28.83, 28.40]
    # y2 = [27.59, 27.59, 27.38, 27.22, 27.00]

    # y1_bleu = [28.67, 28.70, 28.69, 28.82, 28.39]
    # y2_bleu = [27.58, 27.59, 27.37, 27.22, 26.99]
    y1 = [81.92, 81.97, 82.27, 82.03, 82.10]
    y2 = [81.47, 81.47, 81.24, 81.07, 80.82]

    fig, ax = plt.subplots()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.rc('font', family='Times New Roman')

    plt.xlabel('Additional Selection Percentage (%)', fontdict={'fontsize': 15, 'family': 'Times New Roman'})  # x轴标题
    plt.ylabel('COMET', fontdict={'fontsize': 15, 'family': 'Times New Roman'})  # y轴标题
    ax.plot(x, y1, marker='o', markersize=3)  # 绘制折线图，添加数据点，设置点的大小
    ax.plot(x, y2, marker='o', markersize=3)

    for a, b in zip(x, y1):
        if b == 81.92:
            ax.text(a, b, b, ha='left', va='bottom', fontsize=15)
        else:
            ax.text(a, b, b, ha='center', va='bottom', fontsize=15)
    flag = 1
    for a, b in zip(x, y2):
        if flag == 1:
            flag = 0
            ax.text(a, b, b, ha='left', va='bottom', fontsize=15)
        else:
            ax.text(a, b, b, ha='center', va='bottom', fontsize=15)

    plt.legend(['NeFT${_{3+x\%}}$', 'NeFT${_{3\%}}$+Reversed${_{x\%}}$'], fontsize=15, loc='center right')  # 设置折线名称
    # plt.grid(None)

    ax.tick_params('x', labelsize=15, direction='out')
    ax.tick_params('y', labelsize=15, direction='in')

    ax.minorticks_off()
    plt.yticks(fontproperties='Times New Roman', size=15)
    plt.xticks(fontproperties='Times New Roman', size=15)
    plt.tight_layout()

    plt.savefig('adding neurons.pdf')
    plt.show()  # 显示折线图


if __name__ == '__main__':
    # neuron_heatmap

    # neuron_heatmap(
    #     path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/HS_with_fullparam_after_gen/correlaton_W.enzh_sft_layer4.checkpoint-2500_HS.train_enzh_org.2500steps.after_gen_cos.csv',
    #     vmax=1, threshold=0.4)

    # neuron_heatmap(
    #     path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/HS_with_fullparam_after_gen/correlaton_W.enzh_sft_layer0_deepspeed.checkpoint-2500_HS.train_enzh_org.2500steps.after_gen_cos.csv',
    #     vmax=1, threshold=0.4)
    #
    # neuron_heatmap(
    #     path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/HS_with_fullparam_after_gen/correlaton_W.enzh_sft_layer1_deepspeed.checkpoint-2500_HS.train_enzh_org.2500steps.after_gen_cos.csv',
    #     vmax=1, threshold=0.4)
    #
    # neuron_heatmap(
    #     path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/HS_with_fullparam_after_gen/correlaton_W.enzh_sft_layer2_deepspeed.checkpoint-2500_HS.train_enzh_org.2500steps.after_gen_cos.csv',
    #     vmax=1, threshold=0.4)
    #
    # neuron_heatmap(
    #     path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/HS_with_fullparam_after_gen/correlaton_W.enzh_sft_layer3_deepspeed.checkpoint-2500_HS.train_enzh_org.2500steps.after_gen_cos.csv',
    #     vmax=1, threshold=0.4)

    # neuron_heatmap(
    #     path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/correlaton_W.enzh_sft_layer3_deepspeed.checkpoint-2500_HS.enzh_sft_layer3_deepspeed.2500steps.after_gen_cos.csv',
    #     vmax=1, threshold=0.4)

    # neuron_heatmap(
    #     path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/correlaton_W.enzh_sft_layer4.checkpoint-2500_HS.enzh_sft_layer4_deepspeed.2500steps.after_gen_cos.csv',
    #     vmax=1, threshold=0.4)

    # neuron_heatmap(
    #     path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/correlaton_in.train_enzh_org.checkpoint-2500_HS.train_enzh_org.2500steps.before_gen_cos.csv',
    #     vmax=0.4)
    #
    # neuron_heatmap(
    #     path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/correlaton_W.enzh_sft_layer8_deepspeed.checkpoint-2500_HS.enzh_sft_layer8_deepspeed.2500steps.after_gen_cos.csv',
    #     vmax=0.4)

    # neuron_heatmap(
    #     path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/W_HS_self/correlaton_W.train_enzh_org.checkpoint-2500_HS.train_enzh_org.2500steps.after_gen_cos.csv',
    #     vmax=0.4)

    # neuron_heatmap_cos()

    # neuron_overlap()

    # neuron_overlap(
    #     path_1=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/W_HS_self/fullparam_after_gen_neuron_dict.json',
    #     path_2=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/org_allsft/2500steps/neuron_dict.json')

    # neuron_overlap(
    #     path_1=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_enzh_org/1600step/50000/neuron_dict_0.99904525.json',
    #     path_2=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_enzh_org/3200step/50000/neuron_dict_0.99836355.json')

    # neuron_overlap(
    #     path_1=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_enzh_org/1600step/100000/neuron_dict_0.9991139.json',
    #     path_2=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_enzh_org/3200step/100000/neuron_dict_0.99847424.json')

    # show suppressed neuron

    # show_suppressed_neuron()

    # show_suppressed_neuron(
    #     path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/enzh_org-enzh_sn_3200/abs_diff.csv')

    # neuron_overlap(
    #     path_1=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/rank_diff/cos-100000_cos-5w/abs_rank_diff_larger_10W_fall.json',
    #     path_2=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons/final/W_HS/pearsonr/mt/enzh/rank_diff/cos-100000_cos-5w/abs_rank_diff_larger_10W_fall.json')

    # path_list = [
    #     '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_enzh_org/3200step/other/50000+Reversed_100000/neuron_dict.json',
    #     '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_enzh_org/3200step/other/50000+Reversed_100000/neuron_dict.json',
    # ]
    # neuron_overlap_bar(path_list, pic_name='NeFT${_{3\%}}$+Reversed-100000: Distribution of neurons')

    # abs_rank_diff_mean_bar_chart(pic_name='max_sorted_add_rank_diff')
    # abs_rank_diff_layer_number_bar_chart_series(pic_name='layer_number_in_every_percentage_range')

    # run_1()
    path_list = [
        '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_hizh_org/750step/150000/neuron_dict_0.99964154.json',
        '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/find_neuron/results/neurons_sim/final/train_hizh_org/750step/150000/neuron_dict_0.99964154.json']

    neuron_overlap_bar(path_list,
                       pic_name='Distribution of the number of neurons across layers for NeFT${_{9\%}}$ on the Hindi-Chinese data')

    # run_2()
    # run_3()
    # run_4()
    # run_5()
    # run_6()
    # run_7()

    # run_6_and_7()

    # run_7_old()

    # run_8()

    # run_9()
    # run_10()
    # run_11()
    # run_12()
    # run_13()

    # run_14()

    # run_15()
