import pandas as pd

# pd.set_option('display.max_columns', None)
from pandas.core.computation.ops import _in

pd.set_option('display.max_rows', None)


def check_overlap_top_neurons(
        random_path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/en/xnli.en_random_prompt.csv',
        simple_path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/en/xnli.en_simple_prompt.csv',
        standard_path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/en/xnli.en_standard_prompt.csv'):
    random = pd.read_csv(random_path)
    simple = pd.read_csv(simple_path)
    standard = pd.read_csv(standard_path)

    in_layer_overlap = []
    # for layer in simple['neuron'].where(simple['neuron_weight'] == 'in'):
    #     if layer in random['neuron'].where(random['neuron_weight'] == 'in') and layer in standard[
    #         'neuron'].where(standard['neuron_weight'] == 'in'):
    #         in_layer_overlap.append(int(layer))

    for layer in set(simple[simple.neuron_weight == 'in'].neuron):
        if layer in set(random[random.neuron_weight == 'in'].neuron) and layer in set(
                standard[standard.neuron_weight == 'in'].neuron):
            in_layer_overlap.append(int(layer))

    in_layer_overlap = list(set(in_layer_overlap))
    print(in_layer_overlap)

    out_layer_overlap = []
    # for layer in simple['neuron'].where(simple['neuron_weight'] == 'out'):
    #     if layer in random['neuron'].where(random['neuron_weight'] == 'out') and layer in standard[
    #         'neuron'].where(standard['neuron_weight'] == 'out'):
    #         out_layer_overlap.append(int(layer))

    for layer in set(simple[simple.neuron_weight == 'out'].neuron):
        if layer in set(random[random.neuron_weight == 'out'].neuron) and layer in set(
                standard[standard.neuron_weight == 'out'].neuron):
            out_layer_overlap.append(int(layer))

    out_layer_overlap = list(set(out_layer_overlap))

    thr = 0.2

    print(random[random.neuron.isin(in_layer_overlap) & (random.abs_corr >= thr) & (random.neuron_weight == 'in')])

    print(simple[simple.neuron.isin(in_layer_overlap) & (simple.abs_corr >= thr) & (simple.neuron_weight == 'in')])

    print(standard[
              standard.neuron.isin(in_layer_overlap) & (standard.abs_corr >= thr) & (standard.neuron_weight == 'in')])

    print('-' * 50)
    print(out_layer_overlap)

    print(random[random.neuron.isin(out_layer_overlap) & (random.abs_corr >= thr) & (random.neuron_weight == 'out')])

    print(simple[simple.neuron.isin(out_layer_overlap) & (simple.abs_corr >= thr) & (simple.neuron_weight == 'out')])

    print(standard[
              standard.neuron.isin(out_layer_overlap) & (standard.abs_corr >= thr) & (standard.neuron_weight == 'out')])


def check_overlap_top_neurons_4(
        path_1=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/en/xnli.en_simple_prompt.csv',
        path_2=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/de/xnli.de_simple_prompt.csv',
        path_3=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/zh/xnli.zh_simple_prompt.csv',
        path_4=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/ru/xnli.ru_simple_prompt.csv'):
    pd_1 = pd.read_csv(path_1)
    pd_2 = pd.read_csv(path_2)
    pd_3 = pd.read_csv(path_3)
    pd_4 = pd.read_csv(path_4)

    in_layer_overlap = []
    for layer in set(pd_1[pd_1.neuron_weight == 'in'].neuron):
        if layer in set(pd_2[pd_2.neuron_weight == 'in'].neuron) and layer in set(
                pd_3[pd_3.neuron_weight == 'in'].neuron) and layer in set(
            pd_4[pd_4.neuron_weight == 'in'].neuron):
            in_layer_overlap.append(int(layer))

    out_layer_overlap = []
    for layer in set(pd_1[pd_1.neuron_weight == 'out'].neuron):
        if layer in set(pd_2[pd_2.neuron_weight == 'out'].neuron) and layer in set(
                pd_3[pd_3.neuron_weight == 'out'].neuron) and layer in set(
            pd_4[pd_4.neuron_weight == 'out'].neuron):
            out_layer_overlap.append(int(layer))

    in_layer_overlap = list(set(in_layer_overlap))

    thr = 0.1

    print(in_layer_overlap)

    print(pd_1[pd_1.neuron.isin(in_layer_overlap) & (pd_1.abs_corr >= thr) & (pd_1.neuron_weight == 'in')])

    print(pd_2[pd_2.neuron.isin(in_layer_overlap) & (pd_2.abs_corr >= thr) & (pd_2.neuron_weight == 'in')])

    print(pd_3[pd_3.neuron.isin(in_layer_overlap) & (pd_3.abs_corr >= thr) & (pd_3.neuron_weight == 'in')])

    print(pd_4[pd_4.neuron.isin(in_layer_overlap) & (pd_4.abs_corr >= thr) & (pd_4.neuron_weight == 'in')])

    print('-' * 50)

    out_layer_overlap = list(set(out_layer_overlap))
    print(out_layer_overlap)

    print(pd_1[pd_1.neuron.isin(out_layer_overlap) & (pd_1.abs_corr >= thr) & (pd_1.neuron_weight == 'out')])

    print(pd_2[pd_2.neuron.isin(out_layer_overlap) & (pd_2.abs_corr >= thr) & (pd_2.neuron_weight == 'out')])

    print(pd_3[pd_3.neuron.isin(out_layer_overlap) & (pd_3.abs_corr >= thr) & (pd_3.neuron_weight == 'out')])

    print(pd_4[pd_4.neuron.isin(out_layer_overlap) & (pd_4.abs_corr >= thr) & (pd_4.neuron_weight == 'out')])


def check_overlap_top_neurons_two(
        path_1=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/en/xnli.en_standard_prompt.csv',
        path_2=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/zh/xnli.zh_standard_prompt.csv'):
    pd_1 = pd.read_csv(path_1)
    pd_2 = pd.read_csv(path_2)

    # in_layer_overlap = []
    # for layer in pd_1['neuron'].where(pd_1['neuron_weight'] == 'in'):
    #     if layer in pd_2['neuron'].where(pd_2['neuron_weight'] == 'in'):
    #         in_layer_overlap.append(int(layer))
    #
    # in_layer_overlap = list(set(in_layer_overlap))
    # print(in_layer_overlap)

    # out_layer_overlap = []
    # for layer in pd_1['neuron'].where(pd_1['neuron_weight'] == 'out'):
    #     if layer in pd_2['neuron'].where(pd_2['neuron_weight'] == 'out'):
    #         out_layer_overlap.append(int(layer))
    #
    # out_layer_overlap = list(set(out_layer_overlap))

    in_layer_overlap = []
    for layer in set(pd_1[pd_1.neuron_weight == 'in'].neuron):
        if layer in set(pd_2[pd_2.neuron_weight == 'in'].neuron):
            in_layer_overlap.append(int(layer))

    out_layer_overlap = []
    for layer in set(pd_1[pd_1.neuron_weight == 'out'].neuron):
        if layer in set(pd_2[pd_2.neuron_weight == 'out'].neuron):
            out_layer_overlap.append(int(layer))

    in_layer_overlap = list(set(in_layer_overlap))

    thr = 0.2

    print(in_layer_overlap)

    print(pd_1[pd_1.neuron.isin(in_layer_overlap) & (pd_1.abs_corr >= thr) & (pd_1.neuron_weight == 'in')])

    print(pd_2[pd_2.neuron.isin(in_layer_overlap) & (pd_2.abs_corr >= thr) & (pd_2.neuron_weight == 'in')])

    print('-' * 50)
    out_layer_overlap = list(set(out_layer_overlap))
    print(out_layer_overlap)

    print(pd_1[pd_1.neuron.isin(out_layer_overlap) & (pd_1.abs_corr >= thr) & (pd_1.neuron_weight == 'out')])

    print(pd_2[pd_2.neuron.isin(out_layer_overlap) & (pd_2.abs_corr >= thr) & (pd_2.neuron_weight == 'out')])


def check_overlap_top_neurons_all(
        path_1=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/en/xnli.en_simple_prompt.csv',
        path_2=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/de/xnli.de_simple_prompt.csv',
        path_3=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/zh/xnli.zh_simple_prompt.csv',
        path_4=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/ru/xnli.ru_simple_prompt.csv',
        path_5=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/bg/xnli.bg_simple_prompt.csv',
        path_6=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/es/xnli.es_simple_prompt.csv',
        path_7=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/fr/xnli.fr_simple_prompt.csv'):
    pd_1 = pd.read_csv(path_1)
    pd_2 = pd.read_csv(path_2)
    pd_3 = pd.read_csv(path_3)
    pd_4 = pd.read_csv(path_4)
    pd_5 = pd.read_csv(path_5)
    pd_6 = pd.read_csv(path_6)
    pd_7 = pd.read_csv(path_7)

    # in_layer_overlap = []
    # for layer in pd_1['neuron'].where(pd_1['neuron_weight'] == 'in'):
    #     if layer in pd_2['neuron'].where(pd_2['neuron_weight'] == 'in'):
    #         in_layer_overlap.append(int(layer))
    #
    # in_layer_overlap = list(set(in_layer_overlap))
    # print(in_layer_overlap)

    # out_layer_overlap = []
    # for layer in pd_1['neuron'].where(pd_1['neuron_weight'] == 'out'):
    #     if layer in pd_2['neuron'].where(pd_2['neuron_weight'] == 'out'):
    #         out_layer_overlap.append(int(layer))
    #
    # out_layer_overlap = list(set(out_layer_overlap))

    in_layer_overlap = []
    for layer in set(pd_1[pd_1.neuron_weight == 'in'].neuron):
        if layer in set(pd_2[pd_2.neuron_weight == 'in'].neuron) and layer in set(
                pd_3[pd_3.neuron_weight == 'in'].neuron) and layer in set(
            pd_4[pd_4.neuron_weight == 'in'].neuron) and layer in set(
            pd_5[pd_5.neuron_weight == 'in'].neuron) and layer in set(
            pd_6[pd_6.neuron_weight == 'in'].neuron) and layer in set(pd_7[pd_7.neuron_weight == 'in'].neuron):
            in_layer_overlap.append(int(layer))

    out_layer_overlap = []
    for layer in set(pd_1[pd_1.neuron_weight == 'out'].neuron):
        if layer in set(pd_2[pd_2.neuron_weight == 'out'].neuron) and layer in set(
                pd_3[pd_3.neuron_weight == 'out'].neuron) and layer in set(
            pd_4[pd_4.neuron_weight == 'out'].neuron) and layer in set(
            pd_5[pd_5.neuron_weight == 'out'].neuron) and layer in set(
            pd_6[pd_6.neuron_weight == 'out'].neuron) and layer in set(pd_7[pd_7.neuron_weight == 'out'].neuron):
            out_layer_overlap.append(int(layer))

    in_layer_overlap = list(set(in_layer_overlap))

    thr = 0.1

    print(in_layer_overlap)

    print(pd_1[pd_1.neuron.isin(in_layer_overlap) & (pd_1.abs_corr >= thr) & (pd_1.neuron_weight == 'in')])

    print(pd_2[pd_2.neuron.isin(in_layer_overlap) & (pd_2.abs_corr >= thr) & (pd_2.neuron_weight == 'in')])

    print(pd_3[pd_3.neuron.isin(in_layer_overlap) & (pd_3.abs_corr >= thr) & (pd_3.neuron_weight == 'in')])

    print(pd_4[pd_4.neuron.isin(in_layer_overlap) & (pd_4.abs_corr >= thr) & (pd_4.neuron_weight == 'in')])

    print(pd_5[pd_5.neuron.isin(in_layer_overlap) & (pd_5.abs_corr >= thr) & (pd_5.neuron_weight == 'in')])

    print(pd_6[pd_6.neuron.isin(in_layer_overlap) & (pd_6.abs_corr >= thr) & (pd_6.neuron_weight == 'in')])

    print(pd_7[pd_7.neuron.isin(in_layer_overlap) & (pd_7.abs_corr >= thr) & (pd_7.neuron_weight == 'in')])

    print('-' * 50)
    out_layer_overlap = list(set(out_layer_overlap))
    print(out_layer_overlap)

    print(pd_1[pd_1.neuron.isin(out_layer_overlap) & (pd_1.abs_corr >= thr) & (pd_1.neuron_weight == 'out')])

    print(pd_2[pd_2.neuron.isin(out_layer_overlap) & (pd_2.abs_corr >= thr) & (pd_2.neuron_weight == 'out')])

    print(pd_3[pd_3.neuron.isin(out_layer_overlap) & (pd_3.abs_corr >= thr) & (pd_3.neuron_weight == 'out')])

    print(pd_4[pd_4.neuron.isin(out_layer_overlap) & (pd_4.abs_corr >= thr) & (pd_4.neuron_weight == 'out')])

    print(pd_5[pd_5.neuron.isin(out_layer_overlap) & (pd_5.abs_corr >= thr) & (pd_5.neuron_weight == 'out')])

    print(pd_6[pd_6.neuron.isin(out_layer_overlap) & (pd_6.abs_corr >= thr) & (pd_6.neuron_weight == 'out')])

    print(pd_7[pd_7.neuron.isin(out_layer_overlap) & (pd_7.abs_corr >= thr) & (pd_7.neuron_weight == 'out')])


def get_overlap():
    list_1 = [4548, 9350, 3687, 9352, 328, 2505, 7784, 4600, 1936, 4336, 1522, 3699, 20, 2069, 4469, 10040, 409, 1370]
    list_2 = [647, 5138, 20, 2069, 10397, 8094, 6690, 2347, 1325, 1331, 9015, 8769, 8774, 328, 9805, 5201, 9561, 1370,
              3687, 8948, 511]
    list_3 = [9350, 9352, 1800, 4877, 2318, 1936, 7955, 7317, 2069, 5525, 8094, 1834, 5805, 1325, 10040, 6073, 8637,
              1220, 328, 9805, 7374, 1370, 4336, 753, 1522, 3699, 4469, 4600]
    list_4 = [2317, 8973, 20, 2069, 10397, 6947, 1827, 2347, 1325, 1331, 9015, 10695, 328, 9805, 7374, 5201, 9561, 1370,
              9436, 3687, 623, 4336, 8948, 10102]
    list_5 = [4548, 9350, 9352, 328, 7784, 4600, 4877, 1325, 1936, 3920, 4336, 1522, 3699, 7317, 2069, 4469, 10040,
              1370]
    list_6 = [647, 8973, 5138, 20, 2069, 10397, 8094, 2347, 1325, 1331, 9015, 8769, 328, 9805, 5201, 9561, 1370, 3687,
              4336, 8948, 10102]

    overlap_1 = list(set(list_1).intersection(set(list_5)))
    overlap_2 = list(set(list_2).intersection(set(list_6)))

    # overlap_1 = list(set(list_1).intersection(set(list_3)).intersection(set(list_5)))
    # overlap_2 = list(set(list_2).intersection(set(list_4)).intersection(set(list_6)))

    print(overlap_1)
    print(overlap_2)


def single_read(
        path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/neuron_4_328_in_out_progressive_500steps/en/neuron_4_328_in_out_progressive_500steps.en_standard_prompt.csv'):

    pd_1 = pd.read_csv(path)

    neuron_list = []

    for p in pd_1.iterrows():
        name = str(p[1]['neuron_layer']) + '_' + str(p[1]['neuron']) + '_' + str(p[1]['neuron_weight'])
        if name not in neuron_list:
            print(name)
            neuron_list.append(name)

    print(len(neuron_list))


if __name__ == '__main__':
    # check_overlap_top_neurons()
    # check_overlap_top_neurons(
    #     random_path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/en/xnli.en_random_prompt.csv',
    #     simple_path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/zh/xnli.zh_random_prompt.csv',
    #   standard_path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/de/xnli.de_random_prompt.csv')

    # check_overlap_top_neurons(
    #     random_path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/de/xnli.de_random_prompt.csv',
    #     simple_path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/de/xnli.de_simple_prompt.csv',
    #     standard_path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/de/xnli.de_standard_prompt.csv')

    # check_overlap_top_neurons_two(
    #     path_1=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/zh/xnli.zh_standard_prompt.csv',
    #     path_2=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/en/xnli.en_standard_prompt.csv')

    # check_overlap_top_neurons_two(
    #     path_1=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/en/xnli.en_simple_prompt.csv',
    #     path_2=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/de/xnli.de_simple_prompt.csv')

    # get_overlap()

    # ----------------------------------------------------
    # check_overlap_top_neurons_all()

    # check_overlap_top_neurons_two(
    #     path_1=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/en/xnli.en_simple_prompt.csv',
    #     path_2=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/de/xnli.de_simple_prompt.csv')

    # check_overlap_top_neurons_two(
    #     path_1=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/ru/xnli.ru_simple_prompt.csv',
    #     path_2=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/bg/xnli.bg_simple_prompt.csv')

    # check_overlap_top_neurons_two(
    # path_1=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/fr/xnli.fr_simple_prompt.csv',
    # path_2=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/es/xnli.es_simple_prompt.csv')

    # check_overlap_top_neurons_two(
    #     path_1=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/en/xnli.en_simple_prompt.csv',
    #     path_2=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/es/xnli.es_simple_prompt.csv')

    # check_overlap_top_neurons_two(
    #     path_1=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/zh/xnli.zh_simple_prompt.csv',
    #     path_2=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/es/xnli.es_simple_prompt.csv')

    # ----------------------------------------------------

    # SFT model
    # xnli_en_sft_all_layers_500steps
    # check_overlap_top_neurons(
    #     random_path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_all_layers_500steps/en/xnli_en_sft_all_layers_500steps.en_random_prompt.csv',
    #     simple_path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_all_layers_500steps/en/xnli_en_sft_all_layers_500steps.en_simple_prompt.csv',
    #     standard_path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_all_layers_500steps/en/xnli_en_sft_all_layers_500steps.en_standard_prompt.csv')

    # xnli_en_sft_layer4_2500steps
    # check_overlap_top_neurons(
    #     random_path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_layer4_2500steps/en/xnli_en_sft_layer4_2500steps.en_random_prompt.csv',
    #     simple_path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_layer4_2500steps/en/xnli_en_sft_layer4_2500steps.en_simple_prompt.csv',
    #     standard_path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_layer4_2500steps/en/xnli_en_sft_layer4_2500steps.en_standard_prompt.csv')

    # xnli_en_sft_layer30_2500steps
    # check_overlap_top_neurons(
    #     random_path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_layer30_2500steps/en/xnli_en_sft_layer30_2500steps.en_random_prompt.csv',
    #     simple_path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_layer30_2500steps/en/xnli_en_sft_layer30_2500steps.en_simple_prompt.csv',
    #     standard_path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_layer30_2500steps/en/xnli_en_sft_layer30_2500steps.en_standard_prompt.csv')

    # check_overlap_top_neurons_two(
    #     path_1=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_all_layers_500steps/en/xnli_en_sft_all_layers_500steps.en_standard_prompt.csv',
    #     path_2=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_layer30_2500steps/en/xnli_en_sft_layer30_2500steps.en_standard_prompt.csv')

    # check_overlap_top_neurons_two(
    #     path_1=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_all_layers_500steps/en/xnli_en_sft_all_layers_500steps.en_standard_prompt.csv',
    #     path_2=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_layer30_2500steps/en/xnli_en_sft_layer30_2500steps.en_standard_prompt.csv')

    # check_overlap_top_neurons_two(
    #     path_1=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_all_layers_500steps/en/xnli_en_sft_all_layers_500steps.en_standard_prompt.csv',
    #     path_2=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_layer30_2500steps/en/xnli_en_sft_layer30_2500steps.en_standard_prompt.csv')

    # check_overlap_top_neurons_two(
    #     path_1=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/en/xnli.en_standard_prompt.csv',
    #     path_2=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_all_layers_500steps/en/xnli_en_sft_all_layers_500steps.en_standard_prompt.csv')

    # 3 model en
    # check_overlap_top_neurons(
    #     random_path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_all_layers_500steps/en/xnli_en_sft_all_layers_500steps.en_standard_prompt.csv',
    #     simple_path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_layer4_2500steps/en/xnli_en_sft_layer4_2500steps.en_standard_prompt.csv',
    #     standard_path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_layer30_2500steps/en/xnli_en_sft_layer30_2500steps.en_standard_prompt.csv')

    # check_overlap_top_neurons_4(
    #     path_1=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_all_layers_500steps/en/xnli_en_sft_all_layers_500steps.en_standard_prompt.csv',
    #     path_2=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_layer4_2500steps/en/xnli_en_sft_layer4_2500steps.en_standard_prompt.csv',
    #     path_3=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/xnli_en_sft_layer30_2500steps/en/xnli_en_sft_layer30_2500steps.en_standard_prompt.csv',
    #     path_4=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/en/xnli.en_standard_prompt.csv')

    # neuron_4_328_in_out_progressive_500steps
    # check_overlap_top_neurons(
    #     random_path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/neuron_4_328_in_out_progressive_2000steps/en/neuron_4_328_in_out_progressive_2000steps.en_random_prompt.csv',
    #     simple_path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/neuron_4_328_in_out_progressive_2000steps/en/neuron_4_328_in_out_progressive_2000steps.en_simple_prompt.csv',
    #     standard_path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/neuron_4_328_in_out_progressive_2000steps/en/neuron_4_328_in_out_progressive_2000steps.en_standard_prompt.csv')

    # check_overlap_top_neurons_two(
    #     path_1=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/Llama-2-7b-chat-hf/en/xnli.en_standard_prompt.csv',
    #     path_2=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/probes/results/top_neurons/neuron_4_328_in_out_progressive_500steps/en/neuron_4_328_in_out_progressive_500steps.en_standard_prompt.csv')

    single_read()

