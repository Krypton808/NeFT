import os

import jsonlines

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import torch
import transformers
# from transformer_lens import HookedTransformer
from transformers import LlamaTokenizer, LlamaForCausalLM


def W_in_and_W_out_check():
    path = r'/mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf'
    # path = r'/mnt/nfs/algo/intern/haoyunx11/models/sft/xnli/train_neuron/neuron_4_328_in_out/checkpoint-500'
    tokenizer = LlamaTokenizer.from_pretrained(path)
    hf_model = LlamaForCausalLM.from_pretrained(path)
    # model = HookedTransformer.from_pretrained(r'Llama-2-7b-chat-hf', hf_model=hf_model, device="cpu",
    #                                           fold_ln=False,
    #                                           center_writing_weights=False, center_unembed=True,
    #                                           tokenizer=tokenizer)
    # print(model.W_out.shape)
    # print(model.W_out[0])
    # print(model.W_out[0].shape)
    #
    # print('*****************************************')

    for k, v in hf_model.named_parameters():
        if 'layers.4.mlp' in k and 'up_proj' in k:
            print(k)
            print(v)
            print(v.shape)
            print(v[4][300: 400])


def W_in_and_W_out_check_two_model_compare():
    path_1 = r'/mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf'
    path_2 = r'/mnt/nfs/algo/intern/haoyunx11/models/sft/xnli/train_neuron/neuron_4_328_in_out/checkpoint-500'
    hf_model_1 = LlamaForCausalLM.from_pretrained(path_1)
    hf_model_2 = LlamaForCausalLM.from_pretrained(path_2)

    for v in zip(hf_model_1.named_parameters(), hf_model_2.named_parameters()):
        # if v[0][0] != v[1][0] or v[0][1] != v[1][1]:
        #     print(v[0][0])
        #     print(v[1][0])
        #     print(v[0][1])
        #     print(v[1][1])
        # print('*' * 50)

        if v[0][0] != v[1][0]:
            print('k')
            print(print(v[0][0]))
            print(v[1][0])
            break

        if not v[0][1].equal(v[1][1]):
            print('v')
            print(v[0][0])
            print(v[1][0])

            print(v[0][1].shape)

            eq_list = v[0][1].eq(v[1][1])
            print(eq_list)
            for idx, eq in enumerate(eq_list):
                if False in eq:
                    print(idx)
                    for idx_, eq_ in enumerate(eq):
                        if not eq_:
                            print(idx_)

            print('*' * 50)


def test1():
    print(torch.cuda.is_available())
    print(torch.__version__)
    print('*****')


def sn_dict_test():
    sn_dict = {}
    sn_dict['4'] = (328, ('up_proj', 'down_proj'))
    sn_dict['1'] = (20, ('up_proj', 'down_proj'))

    print(sn_dict.keys())
    print(sn_dict['4'])
    print(sn_dict['4'][0])
    print(sn_dict['4'][1][1])


# add in out
def list_to_sn_dict():
    sn_list = list(
        ['1_20_out', '1_1331_out', '1_1936_in', '1_2069_in', '1_2069_out', '1_4600_in', '1_9015_out', '1_9350_in',
         '1_9805_out', '1_10397_out', '2_1325_in', '2_1325_out', '2_3699_in', '2_5201_out', '2_8637_in', '2_10040_in',
         '3_1370_in', '3_1370_out', '3_3687_out', '3_8948_out', '3_9352_in', '4_328_in', '4_328_out', '4_1522_in',
         '4_4336_in', '4_9116_in', '6_4469_in', '2_10695_out', '1_7890_in'])

    sn_dict = {}
    for sn_line in sn_list:
        sn_x, sn_y, sn_z = sn_line.split('_')

        if '.' + sn_x + '.' not in sn_dict.keys():
            sn_dict['.' + sn_x + '.'] = [sn_y]
        else:
            sn_dict['.' + sn_x + '.'].append(sn_y)

    for k, v in sn_dict.items():
        sn_dict[k] = list(set(v))

    print(sn_dict)

    return sn_dict


def test2():
    list = [1, 2, 3, 4, 5, 6]

    print(list[1:5 + 1])

def test3():
    lines = jsonlines.open(
        r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/data/CrossSum/english-chinese_simplified/english-chinese_simplified_test.jsonl')
    writer = jsonlines.open(
        r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/data/CrossSum/english-chinese_simplified/4096_cut_english-chinese_simplified_test.jsonl',
        'w')

    max_len = 0
    for idx, line in enumerate(lines):

        model_name_or_path = '/mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf'
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)

        prompt = "Given the below article, generate a summary in English." + "\nArticle: "
        input_s = prompt + line['text']
        output_s = line['summary']

        tokenized = tokenizer([input_s, output_s])
        input_ids = tokenized['input_ids']

        length = len(input_ids[0])

        if max_len < length:
            max_len = length

        if length >= 4096:
            print(line['text'])
            print(length)
            continue

        writer.write(line)

    print(max_len)

def test4():
    import csv
    with open(r'/data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/sn/greedy_decode/ende/train_ende_org/1500steps/ende_output.csv', 'r', encoding='utf-8') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            print(row[0])

def test5():
    print(300000 / 8)

def test6():
    import evaluate
    metric = evaluate.load("/mnt/nfs/algo/intern/haoyunx9/evaluate_metric/accuracy")


if __name__ == '__main__':
    # W_in_and_W_out_check()
    # W_in_and_W_out_check_two_model_compare()
    # test1()
    # sn_dict_test()
    # list_to_sn_dict()
    # test2()
    # test3()
    # test4()
    test6()

