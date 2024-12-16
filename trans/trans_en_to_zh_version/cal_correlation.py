import os

import nltk
import numpy
import pandas as pd
from tqdm import tqdm
from CKA import cka, gram_linear
import numpy as np
from transformers import AutoTokenizer, LlamaTokenizer, BertTokenizer


def example():
    # source_langs = ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']
    # target_langs = ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']

    source_langs = ['en', 'zh']
    target_langs = ['en', 'zh']

    results = []
    for i, target_lang in enumerate(target_langs):
        print(f"Starting {target_lang} ({i}/{len(target_langs)})")

        BASE_embeddings = np.load(f'./Output/Embeddings/base_{target_lang}.npy')

        for source_lang in tqdm(source_langs):

            FT_embeddings = np.load(f'./Output/Embeddings/{source_lang}_to_{target_lang}.npy')

            for layer in range(1, 13):
                X = BASE_embeddings[:, layer, :]
                Y = FT_embeddings[:, layer, :]
                similarity = cka(gram_linear(X), gram_linear(Y))

                results.append([source_lang, target_lang, layer, similarity])

    results = pd.DataFrame(results, columns=['Source', 'Target', 'Layer', 'CKA'])
    results.to_excel(
        '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/trans/trans_en_to_zh_version/output/correlation/CKA.xlsx',
        index=False)


def example_2():
    results = []

    # (5010, 13, 768)
    embedding_en = np.load(f'/data5/haoyun.xu/study/MI/MMMI/src/example/Ouput/Embeddings/base_en.npy')
    embedding_zh = np.load(f'/data5/haoyun.xu/study/MI/MMMI/src/example/Ouput/Embeddings/base_zh.npy')

    for layer in range(1, 13):
        X = embedding_en[0:4, layer, :]

        Y = embedding_zh[0:4, layer, :]
        print(X.shape)
        print(Y.shape)
        similarity = cka(gram_linear(X), gram_linear(Y))
        results.append(['en', 'zh', layer, similarity])

    results = pd.DataFrame(results, columns=['Source', 'Target', 'Layer', 'CKA'])
    results.to_excel(
        '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/trans/trans_en_to_zh_version/output/correlation/CKA_example.xlsx',
        index=False)


def find_match_index(alist, blist):
    a_idx = 0
    start = -1
    end = -1
    for idx, b in enumerate(blist):
        if a_idx > (len(alist) - 1):
            break
        elif a_idx == (len(alist) - 1):
            end = idx

        if b == alist[a_idx]:
            a_idx += 1
            if start == -1:
                start = idx

        else:
            a_idx = 0
            start = -1
            end = -1

    return start, end


# 测试第一条数据中，简单词和困难词各挑一个做测试
# On 有多个index

def cal_correlation_test():
    tokenizer = LlamaTokenizer.from_pretrained(r"/mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf")

    # 选词
    hard_word_1 = 'cells'
    hard_word_2 = 'diagnostic'

    simple_word_1 = 'On'
    simple_word_2 = 'Monday'

    # 根据对齐数据找到对应token
    aligned_path_enzh = r'/data5/haoyun.xu/study/alignment/awesome-align/output/trans/flores101_enzh/enzh_aligned_False.txt'
    org_path = r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/trans/trans_en_to_zh_version/output/align/ready/trans_enzh_7b_False.txt'
    output_str_path = r'/mnt/nfs/algo/intern/haoyunx11_/idea/MI/trans/en_zh_use_cache_False_2/1/output.txt'

    embedding_zh = np.load(
        r'/mnt/nfs/algo/intern/haoyunx11_/idea/MI/trans/en_zh_use_cache_False_2/1/110_hidden_layer_33.npy')
    embedding_zh = embedding_zh.squeeze()

    f = open(org_path, 'r', encoding='utf-8')
    f_aligned = open(aligned_path_enzh, 'r', encoding='utf-8')
    aligned_line = f_aligned.readlines()[0]
    line = f.readlines()[0]

    line = line.split('|||')

    f = open(output_str_path, 'r', encoding='utf-8')
    text = f.read()

    output_str_input_ids = tokenizer(text)['input_ids'][1:]

    src = line[0].strip().split(' ')
    tgt = line[1].strip().split(' ')

    aligned_line = aligned_line.strip().split(' ')

    results = []

    embedding_1 = None
    embedding_2 = None

    # [hard_word_1, hard_word_2, simple_word_1, simple_word_2]
    for word in [hard_word_1, hard_word_2, simple_word_1, simple_word_2]:

        for aligned in aligned_line:
            aligned = aligned.split('-')
            src_idx = int(aligned[0].strip())
            tgt_idx = int(aligned[1].strip())

            if src[src_idx] == word:

                # get src token index
                input_ids = tokenizer(src[src_idx])['input_ids'][1:]
                if input_ids[0] == 29871:
                    input_ids = input_ids[1:]

                if word == "On":
                    input_ids = [2951]

                start_index, end_index = find_match_index(input_ids, output_str_input_ids)
                # print('start_index: ' + str(start_index))
                # print('end_index: ' + str(end_index))
                # print(tokenizer.decode(output_str_input_ids[start_index: end_index + 1]))

                # get src embedding
                embedding_src = embedding_zh[:, start_index: end_index + 1, :]
                # print(embedding_src.shape)
                embedding_src = embedding_src.transpose(1, 0, 2)
                # print(embedding_src.shape)
                embedding_src = np.mean(embedding_src, axis=0).reshape(1, 33, 4096)
                # embedding_src = np.mean(embedding_src, axis=0).reshape(1, 768)

                if embedding_1 is None:
                    embedding_1 = embedding_src
                else:
                    embedding_1 = numpy.row_stack([embedding_1, embedding_src])

                # get tgt token index
                input_ids = tokenizer(tgt[tgt_idx])['input_ids'][1:]
                if input_ids[0] == 29871:
                    input_ids = input_ids[1:]

                start_index, end_index = find_match_index(input_ids, output_str_input_ids)
                # print('start_index: ' + str(start_index))
                # print('end_index: ' + str(end_index))
                # print(tokenizer.decode(output_str_input_ids[start_index: end_index + 1]))

                # get tgt embedding
                embedding_tgt = embedding_zh[:, start_index: end_index + 1, :]
                # print(embedding_tgt.shape)
                embedding_tgt = embedding_tgt.transpose(1, 0, 2)
                # print(embedding_tgt.shape)
                embedding_tgt = np.mean(embedding_tgt, axis=0).reshape(1, 33, 4096)
                # print(embedding_tgt.shape)
                if embedding_2 is None:
                    embedding_2 = embedding_tgt
                else:
                    embedding_2 = numpy.row_stack([embedding_2, embedding_tgt])

    source_lang = 'en'
    target_lang = 'zh'

    # 33层
    for layer in range(1, 33):
        src_emb = embedding_1[:, layer, :]
        tgt_emb = embedding_2[:, layer, :]
        similarity = cka(gram_linear(src_emb), gram_linear(tgt_emb), debiased=True)
        results.append([source_lang, target_lang, layer, similarity])

    results = pd.DataFrame(results, columns=['Source', 'Target', 'Layer', 'CKA'])
    results.to_excel(
        '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/trans/trans_en_to_zh_version/output/correlation/CKA.xlsx',
        index=False)


# def get_hidden_state_path(list_dir):
#     pattern = '_hidden_layer_33.npy'
#     dir_list = os.listdir(list_dir)
#     d_list = []
#     for d in dir_list:
#         if pattern in d:
#             d_num = d.split(pattern)[0]
#             d_list.append(int(d_num))
#     d_max = max(d_list)
#     hidden_state_path = list_dir + '/' + str(d_max) + pattern
#     return hidden_state_path


def get_hidden_state_path(list_dir):
    pattern = '_33.npy'
    dir_list = os.listdir(list_dir)
    d_list = []
    for d in dir_list:
        if pattern in d:
            d_num = d.split(pattern)[0].replace('generated_token_', '')
            d_list.append(int(d_num))
    d_max = max(d_list)
    hidden_state_path = list_dir + '/' + 'generated_token_' + str(d_max) + pattern
    return hidden_state_path


# max min
def cal_correlation_trans_whole_seq(
        path_dir_1=r'/mnt/nfs/algo/intern/haoyunx11_/idea/MI/trans/en_zh_use_cache_False_2/',
        path_dir_2=r'/mnt/nfs/algo/intern/haoyunx11_/idea/MI/trans/en_zh_qlora_3w_3epoch_use_cache_False/',
        path_w=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/trans/trans_en_to_zh_version/output/correlation/en_org_en_3w_3epoch_CKA.xlsx'):
    output_filename = 'output.txt'
    is_continue = False

    results = []
    embedding_stack_1 = None
    embedding_stack_2 = None

    for i in range(1, 1001):
        if i % 50 == 0:
            print(i)
        list_dir_1 = path_dir_1 + str(i)
        list_dir_2 = path_dir_2 + str(i)
        output_file_1 = list_dir_1 + '/' + output_filename
        output_file_2 = list_dir_2 + '/' + output_filename
        output_text_list = []
        for file in [output_file_1, output_file_2]:
            f = open(file, 'r', encoding='utf-8')
            text = f.read()
            if text == "":
                is_continue = True
                break
            else:
                is_continue = False
            output_text_list.append(text)

        if is_continue:
            print('continue launched...')
            print(i)
            continue

        hidden_state_path_1 = get_hidden_state_path(list_dir_1)
        hidden_state_path_2 = get_hidden_state_path(list_dir_2)

        embedding_1 = np.load(hidden_state_path_1)
        embedding_2 = np.load(hidden_state_path_2)

        embedding_1 = embedding_1.squeeze()
        embedding_2 = embedding_2.squeeze()

        embedding_1 = embedding_1.transpose(1, 0, 2)
        embedding_1 = np.mean(embedding_1, axis=0).reshape(1, 33, 4096)
        embedding_2 = embedding_2.transpose(1, 0, 2)
        embedding_2 = np.mean(embedding_2, axis=0).reshape(1, 33, 4096)

        if embedding_stack_1 is None:
            embedding_stack_1 = embedding_1
        else:
            embedding_stack_1 = numpy.row_stack([embedding_stack_1, embedding_1])

        if embedding_stack_2 is None:
            embedding_stack_2 = embedding_2
        else:
            embedding_stack_2 = numpy.row_stack([embedding_stack_2, embedding_2])

    print(embedding_stack_1.shape)
    print(embedding_stack_2.shape)

    source_lang = 'sum_en_org'
    target_lang = 'sum_en_30w_1epoch'

    # 33层
    for layer in range(1, 33):
        src_emb = embedding_stack_1[:, layer, :]
        tgt_emb = embedding_stack_2[:, layer, :]
        similarity = cka(gram_linear(src_emb), gram_linear(tgt_emb))
        results.append([source_lang, target_lang, layer, similarity])

    results = pd.DataFrame(results, columns=['Source', 'Target', 'Layer', 'CKA'])
    results.to_excel(path_w, index=False)


def pos_test(text):
    words = nltk.tokenize.word_tokenize(text)
    print(nltk.tag.pos_tag(words))


if __name__ == '__main__':
    # cal_correlation_test()
    # example_2()

    # text = "On Monday, scientists from the Stanford University School of Medicine announced the invention of a new diagnostic tool that can sort cells by type: a tiny printable chip that can be manufactured using standard inkjet printers for possibly about one U.S. cent each."
    # pos_test(text)

    # cal_correlation_trans_whole_seq()

    # cal_correlation_trans_whole_seq(
    #     path_dir_1=r'/mnt/nfs/algo/intern/haoyunx11_/idea/MI/trans/en_zh_use_cache_False_2/',
    #     path_dir_2=r'/mnt/nfs/algo/intern/haoyunx11_/idea/MI/trans/en_zh_qlora_3w_1epoch_use_cache_False/',
    #     path_w=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/trans/trans_en_to_zh_version/output/correlation/en_org_en_3w_1epoch_CKA.xlsx')

    # cal_correlation_trans_whole_seq(
    #     path_dir_1=r'/mnt/nfs/algo/intern/haoyunx11_/idea/MI/trans/en_zh_qlora_3w_1epoch_use_cache_False/',
    #     path_dir_2=r'/mnt/nfs/algo/intern/haoyunx11_/idea/MI/trans/en_zh_qlora_3w_3epoch_use_cache_False/',
    #     path_w=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/trans/trans_en_to_zh_version/output/correlation/en_3w_1epoch_en_3w_3epoch_CKA.xlsx')

    # cal_correlation_trans_whole_seq(
    #     path_dir_1=r'/mnt/nfs/algo/intern/haoyunx11_/idea/MI/trans/en_zh_qlora_3w_3epoch_use_cache_False/',
    #     path_dir_2=r'/mnt/nfs/algo/intern/haoyunx11_/idea/MI/trans/en_de_qlora_3w_3epoch_use_cache_False/',
    #     path_w=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/trans/trans_en_to_zh_version/output/correlation/en_3w_3epoch_de_3w_3epoch_CKA.xlsx')

    # cal_correlation_trans_whole_seq(
    #     path_dir_1=r'/mnt/nfs/algo/intern/haoyunx11_/idea/MI/trans/en_de_use_cache_False/',
    #     path_dir_2=r'/mnt/nfs/algo/intern/haoyunx11_/idea/MI/trans/en_de_qlora_3w_1epoch_use_cache_False/',
    #     path_w=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/trans/trans_en_to_zh_version/output/correlation/de_org_de_3w_1epoch_CKA.xlsx')

    # cal_correlation_trans_whole_seq(
    #     path_dir_1=r'/mnt/nfs/algo/intern/haoyunx11_/idea/MI/trans/en_zh_use_cache_False_2/',
    #     path_dir_2=r'/mnt/nfs/algo/intern/haoyunx11_/idea/MI/trans/en_de_use_cache_False/',
    #     path_w=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/trans/trans_en_to_zh_version/output/correlation/en_org_de_org_CKA.xlsx')

    # cal_correlation_trans_whole_seq(
    #     path_dir_1=r'/mnt/nfs/algo/intern/haoyunx11_/idea/MI/trans/en_de_qlora_3w_1epoch_use_cache_False/',
    #     path_dir_2=r'/mnt/nfs/algo/intern/haoyunx11_/idea/MI/trans/en_de_qlora_3w_3epoch_use_cache_False/',
    #     path_w=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/trans/trans_en_to_zh_version/output/correlation/de_3w_1epoch_de_3w_3epoch_CKA.xlsx')

    cal_correlation_trans_whole_seq(
        path_dir_1=r'/mnt/nfs/algo/intern/haoyunx11_/idea/MI/sum/en/',
        path_dir_2=r'/mnt/nfs/algo/intern/haoyunx11_/idea/MI/sum/zh_mtiayn_3epoch/',
        path_w=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/trans/trans_en_to_zh_version/output/correlation/sum_en_org_sum_en_30epoch_CKA.xlsx')
