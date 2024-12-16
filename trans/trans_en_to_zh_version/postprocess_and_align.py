import os
import re
import jieba
import jsonlines
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoConfig, BertTokenizer


# to be aligned format
# 科学家 一直 猜想 土卫六 上 存在 着 由 液态 甲烷 构成 的 海洋 。 ||| scientists have long conjectured that oceans of liquid methane exist on titan .
def postprocess_for_align_zh(path_dir=r'/mnt/nfs/algo/intern/haoyunx11_/idea/MI/trans/en_zh_use_cache_False_2',
                          src_path=r'/data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/enzh/src.txt',
                          path_w=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/trans/trans_en_to_zh_version/output/align/ready/trans_enzh_7b_False.txt'):
    w = open(path_w, 'w', encoding='utf-8')

    tokenizer = BertTokenizer.from_pretrained("/mnt/nfs/algo/intern/haoyunx11/models/model/alignment/model_without_co")

    hpy_list = []

    for i in range(1, 998):

        path = path_dir + "/" + str(i) + '/' + 'output.txt'
        f = open(path, 'r', encoding='utf-8')



        text = f.read()
        if text == '':
            hpy_list.append('')
            continue

        text = text.split('### Response:\n')

        text = text[1].strip()
        findall = re.findall(r'(?:中文.*(?:翻译|翻譯).*：|(?:翻译|翻譯).*中文.*：|對于以下文本的翻译：|这个文本将被翻译成中文。|我很高兴能够帮助您翻译文本。|翻譯結果：|將以下文本翻譯為中文。|中文：|示例：)',
                             text)
        if findall != []:
            print(findall)
            text = text.replace(findall[0], '').strip()

        text = text.split('\n')[0]
        text = text.split('</s>')[0].strip()
        hpy_list.append(text)

    src_lines = open(src_path, 'r', encoding='utf-8')

    # enzh
    for line in zip(src_lines, hpy_list):
        tokenized_en = tokenizer.basic_tokenizer.tokenize(line[0])
        tokenized_en = ' '.join(tokenized_en)

        tokenized_zh = jieba.lcut(line[1])
        tokenized_zh = ' '.join(tokenized_zh)

        ret_line = tokenized_en + " ||| " + tokenized_zh.strip()
        w.write(ret_line + '\n')

def postprocess_for_align_de(path_dir=r'/mnt/nfs/algo/intern/haoyunx11_/idea/MI/trans/en_de',
                          src_path=r'/data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/ende/eval.jsonl',
                          path_w=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/trans/trans_en_to_zh_version/output/align/ready/trans_ende_7b.txt'):
    w = open(path_w, 'w', encoding='utf-8')

    tokenizer = BertTokenizer.from_pretrained("/mnt/nfs/algo/intern/haoyunx11/models/model/alignment/model_without_co")

    hpy_list = []

    for i in range(1, 998):
        path = path_dir + "/" + str(i) + '/' + 'output.txt'
        f = open(path, 'r', encoding='utf-8')

        text = f.read()

        text = text.split('### Response:\n')

        text = text[1].strip()

        findall = re.findall(r'Translation:', text)
        if findall != []:
            print(findall)
            text = text.replace(findall[0], '').strip()

        text = text.split('\n')[0]
        text = text.split('</s>')[0].strip()
        hpy_list.append(text)

    src_lines = jsonlines.open(src_path)
    src_list = []
    for src in src_lines:
        src_list.append(src['instruction'].strip())


    # enzh
    for line in zip(src_list, hpy_list):
        tokenized_en = tokenizer.basic_tokenizer.tokenize(line[0])
        tokenized_en = ' '.join(tokenized_en)

        tokenized_de = tokenizer.basic_tokenizer.tokenize(line[1])
        tokenized_de = ' '.join(tokenized_de)

        ret_line = tokenized_en + " ||| " + tokenized_de.strip()
        w.write(ret_line + '\n')




def extract_keyword_and_align(
        aligned_path=r'/data5/haoyun.xu/study/alignment/awesome-align/output/trans/flores101_enzh/enzh_aligned',
        org_path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/trans/trans_en_to_zh_version/output/align/ready/trans_enzh_7b.txt'):
    f = open(org_path, 'r', encoding='utf-8')
    f_aligned = open(aligned_path, 'r', encoding='utf-8')
    aligned_lines = f_aligned.readlines()
    lines = f.readlines()
    for line, aligned_line in zip(lines, aligned_lines):
        line = line.split('|||')
        src = line[0].strip().split(' ')
        tgt = line[1].strip().split(' ')

        aligned_line = aligned_line.strip().split(' ')
        for aligned in aligned_line:
            aligned = aligned.split('-')
            src_idx = int(aligned[0].strip())
            tgt_idx = int(aligned[1].strip())

            print(src[src_idx])
            print(tgt[tgt_idx])
            print('****************************')




if __name__ == '__main__':
    postprocess_for_align_zh()

    # postprocess_for_align_de()
    # extract_keyword_and_align()

