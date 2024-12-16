import jsonlines
import json
import transformers


def cut_4096():
    lines = jsonlines.open(
        r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/data/CrossSum/english-french/english-french_test.jsonl')
    writer = jsonlines.open(
        r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/data/CrossSum/english-french/4096_cut_english-french_test.jsonl',
        'w')

    model_name_or_path = '/mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf'
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
    prompt = "Given the below English article, generate a summary in French." + "\nArticle: "

    max_len = 0
    for idx, line in enumerate(lines):

        input_s = prompt + line['text']
        output_s = line['summary']

        tokenized = tokenizer([input_s, output_s])
        input_ids = tokenized['input_ids']

        length = len(input_ids[0]) + 2

        if max_len < length:
            max_len = length

        if length >= 4096:
            print(line['text'])
            print(length)
            continue

        writer.write(line)

    print(max_len)


def check_length():
    # 2001
    # path = r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/data/CrossSum/english-french/2000_cut_english-french_train.jsonl'
    # path = r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/data/CrossSum/hindi-french/2000_cut_hindi-french_train.jsonl'
    # path = r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/data/CrossSum/english-french/1200_cut_english-french_train.jsonl'
    # path = r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/data/xnli/final/en_standard_prompt_train_3w.jsonl'
    # path = r'/data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/news/train_2w.jsonl'
    path = r'/data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/news/train_10w_cleaned.jsonl'

    lines = jsonlines.open(path)

    # prompt = 'Translate the following text from French to Chinese.'
    model_name_or_path = '/mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf'
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)

    # prompt = "Given the below Hindi article, generate a summary in Chinese." + "\nArticle: "

    max_len = 0
    # input_ = 'text'
    # output_ = 'summary'

    input_ = 'input'
    output_ = 'output'

    # input_ = 'instruction'
    # output_ = 'output'

    count = 0
    for idx, line in enumerate(lines):
        prompt = 'Translate the following text from English to Chinese.'
        # prompt = sft_config.prompt
        input_s = prompt + ' ' + line['instruction']
        output_s = line['output']

        # instruction = line[input_]
        # output = line[output_]

        # input_s = prompt + ' ' + instruction
        # input_s = instruction
        # output_s = output

        tokenized = tokenizer([input_s, output_s])

        input_ids = tokenized['input_ids']

        length = len(input_ids[0])

        if max_len < length:
            max_len = length

        if length >= 160:
            count += 1

        # if idx <= 315 or idx == 316:
        #     print(line[output])
        #     print(length)
        #     continue

        # if length >= 512:
        #     print(line[output_])
        #     print(length)
        #     continue

    print(max_len)
    print(count)


def gen_largest_set(
        path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/data/CrossSum/english-chinese_simplified/1600_cut_english-chinese_simplified_train.jsonl',
        amount=20):
    writer = jsonlines.open(
        '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/data/CrossSum/english-chinese_simplified/1600_cut_english-chinese_simplified_train_max.jsonl',
        mode='w')

    lines = jsonlines.open(path)
    model_name_or_path = '/mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf'
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)

    prompt = "Given the below English article, generate a summary in Chinese." + "\nArticle: "

    max_len = 0
    input_ = 'text'
    output_ = 'summary'

    # input_ = 'instruction'
    # output_ = 'output'

    # for idx, line in enumerate(lines):
    #     instruction = line[input_]
    #     output = line[output_]
    #
    #     input_s = prompt + ' ' + instruction
    #     output_s = output
    #
    #     tokenized = tokenizer([input_s, output_s])
    #
    #     input_ids = tokenized['input_ids']
    #
    #     length = len(input_ids[0])
    #
    #     if max_len < length:
    #         max_len = length

    # if idx <= 315 or idx == 316:
    #     print(line[output])
    #     print(length)
    #     continue

    # if length >= 512:
    #     print(line[output_])
    #     print(length)
    #     continue

    sorted_list = sorted(lines, key=lambda i: len(tokenizer([prompt + ' ' + i[input_], i[output_]])['input_ids'][0]),
                         reverse=True)
    # print(sorted_list[:20])

    for line in sorted_list[:20]:
        writer.write(line)
        instruction = line[input_]
        output = line[output_]
        input_s = prompt + ' ' + instruction
        output_s = output
        tokenized = tokenizer([input_s, output_s])
        input_ids = tokenized['input_ids']
        length = len(input_ids[0])

        print(line['text'])
        print(line['summary'])
        print(length)


def clean_MT_news(path=r'/data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/news/enfr/train_10w.jsonl',
                  path_w=r'/data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/news/enfr/train_10w_cleaned.jsonl'):
    lines = jsonlines.open(path)
    writer = jsonlines.open(path_w, 'w')

    for line in lines:
        # instruction
        output = line['output'].strip()
        if output == "":
            continue

        if '—' in output:
            if output[-1] == '—':
                continue

        writer.write(line)


def process_lego_mt(path_dir=r'/data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/news/enbs', src='en',
                    tgt='bs'):
    for temp in ['train', 'test', 'validation']:
        # for temp in ['validation']:

        src_path = path_dir + '/' + temp + '.' + src
        tgt_path = path_dir + '/' + temp + '.' + tgt

        writer = jsonlines.open(path_dir + '/' + temp + '.jsonl', 'w')

        f_1 = open(src_path, 'r', encoding='utf-8')
        f_2 = open(tgt_path, 'r', encoding='utf-8')

        lines_1 = f_1.readlines()
        lines_2 = f_2.readlines()

        temp_dict = {}

        for idx, line in enumerate(zip(lines_1, lines_2)):
            if idx >= 10000:
                continue
            temp_dict['instruction'] = line[0].strip()
            temp_dict['output'] = line[1].strip()
            writer.write(temp_dict)


# {"input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nTranslate the following phrase into French.\n\n### Input:\nI miss you\n\n", "output": "### Response:\nJe te manque."}
# 1 + 5
# a b b b b b a
def add_alpaca(alpaca_path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/other/data/alpaca/eval.jsonl',
               mt_path=r'/data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/news/dev_1k.jsonl',
               w_path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/other/data/alpaca/alpaca_add_enzh_1w_eval.jsonl'):
    prefix_instruction = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nTranslate the following text from English to Chinese.\n\n### Input:\n"
    subfix_instruction = "\n\n"
    prefix_output = "### Response:\n"

    alpaca_lines = jsonlines.open(alpaca_path)
    mt_lines = jsonlines.open(mt_path)
    writer = jsonlines.open(w_path, 'w')

    alpaca_dict = {'instruction': [], 'output': []}
    mt_dict = {'instruction': [], 'output': []}
    for alpaca_line in alpaca_lines:
        alpaca_dict['instruction'].append(alpaca_line['input'].strip())
        alpaca_dict['output'].append(alpaca_line['output'].strip())

    for idx, mt_line in enumerate(mt_lines):
        if idx >= 10000:
            break
        instruction = prefix_instruction + mt_line['instruction'].strip() + subfix_instruction
        output = prefix_output + mt_line['output'].strip()
        mt_dict['instruction'].append(instruction)
        mt_dict['output'].append(output)

    temp_dict = {}
    alpaca_idx = 0
    mt_idx = 0
    count = 0
    while True:
        print(count % 6)
        if count % 6 == 0:
            # try:
            for _ in range(6):
                temp_dict['instruction'] = mt_dict['instruction'][mt_idx]
                temp_dict['output'] = mt_dict['output'][mt_idx]
                writer.write(temp_dict)
                mt_idx += 1
            # except:
            #     print('111')
            count += 1

        else:
            for _ in range(6):
                try:
                    temp_dict['instruction'] = alpaca_dict['instruction'][alpaca_idx]
                    temp_dict['output'] = alpaca_dict['output'][alpaca_idx]
                    writer.write(temp_dict)
                    alpaca_idx += 1
                except:
                    print('111')
            count += 1


if __name__ == '__main__':
    # cut_4096()
    # check_length()
    # gen_largest_set()
    clean_MT_news()

    # add_alpaca()

    # process_lego_mt()
