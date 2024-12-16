import torch
import datasets
from transformers import AutoTokenizer



PROMPTS = {
    "xnli": xnli.xnli_PROMPTS
}

"""
train: Dataset({
        features: ['premise', 'hypothesis', 'label'],
        num_rows: 392702
    })
"""

def prepare_data(system_output_1, system_output_2, source_path, human_score_path,
                 prompt=r'Translate the following text from English to Chinese.'):
    f_1 = open(system_output_1, 'r', encoding='utf-8')
    f_2 = open(system_output_2, 'r', encoding='utf-8')
    f_s = open(source_path, 'r', encoding='utf-8')

    system_name_1 = system_output_1.split('/')[-1].replace('.txt')
    system_name_2 = system_output_2.split('/')[-1].replace('.txt')

    system_1_score_list, system_2_score_list = locate_system_human_score(system_name_1, system_name_2, human_score_path)

    for line in zip(system_name_1, system_name_2, source_path, system_1_score_list, system_2_score_list):
        system_1_score = line[3]
        system_2_score = line[4]

        if system_1_score == None or system_2_score == None:
            continue

        input_s = prompt + ' '




def locate_system_human_score(system_name_1, system_name_2, human_score_path):
    f_human_score = open(human_score_path, 'r', encoding='utf-8')

    system_1_score_list = []
    system_2_score_list = []

    lines = f_human_score.readlines()
    for line in lines:
        if system_name_1 in line:
            system_1_score_list.append(line.split('	')[-1].strip())
        elif system_name_2 in line:
            system_2_score_list.append(line.split('	')[-1].strip())

    assert len(system_1_score_list) == len(system_2_score_list), 'not align'
    return system_1_score_list, system_2_score_list




def get_idx_list(language='en'):
    dataset = datasets.load_dataset("/mnt/nfs/algo/intern/haoyunx9/data/NLI/xnli/xnli/" + language)['train']
    # dataset = dataset[:amount+1]

    max_len = 0
    count = 0
    data_list = []
    idx_list = []
    content_list = []
    label_list = []
    for idx, data in enumerate(zip(dataset['premise'], dataset['hypothesis'], dataset['label'])):
        for prompt_dict in PROMPTS['xnli'].items():

            if prompt_dict[0] == 'simple_prompt':
                prompt_text = prompt_dict[1].format_map({"sentence1": data[0], "sentence2": data[1]})

                if len(prompt_text) > 150:
                    continue

                idx_list.append(idx)

    return idx_list



if __name__ == '__main__':
    prepare_data(
        system_output_1=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/scorer/data/enzh/system_outputs/Baidu-system.6932.txt',
        system_output_2=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/scorer/data/enzh/system_outputs/BTRANS.6821.txt',
        source_path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/scorer/data/enzh/source/en-zh.txt',
        human_score_path=r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/scorer/data/enzh/human_scores/en-zh.wmt-z.seg.score')
