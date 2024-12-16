import datasets
from feature_datasets import xnli
import jsonlines

PROMPTS = {
    "xnli": xnli.xnli_PROMPTS
}

# tok_ins = "\n\n### Instruction:\n"
# tok_res = "\n\n### Response:\n"
# prompt_input = tok_ins + "{instruction}" + tok_res

"""
train: Dataset({
        features: ['premise', 'hypothesis', 'label'],
        num_rows: 392702
    })
"""


def get_idx_list(language='en'):
    dataset = datasets.load_dataset("/mnt/nfs/algo/intern/haoyunx9/data/NLI/xnli/xnli/" + language)['train']
    idx_list = []
    for idx, data in enumerate(zip(dataset['premise'], dataset['hypothesis'], dataset['label'])):
        for prompt_dict in PROMPTS['xnli'].items():

            if prompt_dict[0] == 'simple_prompt':
                prompt_text = prompt_dict[1].format_map({"sentence1": data[0], "sentence2": data[1]})

                if len(prompt_text) > 150:
                    continue

                idx_list.append(idx)

    return idx_list


def get_xnli_train_datasets(language='en', prompt_type="standard_prompt"):
    save_path = r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/data/' + "xnli/final" + "/" + language + "_" + prompt_type + '_test.jsonl'

    w = jsonlines.open(save_path, 'w')

    dataset = datasets.load_dataset("/mnt/nfs/algo/intern/haoyunx9/data/NLI/xnli/xnli/" + language)['test']

    idx_list = get_idx_list()

    for idx, data in enumerate(zip(dataset['premise'], dataset['hypothesis'], dataset['label'])):
        if idx in idx_list:
            continue

        for prompt_dict in PROMPTS['xnli'].items():

            if prompt_dict[0] == prompt_type:
                prompt_text = prompt_dict[1].format_map({"sentence1": data[0], "sentence2": data[1]})

                temp_dict = {}
                temp_dict['input'] = prompt_text
                if data[2] == 0:
                    temp_dict['output'] = "entailment"
                elif data[2] == 1:
                    temp_dict['output'] = "neutral"
                else:
                    temp_dict['output'] = "contradiction"

                w.write(temp_dict)


# add prompt
def get_trans_train_datasets(path, language='enzh'):
    save_path = r'/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/data/' + "xnli" + "/" + language + '.jsonl'

    lines = jsonlines.open(path)

    for line in lines:
        instruction = line['instruction']
        output = line['output']


if __name__ == '__main__':
    get_xnli_train_datasets(language='en')
