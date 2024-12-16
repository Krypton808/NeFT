import datasets
import jsonlines
import pandas as pd
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', -1)


def load_mmlu():
    writer = jsonlines.open("/data2/haoyun/mmlu/validation.jsonl", mode='w')
    data = pd.read_parquet('/data5/haoyun.xu/data/mmlu/all/test/validation-00000-of-00001.parquet')
    prefix = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nIdentify the correct answer from the choices below.\n\n### Input:\n"

    for index, row in data.iterrows():
        question = prefix + row['question']

        subject = row['subject']
        choices = row['choices']
        choices_text = ''
        choices_dict = {}
        # \nA. Google\nB. Apple\nC. Microsoft\nD. Twitter
        for idx, c in enumerate(choices):
            if idx == 0:
                choices_text += '\nA. ' + c
                choices_dict[idx] = '\n\n### Response:\nA. ' + c
            elif idx == 1:
                choices_text += '\nB. ' + c
                choices_dict[idx] = '\n\n### Response:\nB. ' + c
            elif idx == 2:
                choices_text += '\nC. ' + c
                choices_dict[idx] = '\n\n### Response:\nC. ' + c
            elif idx == 3:
                choices_text += '\nD. ' + c
                choices_dict[idx] = '\n\n### Response:\nD. ' + c

        instruction = question + choices_text

        answer = row['answer']

        temp_dict = {}
        temp_dict['instruction'] = '[INST]' + instruction + '[/INST]'
        temp_dict['output'] = choices_dict[int(answer)]
        writer.write(temp_dict)

def prepare_mmlu_for_Mistral_7B_v1():
    # f"<s>[INST] {sys_prompt}\nInstruction: {instruction}\nInput: {input} [/INST]"

    # writer = jsonlines.open("/mnt/data1/study/data/mmlu/test.jsonl", mode='w')
    # data = pd.read_parquet('/mnt/data1/study/data/mmlu/test-00000-of-00001.parquet')

    writer = jsonlines.open("/mnt/data1/study/data/mmlu/validation.jsonl", mode='w')
    data = pd.read_parquet('/mnt/data1/study/data/mmlu/validation-00000-of-00001.parquet')

    sys_prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
    instruction = "Identify the correct answer from the choices below."

    # prefix = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nIdentify the correct answer from the choices below.\n\n### Input:\n"

    for index, row in data.iterrows():
        question = row['question']

        subject = row['subject']
        choices = row['choices']
        choices_text = ''
        choices_dict = {}
        # \nA. Google\nB. Apple\nC. Microsoft\nD. Twitter
        for idx, c in enumerate(choices):
            if idx == 0:
                choices_text += '\nA. ' + c
                # choices_dict[idx] = '\n\n### Response:\nA. ' + c
                choices_dict[idx] = 'A. ' + c
            elif idx == 1:
                choices_text += '\nB. ' + c
                # choices_dict[idx] = '\n\n### Response:\nB. ' + c
                choices_dict[idx] = 'B. ' + c
            elif idx == 2:
                choices_text += '\nC. ' + c
                # choices_dict[idx] = '\n\n### Response:\nC. ' + c
                choices_dict[idx] = 'C. ' + c
            elif idx == 3:
                choices_text += '\nD. ' + c
                # choices_dict[idx] = '\n\n### Response:\nD. ' + c
                choices_dict[idx] = 'D. ' + c

        input = question + choices_text
        infer_input = f"<s>[INST] {sys_prompt}\nInstruction: {instruction}\nInput: {input} [/INST]"
        answer = row['answer']

        temp_dict = {}
        temp_dict['instruction'] = infer_input
        temp_dict['output'] = choices_dict[int(answer)]
        writer.write(temp_dict)

def eval_mmlu():
    output_path = "/mnt/data1/output/mmlu/llama3_2_1B_alpaca_neuron_6000_2/6500step/output.jsonl"
    # output_path = "/mnt/data1/output/mmlu/Mistral_7B_v1_alpaca_2/6500step/output.jsonl"
    data_path = "/mnt/data1/study/data/mmlu/validation.jsonl"
    data_lines = jsonlines.open(data_path, mode='r')
    output_lines = jsonlines.open(output_path, mode='r')

    correct_count = 0
    for idx, lines in enumerate(zip(data_lines, output_lines)):
        answer = lines[0]['output']
        response = lines[1]['response']

        if answer[0] == response[0]:
            correct_count += 1

    accuracy = correct_count / (idx + 1)
    print(accuracy)





if __name__ == '__main__':
    # load_mmlu()

    # prepare_mmlu_for_Mistral_7B_v1()

    eval_mmlu()