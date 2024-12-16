import jsonlines
import pandas as pd


def process():
    writer = jsonlines.open(
        '/data5/haoyun.xu/data/mmlu/all/test/test_instruction+output.jsonl', mode='w')

    dataset = pd.read_parquet('/data5/haoyun.xu/data/mmlu/all/test/test-00000-of-00001.parquet')

    input_list = []
    output_list = []
    for line in dataset.iterrows():
        line = line[1]
        question = line['question']
        choices = line['choices']
        answer = line['answer']
        input = question
        for idx, c in enumerate(choices):
            if idx == 0:
                input += '\nA. ' + str(c)
            elif idx == 1:
                input += '\nB. ' + str(c)
            elif idx == 2:
                input += '\nC. ' + str(c)
            elif idx == 3:
                input += '\nD. ' + str(c)

        input_list.append(input)
        output_list.append(answer)

    instruction = 'Identify the correct answer from the choices below.'

    for line in zip(input_list, output_list):

        input = line[0]
        output = line[1]

        if input == "":
            text = '[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant.\n<</SYS>>\n\n' + instruction + ' [/INST]'
        else:
            text = '[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant.\n<</SYS>>\n\n' + instruction + '\n' + input + ' [/INST]'

        temp_dict = {}
        temp_dict['instruction'] = text
        temp_dict['output'] = output

        writer.write(temp_dict)


if __name__ == '__main__':
    process()