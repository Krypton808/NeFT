import datasets
import jsonlines
import pyarrow.parquet as pq
import jsonlines
import json
from transformers import AutoTokenizer


def process():
    writer = jsonlines.open(
        '/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/alpaca/alpaca_instruction+output.jsonl', mode='w')
    dataset = datasets.load_dataset('/mnt/nfs/algo/intern/haoyunx9/data/alpaca')
    print(dataset)
    data = dataset['train']
    instruction_list = data['instruction']
    input_list = data['input']
    # print(input_list)
    output_list = data['output']

    for line in zip(instruction_list, input_list, output_list):
        instruction = line[0]
        input = line[1]
        output = line[2]

        if input == "":
            text = '[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant.\n<</SYS>>\n\n' + instruction + ' [/INST]'
        else:
            text = '[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant.\n<</SYS>>\n\n' + instruction + '\n' + input + ' [/INST]'

        temp_dict = {}
        temp_dict['instruction'] = text
        temp_dict['output'] = output

        writer.write(temp_dict)


def process_data_Llama3_2_1B_Instruct():
    table = pq.read_table("/mnt/data1/study/data/alpaca_gpt4/train-00000-of-00001-6ef3991c06080e14.parquet")
    # table = pq.read_table("/mnt/data1/study/data/alpaca/train-00000-of-00001-a09b74b3ef9c3b56.parquet")
    df = table.to_pandas()
    writer = jsonlines.open(
        "/mnt/data1/study/data/alpaca_gpt4/jsonl/split_sys_prompt/alpaca_gpt4_split_sys_prompt_2.jsonl",
        mode='w')
    # writer = jsonlines.open("/mnt/data1/study/data/alpaca/jsonl/split_sys_prompt/alpaca_split_sys_prompt_2.jsonl",
    #                         mode='w')

    for idx, row in df.iterrows():
        # print(idx)

        instruction = row['instruction']
        input = row['input']
        output = row['output']
        text = row['text']

        temp_dict = {
            "instruction": instruction,
            "input": input,
            "output": output,
            "text": text
        }

        writer.write(temp_dict)


def process_data_Llama3_2_1B_Instruct_add_split():
    # table = pq.read_table("/mnt/data1/study/data/alpaca/train-00000-of-00001-a09b74b3ef9c3b56.parquet")
    table = pq.read_table("/mnt/data1/study/data/alpaca_gpt4/train-00000-of-00001-6ef3991c06080e14.parquet")
    df = table.to_pandas()
    # writer = jsonlines.open(
    #     "/mnt/data1/study/data/alpaca/jsonl/split_sys_prompt/alpaca_split_sys_prompt_2.jsonl",
    #     mode='w')
    writer = jsonlines.open(
        "/mnt/data1/study/data/alpaca_gpt4/jsonl/split_sys_prompt/alpaca_gpt4_split_sys_prompt_2.jsonl",
        mode='w')

    for idx, row in df.iterrows():
        instruction = row['instruction']
        input = row['input']
        output = row['output']
        text = row['text']
        sys_prompt = text.split("\n\n### Instruction:\n")[0]
        if input == "":
            ft_input = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{sys_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>Instruction: {instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        else:
            ft_input = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{sys_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>Instruction: {instruction}\nInput: {input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

        output = f"{output}<|eot_id|><|end_of_text|>"

        temp_dict = {
            "input": ft_input,
            "output": output
        }

        writer.write(temp_dict)


def process_data_Mistral_7B_v1_add_split(input_file, output_file):
    # <s>[INST] Instruction [/INST] Model answer</s>[INST] Follow-up instruction [/INST]

    table = pq.read_table(input_file)
    df = table.to_pandas()
    writer = jsonlines.open(output_file, mode='w')

    for idx, row in df.iterrows():
        instruction = row['instruction']
        input = row['input']
        output = row['output']
        text = row['text']
        sys_prompt = text.split("\n\n### Instruction:\n")[0]
        if input == "":
            ft_input = f"<s>[INST] {sys_prompt}\nInstruction: {instruction} [/INST]"
        else:
            ft_input = f"<s>[INST] {sys_prompt}\nInstruction: {instruction}\nInput: {input} [/INST]"

        output = f"{output}</s>"

        temp_dict = {
            "input": ft_input,
            "output": output
        }

        writer.write(temp_dict)


def encoded():
    tokenizer_path = "/data/tigerbot/tigerbot_geely/test/haoyunx/work/models/llama/Llama-3.2-1B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    tokenized = tokenizer.decode([128001])
    print(tokenized)

    tokenized = tokenizer.decode([128008])
    print(tokenized)

    tokenized = tokenizer.decode([128009])
    print(tokenized)


if __name__ == '__main__':
    # process()

    # process_data_Llama3_2_1B_Instruct()

    process_data_Llama3_2_1B_Instruct_add_split()

    process_data_Mistral_7B_v1_add_split(
        input_file="/mnt/data1/study/data/alpaca/train-00000-of-00001-a09b74b3ef9c3b56.parquet",
        output_file="/mnt/data1/study/data/alpaca/jsonl/split_sys_prompt/Mistral_7B_Instruct_v3_alpaca_split_sys_prompt.jsonl")
    process_data_Mistral_7B_v1_add_split(
        input_file="/mnt/data1/study/data/alpaca_gpt4/train-00000-of-00001-6ef3991c06080e14.parquet",
        output_file="/mnt/data1/study/data/alpaca_gpt4/jsonl/split_sys_prompt/Mistral_7B_Instruct_v3_alpaca_gpt4_split_sys_prompt.jsonl")

    # encoded()
