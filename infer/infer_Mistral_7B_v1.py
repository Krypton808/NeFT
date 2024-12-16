import os

import fire
import jsonlines
import torch
import transformers
import datasets
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoConfig, LlamaTokenizer, \
    LlamaForCausalLM
from peft import PeftModel
import csv


def get_model(model_path, lora_weight_path=False, generation_config=False):
    if generation_config:
        model = transformers.AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                                                  device_map='auto')
    else:
        config = AutoConfig.from_pretrained(model_path)
        config.max_new_tokens = 1024
        model = transformers.AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                                                  device_map='auto', config=config)
    if lora_weight_path:
        print("load lora weight...")
        model = PeftModel.from_pretrained(model, lora_weight_path, is_trainable=False)
        model = model.merge_and_unload()

    return model


def main(
        model_path: str,
        validate_file_path: str,
        key_name,
        prompt='',
        lora_weight_path=False,
        max_input_length: int = 512,
        max_generate_length: int = 1024,
        pad_to_max_length=True,
        save_path='temp_output.csv'
):
    writer = jsonlines.open(save_path, mode='w')

    print(f"loading model: {model_path}...")

    generation_config = GenerationConfig.from_pretrained(model_path)
    generation_config.do_sample = False
    generation_config.use_cache = True
    generation_config.no_repeat_ngram_size = 4
    generation_config.max_new_tokens = 1024

    model = get_model(model_path, lora_weight_path, generation_config)

    device = torch.cuda.current_device()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=max_generate_length,
        padding_side="left",
        truncation_side='left',
        padding=True,
        truncation=True
    )

    tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.model_max_length is None or tokenizer.model_max_length > max_generate_length:
        tokenizer.model_max_length = max_generate_length

    val_dataset = datasets.load_dataset("json", data_files={'validation': validate_file_path})

    val_input = val_dataset['validation'][key_name]

    input_list = []
    # input = ""
    sys_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    for instruction in val_input:
        generate_input = f"<s>[INST] {sys_prompt}\nInstruction: {instruction} [/INST]"

        input_list.append(generate_input)

    dataloader = DataLoader(input_list, batch_size=1, shuffle=False)

    output_str_list = []

    for Instruction, data in zip(val_input, dataloader):
        # print(data)
        # print()
        inputs = tokenizer(data, return_tensors='pt', padding=True, max_length=pad_to_max_length,
                           add_special_tokens=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        output = model.generate(**inputs, **generation_config.to_dict())
        i = 0
        for o in output:
            output_str = tokenizer.decode(o, skip_special_tokens=False, spaces_between_special_tokens=False)
            # print(output_str.strip())

            # output_str_list_count_eot = output_str.split("<|eot_id|>")
            output_str_list_count_eot = output_str.split("</s>")
            print(output_str.strip())
            output_str_raw = output_str.strip()

            print(len(output_str_list_count_eot))

            output_str = output_str.split("[/INST]")[-1]
            # output_str = output_str.replace("<|eot_id|>", "")
            output_str = output_str.replace("</s>", "")
            output_str_list.append(output_str)
            print('*******************')
            writer.write({"prompt": Instruction, "response": output_str.strip(), "response_raw": output_str_raw})
            i += 1


if __name__ == "__main__":
    fire.Fire(main)

"""

CUDA_VISIBLE_DEVICES=4 python infer_Llama-3.2-1B-Instruct.py \
    --model_path /mnt/data1/output/model/llama3_2_1B_alpaca/checkpoint-6500 \
    --validate_file_path /data/tigerbot/tigerbot_geely/test/haoyunx/study/google_research/instruction_following_eval/data/input_data.jsonl \
    --key_name 'prompt' \
    --save_path /mnt/data1/output/ifeval/llama3_2_1B_alpaca/6500step/output.jsonl
    


CUDA_VISIBLE_DEVICES=5 python infer_Llama-3.2-1B-Instruct.py \
    --model_path /mnt/data1/output/model/llama3_2_1B_alpaca_neuron_6000/checkpoint-19500 \
    --validate_file_path /data/tigerbot/tigerbot_geely/test/haoyunx/study/google_research/instruction_following_eval/data/input_data.jsonl \
    --key_name 'prompt' \
    --save_path /mnt/data1/output/ifeval/llama3_2_1B_alpaca_6500step_NeFT/neuron_6500step/6000/19500step/output.jsonl
    
    
CUDA_VISIBLE_DEVICES=7 python infer_Mistral_7B_v1.py \
    --model_path /mnt/data1/output/model/Mistral_7B_v1_alpaca_neuron_150000_2/checkpoint-6500 \
    --validate_file_path /data/tigerbot/tigerbot_geely/test/haoyunx/study/google_research/instruction_following_eval/data/input_data.jsonl \
    --key_name 'prompt' \
    --save_path /mnt/data1/output/ifeval/Mistral_7B_v1_alpaca_neuron_150000_2/6500step/output.jsonl
    
    
CUDA_VISIBLE_DEVICES=3 python infer_Mistral_7B_v1.py \
    --model_path /mnt/data1/output/model/Mistral_7B_v1_alpaca_neuron_150000_2/checkpoint-6500 \
    --validate_file_path /mnt/data1/study/data/mmlu/validation.jsonl \
    --key_name 'instruction' \
    --save_path /mnt/data1/output/mmlu/Mistral_7B_v1_alpaca_neuron_150000_2/6500step/output.jsonl


"""
