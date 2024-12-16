import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import fire
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
        config.max_new_tokens = 2048
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
        max_generate_length: int = 2048,
        pad_to_max_length=True,
        save_path='temp_output.csv'
):
    csv_file = open(save_path, 'w', newline='', encoding='utf-8')
    writer = csv.writer(csv_file)

    print(f"loading model: {model_path}...")

    # generation_config = False

    generation_config = GenerationConfig.from_pretrained(model_path)
    # generation_config.temperature = 0.0
    # generation_config.output_hidden_states = True
    # generation_config.output_attentions = True
    # generation_config.return_dict_in_generate = True
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

    # prompt = 'Переведите следующий текст с английского языка на русский.'
    # prompt = 'Translate the following text from German to English.'

    val_input = val_dataset['validation'][key_name]

    input_list = []
    for input in val_input:
        input_list.append(prompt + ' ' + input)

    dataloader = DataLoader(input_list, batch_size=1, shuffle=False)

    output_str_list = []

    # w = open(save_path, 'w', encoding='utf-8')

    for data in dataloader:
        inputs = tokenizer(data, return_tensors='pt', padding=True, max_length=pad_to_max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        output = model.generate(**inputs, **generation_config.to_dict())
        # output = model.generate(**inputs, **generation_config.to_dict())
        i = 0
        for o in output:
            output_str = tokenizer.decode(o, skip_special_tokens=True, spaces_between_special_tokens=False)
            output_str_list.append(output_str)

            print(output_str.strip())
            # w.write(output_str.replace(data[i], '').strip() + '\n')
            writer.writerow([output_str.replace(data[i], '').replace('\n', '').strip()])
            i += 1

    # for s in output_str_list:
    #     w.write(s.replace(s.split('.')[0], '').replace(s.split('.')[1], '') + '\n')
    #     print(s)
    # print(len(output_str_list))


if __name__ == "__main__":
    fire.Fire(main)

"""

final

CUDA_VISIBLE_DEVICES=4,5 python lora_model_infer.py \
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/final/mt/train_neuron/hizh/750step_neuron/checkpoint-750 \
    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/enzh/eval.jsonl \
    --prompt 'Translate the following text from English to Chinese.' \
    --key_name 'instruction' \
    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/mt/train_neuron/hizh/750step/enzh_output.csv


CUDA_VISIBLE_DEVICES=2,3 python lora_model_infer.py \
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/final/mt/train_neuron/hifr/750step_neuron/checkpoint-750 \
    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/enfr/eval.jsonl \
    --prompt 'Translate the following text from English to French.' \
    --key_name 'instruction' \
    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/mt/train_neuron/hifr/750step/enfr_output.csv


CUDA_VISIBLE_DEVICES=6,7 python lora_model_infer.py \
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/final/mt/org/enzh/checkpoint-3200 \
    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/hizh/eval.jsonl \
    --prompt 'Translate the following text from Hindi to Chinese.' \
    --key_name 'instruction' \
    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/mt/org/enzh/3200step/hizh_output.csv >ACL_infer_output_log.out 2>&1 &
    
CUDA_VISIBLE_DEVICES=6 python infer.py \
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/final/mt/org/enfr/checkpoint-800 \
    --validate_file_path /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/enfr/eval.jsonl \
    --prompt 'Translate the following text from English to French.' \
    --key_name 'instruction' \
    --save_path /data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/infer/output/final/mt/org/enfr/800step/enfr_output.csv >infer_output_log.out 2>&1 &


bash infer.sh >infer_output.out 2>&1 &

"""
