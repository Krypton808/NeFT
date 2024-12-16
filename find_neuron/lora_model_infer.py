import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

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

    generation_config = GenerationConfig.from_pretrained("/data2/haoyun/models/llm/Llama-2-7b-chat-hf")
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
    if prompt == '':
        for input in val_input:
            input_list.append(input)
    else:
        for input in val_input:
            input_list.append(prompt + ' ' + input)

    dataloader = DataLoader(input_list, batch_size=4, shuffle=False)

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
            temp = output_str.replace(data[i], '').replace('\n', '').strip()
            temp = temp.replace('### Response:', '').strip()
            # writer.writerow([output_str.replace(data[i], '').replace('\n', '').strip()])
            writer.writerow([temp])
            i += 1

    # for s in output_str_list:
    #     w.write(s.replace(s.split('.')[0], '').replace(s.split('.')[1], '') + '\n')
    #     print(s)
    # print(len(output_str_list))


if __name__ == "__main__":
    fire.Fire(main)

"""

# enzh
CUDA_VISIBLE_DEVICES=0,1 python lora_model_infer.py \
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/mt/train_neuron/train_neuron_enzh_10w_/checkpoint-1250 \
    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/enzh/eval.jsonl \
    --prompt 'Translate the following text from English to Chinese.' \
    --key_name 'instruction' \
    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/sn/greedy_decode/train_neuron/enzh/temp/enzh_output.csv


CUDA_VISIBLE_DEVICES=2,3 python lora_model_infer.py \
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/mt/train_enzh_org_/checkpoint-2500 \
    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/enzh/eval.jsonl \
    --prompt 'Translate the following text from English to Chinese.' \
    --key_name 'instruction' \
    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/sn/greedy_decode/enzh/temp/enzh_output.csv


python lora_model_infer.py \
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/mt/train_neuron/enzh_train_neuron_4_328_in_out_progressive/checkpoint-500 \
    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/enzh/eval.jsonl \
    --prompt 'Translate the following text from Chinese to English.' \
    --key_name 'output' \
    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/sn/llama2_7b_chat_3w_500steps_enzh_4_328_in_out_step_progressive/zhen_output.csv >llama2_7b_chat_3w_500steps_enzh_4_328_in_out_step_progressive_zhen_output.out 2>&1 &


# ende
python lora_model_infer.py \
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/mt/train_neuron/enzh_train_neuron_4_328_in_out_progressive/checkpoint-500 \
    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/ende/eval.jsonl \
    --prompt 'Translate the following text from English to German.' \
    --key_name 'instruction' \
    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/sn/llama2_7b_chat_3w_500steps_enzh_4_328_in_out_step_progressive/ende_output.csv >llama2_7b_chat_3w_500steps_enzh_4_328_in_out_step_progressive_ende_output.out 2>&1 &


python lora_model_infer.py \
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/mt/train_neuron/enzh_train_neuron_4_328_in_out_progressive/checkpoint-500 \
    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/ende/eval.jsonl \
    --prompt 'Translate the following text from German to English.' \
    --key_name 'output' \
    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/sn/llama2_7b_chat_3w_500steps_enzh_4_328_in_out_step_progressive/deen_output.csv >llama2_7b_chat_3w_500steps_enzh_4_328_in_out_step_progressive_deen_output.out 2>&1 &


# dezh
CUDA_VISIBLE_DEVICES=0,1 python lora_model_infer.py \
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/mt/train_neuron/train_neuron_enzh_10w_/checkpoint-1250 \
    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/dezh/eval.jsonl \
    --prompt 'Translate the following text from German to Chinese.' \
    --key_name 'instruction' \
    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/sn/greedy_decode/train_neuron/enzh/temp/dezh_output.csv

CUDA_VISIBLE_DEVICES=2,3 python lora_model_infer.py \
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/mt/train_enzh_org_/checkpoint-2500 \
    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/dezh/eval.jsonl \
    --prompt 'Translate the following text from German to Chinese.' \
    --key_name 'instruction' \
    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/sn/greedy_decode/enzh/temp/dezh_output.csv



python lora_model_infer.py \
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/mt/train_neuron/enzh_train_neuron_4_328_in_out_progressive/checkpoint-500 \
    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/dezh/eval.jsonl \
    --prompt 'Translate the following text from Chinese to German.' \
    --key_name 'output' \
    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/sn/llama2_7b_chat_3w_500steps_enzh_4_328_in_out_step_progressive/zhde_output.csv >llama2_7b_chat_3w_500steps_enzh_4_328_in_out_step_progressive_zhde_output.out 2>&1 &

# cspt
python lora_model_infer.py \
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/mt/train_neuron/cspt_train_neuron_all_layer_328_in_out_progressive/checkpoint-500 \
    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/cspt/eval.jsonl \
    --prompt 'Translate the following text from Czech to Portuguese.' \
    --key_name 'output' \
    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/sn/greedy_decode/cspt/cspt_train_neuron_all_layers_328_in_out_progressive/500steps/cspt_output.csv >csptall_layers_328_in_out_progressive.out 2>&1 &





# enzh
python lora_model_infer.py \
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/mt/train_neuron/enzh_train_neuron_cos_score_0.9995/checkpoint-1500 \
    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/enzh/eval.jsonl \
    --prompt 'Translate the following text from English to Chinese.' \
    --key_name 'instruction' \
    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/sn/greedy_decode/train_neuron/enzh/enzh_train_neuron_cos_score_0.9995/1500steps/enzh_output.csv >enzh_train_neuron_cos_score_0.9995.out 2>&1 &

# ende
CUDA_VISIBLE_DEVICES=0,1 python lora_model_infer.py \
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/mt/train_ende_org/checkpoint-1500 \
    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/ende/eval.jsonl \
    --prompt 'Translate the following text from English to German.' \
    --key_name 'instruction' \
    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/sn/greedy_decode/ende/train_ende_org/1500steps/ende_output.csv >train_ende_org.out 2>&1 &


python lora_model_infer.py \
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/mt/train_enzh_org/checkpoint-500 \
    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/ende/eval.jsonl \
    --prompt 'Translate the following text from German to English.' \
    --key_name 'output' \
    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/sn/llama2_7b_chat_3w_500steps_enzh/deen_output.csv >llama2_7b_chat_3w_500steps_enzh_deen_output.out 2>&1 &


# dezh
CUDA_VISIBLE_DEVICES=6,7 python lora_model_infer.py \
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/mt/train_enzh_org/checkpoint-2500 \
    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/dezh/eval.jsonl \
    --prompt 'Translate the following text from German to Chinese.' \
    --key_name 'instruction' \
    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/sn/greedy_decode/enzh/llama2_7b_chat_3w_2500steps_enzh/dezh_output_full.csv >dezh_output.out 2>&1 &


python lora_model_infer.py \
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/mt/train_enzh_org/checkpoint-500 \
    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/dezh/eval.jsonl \
    --prompt 'Translate the following text from Chinese to German.' \
    --key_name 'output' \
    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/sn/greedy_decode/llama2_7b_chat_3w_1000steps_enzh/zhde_output.csv >llama2_7b_chat_3w_500steps_enzh_zhde_output.out 2>&1 &

# cspt
CUDA_VISIBLE_DEVICES=6,7 python lora_model_infer.py \
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/mt/train_neuron/enzh_train_neuron_cos_score_0.9995/checkpoint-1500 \
    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/cspt/eval.jsonl \
    --prompt 'Translate the following text from Czech to Portuguese.' \
    --key_name 'instruction' \
    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/sn/greedy_decode/train_neuron/cspt/enzh_train_neuron_cos_score_0.9995/1500steps/cspt_output.csv >cspt_enzh_train_neuron_cos_score_0.9995.out 2>&1 &


python lora_model_infer.py \
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/mt/train_enzh_org/checkpoint-500 \
    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/dezh/eval.jsonl \
    --prompt 'Translate the following text from Chinese to German.' \
    --key_name 'output' \
    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/sn/greedy_decode/llama2_7b_chat_3w_1000steps_enzh/zhde_output.csv >llama2_7b_chat_3w_500steps_enzh_zhde_output.out 2>&1 &




# layer
# enzh
python lora_model_infer.py \
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/mt/layer/enzh_sft_layer1_deepspeed/checkpoint-500 \
    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/enzh/eval.jsonl \
    --prompt 'Translate the following text from English to Chinese.' \
    --key_name 'instruction' \
    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/sn/greedy_decode/layer/enzh_sft_layer1_deepspeed/500steps/enzh_output.csv >enzh_sft_layer.out 2>&1 &


python lora_model_infer.py \
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/mt/train_neuron/enzh_sft_train_neuron_picked25_in_out_progressive/checkpoint-500 \
    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/enzh/eval.jsonl \
    --prompt 'Translate the following text from English to Chinese.' \
    --key_name 'instruction' \
    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/sn/greedy_decode/train_neuron/enzh/enzh_sft_train_neuron_picked25_in_out_progressive/500steps/enzh_output.csv >enzh_sft_train_neuron_picked25_in_out_progressive.out 2>&1 &




# dezh
python lora_model_infer.py \
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/mt/layer/enzh_sft_layer4/checkpoint-2500 \
    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/dezh/eval.jsonl \
    --prompt 'Translate the following text from German to Chinese.' \
    --key_name 'instruction' \
    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/sn/greedy_decode/layer/enzh_sft_layer4/2500steps/dezh_output.csv >dezh_train_enzh_sft_layer4.out 2>&1 &





# org
# enzh
python lora_model_infer.py \
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf \
    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/enzh/eval.jsonl \
    --prompt 'Translate the following text from English to Chinese.' \
    --key_name 'instruction' \
    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/sn/greedy_decode/org/enzh_output.csv >llama2_7b_chat_org_enzh_output.out 2>&1 &

# dezh 
python lora_model_infer.py \
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf \
    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/dezh/eval.jsonl \
    --prompt 'Translate the following text from German to Chinese.' \
    --key_name 'instruction' \
    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/sn/greedy_decode/org/dezh_output.csv >llama2_7b_chat_org_dezh_output.out 2>&1 &








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


CUDA_VISIBLE_DEVICES=4,5 python lora_model_infer.py \
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/final/mt/train_neuron/enzh/3200step_150000/checkpoint-800 \
    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/enzh/eval.jsonl \
    --prompt 'Translate the following text from English to Chinese.' \
    --key_name 'instruction' \
    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/mt/train_neuron/enzh/3200step_150000/800step/enzh_output.csv >ACL_infer_output_log.out 2>&1 &



CUDA_VISIBLE_DEVICES=4,5 python lora_model_infer.py \
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/final/xnli/train_neuron/en/cos/org/1000step/100000/checkpoint-1000 \
    --validate_file_path /data/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/data/xnli/final/en_standard_prompt_test.jsonl \
    --prompt '' \
    --key_name 'input' \
    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/xnli/en/train_neuron/cos/1000step/100000/1000step/en_output.csv >output_log.out 2>&1 &

CUDA_VISIBLE_DEVICES=0,1 python lora_model_infer.py \
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/final/xnli/train_neuron/en/train_neuron_4_328_in_out_progressive/checkpoint-1000 \
    --validate_file_path /data/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/data/xnli/final/zh_zh_standard_prompt_test_prompt_zh.jsonl \
    --prompt '' \
    --key_name 'input' \
    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/xnli/en/org_en_2/500/en_output.csv >output_log.out 2>&1 &


CUDA_VISIBLE_DEVICES=4,5,6,7 python lora_model_infer.py \
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/final/xnli/train_neuron/de/probe/org/100000/checkpoint-500 \
    --validate_file_path /data/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/data/xnli/final/de_standard_prompt_test.jsonl \
    --prompt "" \
    --key_name 'input' \
    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/xnli/de/train_neuron/probe/org/100000/500step/de_output.csv







"""
