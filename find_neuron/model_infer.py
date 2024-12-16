import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

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
        max_generate_length: int = 512,
        pad_to_max_length=True,
        save_path='temp_output.csv'
):
    csv_file = open(save_path, 'w', newline='', encoding='utf-8')
    writer = csv.writer(csv_file)

    print(f"loading model: {model_path}...")

    # generation_config = False

    generation_config = GenerationConfig.from_pretrained(
        "/mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf")
    # generation_config.temperature = 0.0
    # generation_config.output_hidden_states = True
    # generation_config.output_attentions = True
    # generation_config.return_dict_in_generate = True
    generation_config.do_sample = False
    generation_config.use_cache = True
    generation_config.no_repeat_ngram_size = 4
    generation_config.max_new_tokens = 512

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

    dataloader = DataLoader(input_list, batch_size=8, shuffle=False)

    output_str_list = []

    # w = open(save_path, 'w', encoding='utf-8')

    for data in dataloader:
        inputs = tokenizer(data, return_tensors='pt', padding=True, max_length=pad_to_max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        output = model.generate(**inputs, **generation_config.to_dict())
        print(len(output))
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

# enzh
python lora_model_infer.py \
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/mt/train_enzh_org/checkpoint-2500 \
    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/enzh/eval.jsonl \
    --prompt 'Translate the following text from English to Chinese.' \
    --key_name 'instruction' \
    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/sn/greedy_decode/llama2_7b_chat_3w_2500steps_enzh/enzh_output.csv >train_enzh_org_enzh_2500steps.out 2>&1 &


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
python lora_model_infer.py \
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/mt/train_neuron/dezh_train_neuron_all_layer_328_in_out_progressive/checkpoint-2000 \
    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/dezh/eval.jsonl \
    --prompt 'Translate the following text from German to Chinese.' \
    --key_name 'instruction' \
    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/sn/greedy_decode/dezh/dezh_train_neuron_all_layers_328_in_out_progressive/2000steps/dezh_output.csv >dezh_train_neuron_all_layers_328_in_out_progressive_2000steps_dezh.out 2>&1 &


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
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/mt/train_neuron/enzh_train_neuron_4_328_layer4_in_out_progressive/checkpoint-2500 \
    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/enzh/eval.jsonl \
    --prompt 'Translate the following text from English to Chinese.' \
    --key_name 'instruction' \
    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/sn/greedy_decode/train_neuron/enzh/enzh_train_neuron_4_328_layer4_in_out_progressive/2500steps/enzh_output.csv >enzh_train_neuron_4_328_layer4_in_out_progressive.out 2>&1 &


python lora_model_infer.py \
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/mt/train_enzh_org/checkpoint-500 \
    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/enzh/eval.jsonl \
    --prompt 'Translate the following text from Chinese to English.' \
    --key_name 'output' \
    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/sn/greedy_decode/llama2_7b_chat_3w_500steps_enzh/zhen_output.csv >llama2_7b_chat_3w_500steps_enzh_zhen_output.out 2>&1 &

# ende
python lora_model_infer.py \
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/mt/train_enzh_org/checkpoint-500 \
    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/ende/eval.jsonl \
    --prompt 'Translate the following text from English to German.' \
    --key_name 'instruction' \
    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/sn/llama2_7b_chat_3w_500steps_enzh/ende_output.csv >llama2_7b_chat_3w_500steps_enzh_ende_output.out 2>&1 &


python lora_model_infer.py \
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/mt/train_enzh_org/checkpoint-500 \
    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/ende/eval.jsonl \
    --prompt 'Translate the following text from German to English.' \
    --key_name 'output' \
    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/sn/llama2_7b_chat_3w_500steps_enzh/deen_output.csv >llama2_7b_chat_3w_500steps_enzh_deen_output.out 2>&1 &


# dezh
python lora_model_infer.py \
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/mt/train_dezh_org/checkpoint-2500 \
    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/dezh/eval.jsonl \
    --prompt 'Translate the following text from German to Chinese.' \
    --key_name 'instruction' \
    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/sn/greedy_decode/dezh/train_dezh_org/2500steps/dezh_output.csv >train_dezh_org.out 2>&1 &


python lora_model_infer.py \
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/mt/train_enzh_org/checkpoint-500 \
    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/dezh/eval.jsonl \
    --prompt 'Translate the following text from Chinese to German.' \
    --key_name 'output' \
    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/sn/greedy_decode/llama2_7b_chat_3w_1000steps_enzh/zhde_output.csv >llama2_7b_chat_3w_500steps_enzh_zhde_output.out 2>&1 &

# cspt
python lora_model_infer.py \
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/mt/train_cspt_org/checkpoint-2500 \
    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/cspt/eval.jsonl \
    --prompt 'Translate the following text from Czech to Portuguese.' \
    --key_name 'instruction' \
    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/sn/greedy_decode/cspt/train_cspt_org/2500steps/cspt_output.csv >train_cspt_org.out 2>&1 &


python lora_model_infer.py \
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/mt/train_enzh_org/checkpoint-500 \
    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/dezh/eval.jsonl \
    --prompt 'Translate the following text from Chinese to German.' \
    --key_name 'output' \
    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/sn/greedy_decode/llama2_7b_chat_3w_1000steps_enzh/zhde_output.csv >llama2_7b_chat_3w_500steps_enzh_zhde_output.out 2>&1 &




# layer
# enzh
python lora_model_infer.py \
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/mt/layer/enzh_sft_layer0_deepspeed/checkpoint-2500 \
    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/enzh/eval.jsonl \
    --prompt 'Translate the following text from English to Chinese.' \
    --key_name 'instruction' \
    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/sn/greedy_decode/layer/enzh_sft_layer0_deepspeed/2500steps/enzh_output.csv >enzh_sft_layer0_deepspeed.out 2>&1 &

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


python model_infer.py \
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/mt/layer/enzh_sft_layer8_deepspeed/checkpoint-2500 \
    --validate_file_path /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/enzh/eval.jsonl \
    --prompt 'Translate the following text from English to Chinese.' \
    --key_name 'instruction' \
    --save_path /mnt/nfs/algo/intern/haoyunx11_/idea/MI/sn/mt_activations/layer/enzh_sft_layer8_deepspeed/2500steps/flores101/enzh/enzh_output_infer_2.csv >train_enzh_org_enzh_2500steps_infer.out 2>&1 &


"""
