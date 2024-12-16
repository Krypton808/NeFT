import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import fire
import torch
import transformers
import datasets
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoConfig, LlamaTokenizer, \
    LlamaForCausalLM
from peft import PeftModel
import csv
import numpy as np


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
        max_input_length: int = 4096,
        max_generate_length: int = 2048,
        pad_to_max_length=True,
        path_dir=''
):
    save_path = path_dir + '/en_output.csv'
    csv_file = open(save_path, 'w', newline='', encoding='utf-8')
    writer = csv.writer(csv_file)

    print(f"loading model: {model_path}...")

    # generation_config = False

    generation_config = GenerationConfig.from_pretrained(
        "/data2/haoyun/models/llm/Llama-2-7b-chat-hf")
    # generation_config.temperature = 0.0
    generation_config.output_hidden_states = True
    generation_config.return_dict_in_generate = True
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

    dataloader = DataLoader(input_list, batch_size=1, shuffle=False)

    output_str_list = []

    # w = open(save_path, 'w', encoding='utf-8')

    all_layer_all_sentence_dict = {}
    all_layer_all_sentence_dict_before_gen = {}

    for data_idx, data in enumerate(dataloader):
        inputs = tokenizer(data, return_tensors='pt', padding=True, max_length=pad_to_max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        output = model.generate(**inputs, **generation_config.to_dict())
        # print(len(output))
        i = 0
        for o in output[0]:
            # print(len(o))

            output_str = tokenizer.decode(o, skip_special_tokens=False, spaces_between_special_tokens=False)
            output_str_list.append(output_str)

            # print(output_str.strip())
            # w.write(output_str.replace(data[i], '').strip() + '\n')
            # writer.writerow([output_str.replace(data[i], '').replace('\n', '').strip()])
            writer.writerow([output_str.replace(data[i], '').replace('\n', '').strip()])
            i += 1

        # print(len(output[1]))   # 114

        # all_layer_single_sentence_dict = {}

        for i in range(33):
            hidden_states_stack_all_token_single_layer = None
            for idx, out in enumerate(output[1]):
                # merge token
                # print(out[i])
                # print(out[i].shape)
                out_temp = out[i].detach().to(torch.float).cpu().numpy()

                if hidden_states_stack_all_token_single_layer is None:
                    hidden_states_stack_all_token_single_layer = out_temp

                    hidden_states_stack_all_token_single_layer_mean_aggr_before_gen = np.mean(
                        hidden_states_stack_all_token_single_layer, axis=1)

                    print(out_temp.shape)
                else:
                    hidden_states_stack_all_token_single_layer = np.column_stack(
                        [hidden_states_stack_all_token_single_layer, out_temp])

            # print(hidden_states_stack_all_token_single_layer.shape)
            hidden_states_stack_all_token_single_layer_mean_aggr = np.mean(hidden_states_stack_all_token_single_layer,
                                                                           axis=1)  # 单句单层所有token mean
            # print(hidden_states_stack_all_token_single_layer_mean_aggr.shape) (1, 4096)

            # before_gen
            if str(i) not in all_layer_all_sentence_dict_before_gen.keys():
                all_layer_all_sentence_dict_before_gen[
                    str(i)] = hidden_states_stack_all_token_single_layer_mean_aggr_before_gen
            else:
                all_layer_all_sentence_dict_before_gen[str(i)] = np.row_stack(
                    [all_layer_all_sentence_dict_before_gen[str(i)],
                     hidden_states_stack_all_token_single_layer_mean_aggr_before_gen])

            # after_gen
            if str(i) not in all_layer_all_sentence_dict.keys():
                all_layer_all_sentence_dict[str(i)] = hidden_states_stack_all_token_single_layer_mean_aggr
            else:
                all_layer_all_sentence_dict[str(i)] = np.row_stack(
                    [all_layer_all_sentence_dict[str(i)], hidden_states_stack_all_token_single_layer_mean_aggr])

        # if data_idx == 5:
        #     break

    path_dir_before_gen = path_dir + '/before_gen/'
    for k, v in all_layer_all_sentence_dict_before_gen.items():
        # print(k)
        # print(v.shape)
        # print('***************')

        np.save(path_dir_before_gen + f'layer_{k}.npy', v)

    path_dir_after_gen = path_dir + '/after_gen/'
    for k, v in all_layer_all_sentence_dict.items():
        # print(k)
        # print(v.shape)
        # print('***************')

        np.save(path_dir_after_gen + f'layer_{k}.npy', v)

        # print(len(out))   # 33
        #
        # hidden_states_stack_single_token_all_layers = None
        #
        # for o in out:
        #     print(o.shape)  # 第一个是torch.Size([1, 69(input_size), 4096]), 之后是 torch.Size([1, 1, 4096])
        #
        #     o = o.detach().to(torch.float).cpu().numpy()
        #
        #     # hidden_layers.append(o)
        #
        #     if hidden_states_stack_single_token_all_layers is None:
        #         hidden_states_stack_single_token_all_layers = o
        #     else:
        #         hidden_states_stack_single_token_all_layers = np.row_stack([hidden_states_stack_single_token_all_layers, o])
        #
        #     print(hidden_states_stack_single_token_all_layers.shape)
        #     break
        #
        # print(hidden_states_stack_single_token_all_layers.shape)
        # if idx2 == 5:
        #     break

    # for s in output_str_list:
    #     w.write(s.replace(s.split('.')[0], '').replace(s.split('.')[1], '') + '\n')
    #     print(s)
    # print(len(output_str_list))


if __name__ == "__main__":
    fire.Fire(main)

"""


"""

"""
# org
# enzh
python model_infer_and_save_hs.py \
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/final/mt/train_neuron/hizh/750step_100000/checkpoint-750 \
    --validate_file_path /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/hizh/eval.jsonl \
    --prompt 'Translate the following text from Hindi to Chinese.' \
    --key_name 'instruction' \
    --path_dir /mnt/nfs/algo/intern/haoyunx11_/idea/MI/sn/final/mt_activations/train_hizh_750step_100000/750step/flores_101/hizh >ACL_train_hizh_750step_100000_neuron_infer_and_save.out 2>&1 &

python model_infer_and_save_hs.py \
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/final/mt/org/hizh/checkpoint-750 \
    --validate_file_path /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/hizh/eval.jsonl \
    --prompt 'Translate the following text from Hindi to Chinese.' \
    --key_name 'instruction' \
    --path_dir /mnt/nfs/algo/intern/haoyunx11_/idea/MI/sn/final/mt_activations/train_hizh_org/750step/flores_101/hizh >ACL_train_hizh_org_750step_infer_and_save.out 2>&1 &




python model_infer_and_save_hs.py \
    --model_path /data2/haoyun/models/llm/Llama-2-7b-chat-hf \
    --lora_weight_path /mnt/nfs/algo/intern/haoyunx11/models/sft/final/mt/lora/enzh/rank_128/checkpoint-8800/adapter_model \
    --validate_file_path /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/enzh/eval.jsonl \
    --prompt 'Translate the following text from English to Chinese.' \
    --key_name 'instruction' \
    --path_dir /data2/haoyun/MA_project/mt_activations/LoRA/enzh/rank_128 >infer_and_save_.out 2>&1 &

python model_infer_and_save_hs.py \
    --model_path /data2/haoyun/models/llm/Llama-2-7b-chat-hf \
    --lora_weight_path /mnt/nfs/algo/intern/haoyunx11/models/sft/final/mt/lora/enzh/rank_8/checkpoint-9600/adapter_model \
    --validate_file_path /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/frzh/eval.jsonl \
    --prompt 'Translate the following text from French to Chinese.' \
    --key_name 'instruction' \
    --path_dir /data2/haoyun/MA_project/mt_activations/overlap/frzh/enzh_200000_find_118103_ >infer_and_save.out 2>&1 &







python model_infer_and_save_hs.py \
    --model_path /data2/haoyun/models/sft/final/mt/train_neuron/frzh/pearson/3200step_100000/checkpoint-3200 \
    --validate_file_path /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/frzh/eval.jsonl \
    --prompt 'Translate the following text from French to Chinese.' \
    --key_name 'instruction' \
    --path_dir /data2/haoyun/mt_activations/train_frzh_pearson_100000/3200step/flores_101/frzh >infer_and_save.out 2>&1 &




python model_infer_and_save_hs.py \
    --model_path /home/tigerbot/test/model/final/mt/frzh/pearson/3200step_100000/checkpoint-3200 \
    --validate_file_path /home/tigerbot/test/data/mt/frzh/flores_101/eval.jsonl \
    --prompt 'Translate the following text from French to Chinese.' \
    --key_name 'instruction' \
    --path_dir /home/tigerbot/test/find/hs/final/train_frzh_pearson_100000/flores_101/frzh >infer_and_save.out 2>&1 &



python model_infer_and_save_hs.py \
    --model_path /data2/haoyun/models/sft/final/mt/org/hizh/checkpoint-750 \
    --validate_file_path /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/hizh/eval.jsonl \
    --prompt 'Translate the following text from Hindi to Chinese.' \
    --key_name 'instruction' \
    --path_dir /data2/haoyun/mt_activations/train_hizh_org/750step/flores_101/hizh >infer_and_save.out 2>&1 &




python model_infer_and_save_hs.py \
    --model_path /data2/haoyun/models/llm/Llama-2-7b-chat-hf \
    --lora_weight_path /data2/haoyun/models/sft/final/mt/lora/enzh/temp \
    --validate_file_path /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/enzh/eval.jsonl \
    --prompt 'Translate the following text from English to Chinese.' \
    --key_name 'instruction' \
    --path_dir /data2/haoyun/mt_activations/train_enzh_10w_lora_rank_64/50001step/flores_101/enzh >infer_and_save.out 2>&1 &




# xnli en
python model_infer_and_save_hs.py \
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/final/xnli/org/en_2/checkpoint-1000 \
    --validate_file_path /data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/data/xnli/final/en_standard_prompt_test.jsonl \
    --prompt '' \
    --key_name 'input' \
    --path_dir /mnt/nfs/algo/intern/haoyunx11_/idea/MI/sn/final/xnli_activations/train_en_org/1000step/test_set/en >infer_and_save.out 2>&1 &




"""
