import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import fire
import torch
import transformers
import datasets
import jsonlines
import numpy as np

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoConfig, LlamaTokenizer, \
    LlamaForCausalLM
from peft import PeftModel

tok_ins = "\n\n### Instruction:\n"
tok_res = "\n\n### Response:\n"
prompt_input = tok_ins + "{instruction}" + tok_res


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
        lora_weight_path=False,
        max_input_length: int = 512,
        max_generate_length: int = 2048,
        pad_to_max_length=True,
        # save_path='temp_output.jsonl'
):
    print(f"loading model: {model_path}...")

    # generation_config = False

    generation_config = GenerationConfig.from_pretrained(model_path)
    # generation_config.temperature = 0.0
    generation_config.output_hidden_states = True
    # generation_config.output_attentions = True
    generation_config.return_dict_in_generate = True
    generation_config.do_sample = False
    generation_config.use_cache = True
    generation_config.no_repeat_ngram_size = 4
    generation_config.max_new_tokens = 512

    model = get_model(model_path, lora_weight_path, generation_config)

    # generation_config.max_length = max_generate_length
    # print(generation_config)

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


    # Trans
    # prompt = 'Translate the following English statements to Chinese.'
    prompt = 'Translate the following text from English to Chinese.'
    # prompt = 'Translate the following text from English to German.'

    # Summary
    # Summary 数据文件中 prompt 已经加入了，但没添加 tok_ins tok_res
    # prompt = "Document：{text}\n" + "Based on the previous text, provide a brief single summary in English:"

    val_input = val_dataset['validation']['instruction']

    input_list = []
    # Trans
    for input in val_input:
        instruction = prompt + ' ' + input

        input_text = prompt_input.format_map({'instruction': instruction})

        input_list.append(input_text)

    # Sum
    # input_list = val_input

    dataloader = DataLoader(input_list, batch_size=1, shuffle=False)

    # w = open(save_path, 'w', encoding='utf-8')

    # w = jsonlines.open(save_path, 'w')

    # outputs = []

    # batch size == 1

    # path_dir = r'/mnt/nfs/algo/intern/haoyunx11_/idea/MI/trans/en_zh_use_cache_False_2/'
    # path_dir = r'/mnt/nfs/algo/intern/haoyunx11_/idea/MI/sum/en_mtiayn_3epoch/'
    # path_dir = r'/mnt/nfs/algo/intern/haoyunx11_/idea/MI/sum/zh_mtiayn_3epoch/'

    for idx1, data in enumerate(dataloader):
        sentence_number = str(idx1 + 1)

        # if int(sentence_number) < 23:
        #     continue

        if int(sentence_number) > 1000:
            break

        # path = path_dir + sentence_number
        # if not os.path.exists(path):
        #     os.makedirs(path)

        # w_txt = open(path_dir + f'{sentence_number}/output.txt', 'w', encoding='utf-8')
        inputs = tokenizer(data, return_tensors='pt', padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # output = model.generate(**inputs)
        output = model.generate(**inputs, **generation_config.to_dict())

        # use_cache = True 的情况
        try:
            for o in output[0]:
                output_str = tokenizer.decode(o, skip_special_tokens=False, spaces_between_special_tokens=False)
                print(output_str)
                # w_txt.write(output_str)

            hidden_states_stack_all = None
            print(len(output[1]))
            for idx2, out in enumerate(output[1]):
                generated_token = "generated_token_" + str(idx2 + 1)

                # hidden_layers = []
                hidden_states_stack = None
                # print(len(out))
                for idx3, o in enumerate(out):
                    # print(o.shape)
                    # 只存第33个的，第33个包括了之前所有层的
                    hidden_layer = "hidden_layer_" + str(idx3 + 1)
                    o = o.detach().to(torch.float).cpu().numpy()

                    # hidden_layers.append(o)

                    if hidden_states_stack is None:
                        hidden_states_stack = o
                    else:
                        hidden_states_stack = np.row_stack([hidden_states_stack, o])

                if hidden_states_stack_all is None:
                    hidden_states_stack_all = hidden_states_stack
                else:
                    hidden_states_stack_all = np.column_stack([hidden_states_stack_all, hidden_states_stack])

                # print(hidden_states_stack_all.shape)

                    # print(hidden_states_stack.shape)

                    # if idx3 != 32:
                    #     continue
                # if idx2 == 0:
                #     np.save(path_dir + f'{sentence_number}/{generated_token}_33.npy', hidden_states_stack_all)

            # np.save(path_dir + f'{sentence_number}/{generated_token}_33.npy', hidden_states_stack_all)

        except:
            print("expect happened" + sentence_number)
            continue

        # use_cache = False 的情况
        # for o in output[0]:
        #     output_str = tokenizer.decode(o, skip_special_tokens=False, spaces_between_special_tokens=False)
        #     w_txt.write(output_str)
        #
        # hidden_layers = []
        #
        # for idx3, o in enumerate(output[1][0]):
        #     hidden_layer = "hidden_layer_" + str(idx3 + 1)
        #     o = o.detach().to(torch.float).cpu().numpy()
        #     hidden_layers.append(o)
        #
        #     # 只存第33个的，第33个包括了之前所有层的
        #     if idx3 != 32:
        #         continue
        #     np.save(path_dir + f'{sentence_number}/1_{hidden_layer}.npy', hidden_layers)
        #
        # hidden_layers = []
        # generated_token = str(len(output[1]))
        # for idx3, o in enumerate(output[1][-1]):
        #     hidden_layer = "hidden_layer_" + str(idx3 + 1)
        #     o = o.detach().to(torch.float).cpu().numpy()
        #     hidden_layers.append(o)
        #
        #     # 只存第33个的，第33个包括了之前所有层的
        #     if idx3 != 32:
        #         continue
        #     np.save(path_dir + f'{sentence_number}/{generated_token}_{hidden_layer}.npy', hidden_layers)


if __name__ == "__main__":
    fire.Fire(main)

"""
python infer.py --validate_file_path /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/enzh/eval.jsonl --save_path /data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/trans/trans_en_to_zh_version/output/output_hs_hpy.jsonl --model_path /mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf

nohup python infer.py --validate_file_path /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/enzh/eval.jsonl --save_path /data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/trans/trans_en_to_zh_version/output/output_hs_hpy.jsonl --model_path /mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf >save_hidden_state.out 2>&1 &


"""
