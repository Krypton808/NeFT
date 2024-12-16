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

tok_ins = "\n\n### Instruction:\n"
tok_res = "\n\n### Response:\n"
prompt_input = tok_ins + "{instruction}" + tok_res


def get_model(model_path, lora_weight_path=False, generation_config=False):
    if generation_config:
        model = transformers.AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                                                  device_map='auto')
    else:
        config = AutoConfig.from_pretrained(model_path)
        config.max_new_tokens = 512
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
        max_generate_length: int = 512,
        pad_to_max_length=True,
        save_path='temp_output.txt'
):
    print(f"loading model: {model_path}...")

    # generation_config = False

    generation_config = GenerationConfig.from_pretrained(model_path)
    # generation_config.temperature = 0.0
    generation_config.output_hidden_states = True
    # generation_config.output_attentions = True
    generation_config.return_dict_in_generate = True
    generation_config.do_sample = False
    generation_config.use_cache = False
    generation_config.no_repeat_ngram_size = 4

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

    # prompt = 'Translate the following English statements to Chinese.'
    prompt = 'Translate the following text from English to Chinese.'

    val_input = val_dataset['validation']['instruction']

    input_list = []
    for input in val_input:
        instruction = prompt + '\n' + input

        input_text = prompt_input.format_map({'instruction': instruction})

        input_list.append(input_text)

    dataloader = DataLoader(input_list, batch_size=1, shuffle=False)

    output_str_list = []

    w = open(save_path, 'w', encoding='utf-8')

    outputs = []

    for data in dataloader:
        inputs = tokenizer(data, return_tensors='pt', padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # output = model.generate(**inputs)
        output = model.generate(**inputs, **generation_config.to_dict())

        # print(output)
        # print(output)
        print(len(output))  # 2
        print(output[0].size())  # torch.Size([1, 191])

        print()
        print(len(inputs['input_ids'][0]))  # 85

        # (generated_length, hidden_layers, batch_size, input_ids_length, hidden_state)
        # print(len(output[1]))  # 106 generated length
        # print(len(output[1][0]))  # 33 hidden layers number
        # print(len(output[1][0][0]))  # 1 batch size
        # print(len(output[1][0][0][0]))  # 85 input_ids length or 1
        # print(len(output[1][0][0][0][0]))  # 4096 hidden state
        # print(output[1][0][0][0][0][0])  # tensor(0.0018, device='cuda:0', dtype=torch.bfloat16)

        print(output[1][-1][0].shape)


        count = 0
        for g in output[1]:
            print('----------------------')
            for idx, h in enumerate(g):
                print('layers: ' + str(idx+1))
                for b in h:
                    print(b)
                    print(len(b))
                    print(len(b[0]))
                    print('++++++++++++++++++++++')
                    count += 1
                    if count == 100:
                        return


        # print('+++++++++++++++++++++++')
        # out = torch.stack(output, dim=1).detach().cpu()  # We only keep the [CLS] embedding
        #
        # print(out)
        # print(out.shape())

        # return

        # outputs.append(out)
        i = 0
        for o in output[0]:
            output_str = tokenizer.decode(o, skip_special_tokens=False, spaces_between_special_tokens=False)
            output_str_list.append(output_str)

            # print(output_str.replace(data[i], '').strip())
            # w.write(output_str.replace(data[i], '').strip() + '\n')

            print(output_str.strip())
            print('*********************')
            w.write(output_str.replace(data[i], '').strip() + '\n')
            i += 1
            return


if __name__ == "__main__":
    fire.Fire(main)

"""
python infer_test.py --validate_file_path /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/enzh/eval.jsonl --save_path /data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/trans/trans_en_to_zh_version/output/output_hs_hpy.txt --model_path /mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf


"""
