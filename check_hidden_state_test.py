import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import fire
import torch
import transformers
import datasets
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoConfig, LlamaTokenizer, LlamaForCausalLM
from peft import PeftModel


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
    generation_config.output_hidden_states = True
    generation_config.return_dict_in_generate = True

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

    if tokenizer.model_max_length is None or tokenizer.model_max_length > max_generate_length:
        tokenizer.model_max_length = max_generate_length

    val_dataset = datasets.load_dataset("json", data_files={'validation': validate_file_path})

    prompt = 'Translate the following text from English to Russian.'

    val_input = val_dataset['validation']['instruction']

    input_list = []
    for input in val_input:
        input_list.append(prompt + ' ' + input)

    dataloader = DataLoader(input_list, batch_size=8, shuffle=False)

    output_str_list = []

    w = open(save_path, 'w', encoding='utf-8')

    for data in dataloader:
        inputs = tokenizer(data, return_tensors='pt', padding=True, max_length=pad_to_max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        output = model.generate(**inputs)
        # output = model.generate(**inputs, **generation_config.to_dict())
        i = 0
        for o in output:
            output_str = tokenizer.decode(o, skip_special_tokens=True, spaces_between_special_tokens=False)
            output_str_list.append(output_str)

            print(output_str.replace(data[i], '').strip())
            w.write(output_str.replace(data[i], '').strip() + '\n')
            i += 1

    # for s in output_str_list:
    #     w.write(s.replace(s.split('.')[0], '').replace(s.split('.')[1], '') + '\n')
    #     print(s)
    # print(len(output_str_list))


if __name__ == "__main__":
    fire.Fire(main)
