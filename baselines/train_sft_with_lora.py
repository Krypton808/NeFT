# import sys
# sys.path.append(r'/data/haoyun.xu/pkg/transformers/src/transformers')
# sys.path.insert(0, r'/data/haoyun.xu/pkg/transformers/src/transformers')

import os
# from typing import no_type_check
# from torch.distributed.fsdp.flat_param import FlatParamHandle
# from torch.distributed.fsdp import _runtime_utils

# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4"
os.environ["FSDP_USE_ORIG_PARAMS"] = "true"
from dataclasses import dataclass, field
from typing import Optional
from peft import get_peft_model, LoraConfig
import datasets
import evaluate
import torch
import transformers

# from trainer_add_grad_mask import Trainer as Trainer_add_grad_mask
# from trainer_add_grad_mask_several_sn import Trainer as Trainer_add_grad_mask
# from trainer_add_grad_mask_load_neuron_dict import Trainer as Trainer_add_grad_mask

# from trainer_add_grad_mask_load_neuron_dict_add_gate import Trainer as Trainer_add_grad_mask
# from trainer_add_grad_mask_load_neuron_dict_offload import Trainer as Trainer_add_grad_mask
# from trainer_add_grad_mask_load_neuron_dict_summon_full_param import Trainer as Trainer_add_grad_mask

os.environ["WANDB_DISABLED"] = "true"

@dataclass
class SFTConfig:
    model_name_or_path: Optional[str] = field(metadata={"help": "Path to pretrained model checkpoint"})
    dataset_name: Optional[str] = field(default=None, metadata={"help": "Huggingface dataset name"})
    train_file_path: Optional[str] = field(default=None, metadata={"help": "Path to train data file/directory"})
    validate_file_path: Optional[str] = field(default=None, metadata={"help": "Path to validation data file/directory"})
    max_length: int = field(default=4096, metadata={"help": "Max length of input"})
    text_key_name: Optional[str] = field(default="content",
                                         metadata={"help": "key to text field name in train and validation file"})
    preprocess_num_workers: int = field(default=8,
                                        metadata={"help": "The number of processes to use for the preprocessing."})
    # prompt: str = field(default='')
    # mask_dict_path: str = field(default='')


def check_file_exist(path: str):
    if not os.path.exists(path):
        raise ValueError(f"Path: {path} not exists!")


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    metric = evaluate.load("/mnt/nfs/algo/intern/haoyunx9/evaluate_metric/accuracy")
    return metric.compute(predictions=preds, references=labels)


def main():
    transformers.set_seed(1234)
    parser = transformers.HfArgumentParser((SFTConfig, transformers.TrainingArguments))
    sft_config, training_args = parser.parse_args_into_dataclasses()

    print(training_args.report_to)
    # print(sft_config.prompt)

    # check file existence
    if sft_config.dataset_name is None and sft_config.train_file_path is None:
        raise ValueError(f"One of --dataset_name or --train_file_path must be set")
    if sft_config.train_file_path:
        check_file_exist(sft_config.train_file_path)
    if sft_config.validate_file_path:
        check_file_exist(sft_config.validate_file_path)

    # load model, tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(sft_config.model_name_or_path, padding_side='right',
                                                           trunction_side="right",
                                                           max_length=sft_config.max_length)
    tokenizer.pad_token = tokenizer.eos_token

    model = transformers.LlamaForCausalLM.from_pretrained(sft_config.model_name_or_path)
    peft_config = LoraConfig(
        task_type='CAUSAL_LM',
        r=8,
        lora_alpha=16,
        lora_dropout=0.0
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    if sft_config.dataset_name:
        ds = datasets.load_dataset(sft_config.dataset_name)
        train_ds, validation_ds = ds['train'], ds['validation']
        raw_datasets = datasets.DatasetDict({"train": train_ds, "validation": validation_ds})
    else:
        # Split 20% of train data as validation data
        if not sft_config.validate_file_path:
            train_ds, validation_ds = datasets.load_dataset('json', data_files=sft_config.train_file_path,
                                                            split=['train[:80%]', 'train[80%:]'])
            raw_datasets = datasets.DatasetDict({"train": train_ds, "validation": validation_ds})
        else:
            raw_datasets = datasets.load_dataset("json", data_files={'train': sft_config.train_file_path,
                                                                     'validation': sft_config.validate_file_path})

    def process_supervised(record):
        # MT

        prompt = 'Translate the following text from English to Chinese.'
        # prompt = sft_config.prompt
        input_s = prompt + ' ' + record['instruction']
        output_s = record['output']

        # Summary
        # prompt = "Given the below Hindi article, generate a summary in French." + "\nArticle: "
        # input_s = prompt + record['text']
        # output_s = record['summary']

        # Xnli
        # input_s = record['input']
        # output_s = record['output']

        # input_s = record['input']
        tokenized = tokenizer([input_s, output_s])
        token_ids = [tok_id for tok_ids in tokenized['input_ids'] for tok_id in tok_ids]
        attention_mask = [mask for masks in tokenized['attention_mask'] for mask in masks]
        if token_ids[-1] != tokenizer.eos_token_id:
            token_ids += [tokenizer.eos_token_id]
            attention_mask += [1]
        processed_record = {
            "input_ids": token_ids[:sft_config.max_length],
            "attention_mask": attention_mask[:sft_config.max_length],
            "labels": token_ids.copy()[:sft_config.max_length]
        }
        # ignore input label, label is ignored if value is -100
        processed_record["labels"][:min(len(tokenized["input_ids"][0]), sft_config.max_length)] = [-100] * min(
            len(tokenized["input_ids"][0]), sft_config.max_length)
        return {k: torch.tensor(v, dtype=torch.int) for k, v in processed_record.items()}

    with training_args.main_process_first(desc="Process supervised dataset"):
        sft_dataset = raw_datasets.map(
            process_supervised,
            batched=False,
            # num_proc=sft_config.preprocess_num_workers,
            remove_columns=raw_datasets["train"].column_names,
            desc="Process supervised dataset"
        )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=sft_dataset["train"],
        eval_dataset=sft_dataset["validation"],
        tokenizer=tokenizer,  # trainer need tokenizer.pad_token_id,
        data_collator=transformers.DataCollatorForTokenClassification(tokenizer=tokenizer, padding="longest",
                                                                      max_length=sft_config.max_length,
                                                                      label_pad_token_id=-100),
        compute_metrics=compute_metrics if training_args.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        # mask_dict=sft_config.mask_dict_path
    )

    # trigger Training
    trainer.train()
    trainer.save_model()
    # trainer.save_state()


if __name__ == '__main__':
    main()

"""
nohup deepspeed \
--include="localhost:6,7" \
./train_sft.py \
--deepspeed ./ds_config/ds_config_zero1.json \
--model_name_or_path /mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf \
--train_file_path /data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/data/xnli/en_standard_prompt_3w.jsonl \
--validate_file_path /data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/data/xnli/en_standard_prompt_last1k_eval.jsonl \
--do_train \
--output_dir /mnt/nfs/algo/intern/haoyunx11/models/sft/final/xnli/train_neuron/en/train_neuron_4_328_in_out_progressive \
--overwrite_output_dir \
--preprocess_num_workers 8 \
--num_train_epochs 4 \
--learning_rate 2e-5 \
--evaluation_strategy steps \
--eval_steps 500 \
--bf16 True \
--save_strategy steps \
--save_steps 500 \
--logging_steps 10 \
--tf32 True \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--save_total_limit 5 \
--report_to wandb >ACL_sft_xnli_en_train_neuron_4_328_in_out_progressive.out 2>&1 &


nohup deepspeed \
--master_port=25640 \
--include="localhost:0" \
./train_sft.py \
--deepspeed ./ds_config/ds_config_zero1.json \
--model_name_or_path /mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf \
--train_file_path /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/news/train_3w.jsonl \
--validate_file_path /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/enzh/eval.jsonl \
--do_train \
--output_dir /mnt/nfs/algo/intern/haoyunx11/models/sft/mt/layer/enzh_sft_layer14_deepspeed \
--overwrite_output_dir \
--preprocess_num_workers 8 \
--num_train_epochs 1 \
--learning_rate 2e-5 \
--evaluation_strategy steps \
--eval_steps 500 \
--bf16 True \
--save_strategy steps \
--save_steps 500 \
--logging_steps 10 \
--tf32 True \
--per_device_train_batch_size 12 \
--per_device_eval_batch_size 12 \
--save_total_limit 5 \
--report_to wandb >enzh_sft_layer14_deepspeed.out 2>&1 &



python train_sft.py \
--model_name_or_path /mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf \
--train_file_path /data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/data/xnli/en_standard_prompt_3w.jsonl \
--validate_file_path /data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/data/xnli/en_standard_prompt_last1k_eval.jsonl \
--do_train \
--output_dir /mnt/nfs/algo/intern/haoyunx11/models/sft/xnli/train_neuron/en_test \
--overwrite_output_dir \
--preprocess_num_workers 8 \
--num_train_epochs 1 \
--learning_rate 2e-5 \
--evaluation_strategy steps \
--eval_steps 500 \
--bf16 True \
--save_strategy steps \
--save_steps 500 \
--logging_steps 10 \
--tf32 True \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--save_total_limit 5


for fsdp

torchrun --nproc_per_node=3 --master_port=25645 train_sft.py \
--model_name_or_path /mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf \
--train_file_path /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/news/train_3w.jsonl \
--validate_file_path /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/enzh/eval.jsonl \
--do_train \
--output_dir /mnt/nfs/algo/intern/haoyunx11/models/sft/mt/train_neuron/enzh_train_neuron_cos_score_0.9995 \
--overwrite_output_dir \
--preprocess_num_workers 8 \
--num_train_epochs 1 \
--learning_rate 2e-5 \
--evaluation_strategy steps \
--eval_steps 500 \
--bf16 True \
--save_strategy steps \
--save_steps 500 \
--logging_steps 10 \
--tf32 True \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--save_total_limit 5 \
--fsdp "full_shard" \
--fsdp_config ./fsdp_config/fsdp_config.json \
--report_to wandb >enzh_train_neuron_cos_score_0.9995.out 2>&1 &


torchrun --nproc_per_node=4 --master_port=25645 train_sft.py \
--model_name_or_path /mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf \
--train_file_path /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/news/enfr/train_2w_.jsonl \
--validate_file_path /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/news/enfr/dev_1k_.jsonl \
--do_train \
--output_dir /mnt/nfs/algo/intern/haoyunx11/models/sft/mt/train_enfr_org \
--overwrite_output_dir \
--preprocess_num_workers 8 \
--num_train_epochs 4 \
--learning_rate 2e-5 \
--evaluation_strategy steps \
--eval_steps 100 \
--bf16 True \
--save_strategy steps \
--save_steps 1250 \
--logging_steps 10 \
--tf32 True \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--save_total_limit 20 \
--fsdp "full_shard" \
--fsdp_config ./fsdp_config/fsdp_config.json \
--report_to wandb >train_enfr_org.out 2>&1 &


# clean-frost-22
torchrun --nproc_per_node=4 --master_port=25646 train_sft.py \
--model_name_or_path /mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf \
--train_file_path /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/news/train_2w.jsonl \
--validate_file_path /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/news/dev_1k.jsonl \
--do_train \
--output_dir /mnt/nfs/algo/intern/haoyunx11/models/sft/mt/train_enzh_org \
--overwrite_output_dir \
--preprocess_num_workers 8 \
--num_train_epochs 4 \
--learning_rate 2e-5 \
--evaluation_strategy steps \
--eval_steps 100 \
--bf16 True \
--save_strategy steps \
--save_steps 400 \
--logging_steps 10 \
--tf32 True \
--per_device_train_batch_size 3 \
--per_device_eval_batch_size 3 \
--save_total_limit 20 \
--fsdp "full_shard" \
--fsdp_config ./fsdp_config/fsdp_config.json \
--load_best_model_at_end \
--report_to wandb >train_enzh_org.out 2>&1 &


torchrun --nproc_per_node=3 --master_port=25645 train_sft.py \
--model_name_or_path /mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf \
--train_file_path /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/news/train_2w.jsonl \
--validate_file_path /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/news/dev_1k.jsonl \
--do_train \
--output_dir /mnt/nfs/algo/intern/haoyunx11/models/sft/mt/train_neuron/temp \
--overwrite_output_dir \
--preprocess_num_workers 8 \
--num_train_epochs 4 \
--learning_rate 2e-5 \
--evaluation_strategy steps \
--eval_steps 100 \
--bf16 True \
--save_strategy steps \
--save_steps 10 \
--logging_steps 10 \
--tf32 True \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--save_total_limit 20 \
--fsdp "full_shard" \
--fsdp_config ./fsdp_config/fsdp_config.json \
--report_to wandb >train_neuron_enzh_10w.out 2>&1 &

torchrun --nproc_per_node=3 --master_port=25645 train_sft.py \
--model_name_or_path /mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf \
--train_file_path /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/news/train_2w.jsonl \
--validate_file_path /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/news/dev_1k.jsonl \
--do_train \
--output_dir /mnt/nfs/algo/intern/haoyunx11/models/sft/final/mt/org/enzh \
--overwrite_output_dir \
--preprocess_num_workers 8 \
--num_train_epochs 4 \
--learning_rate 2e-5 \
--evaluation_strategy steps \
--eval_steps 100 \
--bf16 True \
--save_strategy steps \
--save_steps 800 \
--logging_steps 10 \
--tf32 True \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--save_total_limit 20 \
--fsdp "full_shard" \
--fsdp_config ./fsdp_config/fsdp_config.json \
--report_to wandb >ACL_train_enzh_org.out 2>&1 &







torchrun --nproc_per_node=4 --master_port=25620 train_sft.py \
--model_name_or_path /mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf \
--train_file_path /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/news/frzh/train_2w.jsonl \
--validate_file_path /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/news/frzh/dev_1k.jsonl \
--do_train \
--output_dir /mnt/nfs/algo/intern/haoyunx11/models/sft/mt/train_frzh_org \
--overwrite_output_dir \
--preprocess_num_workers 8 \
--num_train_epochs 4 \
--learning_rate 2e-5 \
--evaluation_strategy steps \
--eval_steps 100 \
--bf16 True \
--save_strategy steps \
--save_steps 1250 \
--logging_steps 10 \
--tf32 True \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--save_total_limit 20 \
--fsdp "full_shard" \
--fsdp_config ./fsdp_config/fsdp_config.json \
--report_to wandb >train_frzh_org.out 2>&1 &


torchrun --nproc_per_node=3 --master_port=25647 train_sft.py \
--model_name_or_path /mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf \
--train_file_path /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/news/hien/train.jsonl \
--validate_file_path /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/news/hien/dev_558.jsonl \
--do_train \
--output_dir /mnt/nfs/algo/intern/haoyunx11/models/sft/mt/train_neuron/train_neuron_hien_10w \
--overwrite_output_dir \
--preprocess_num_workers 8 \
--num_train_epochs 4 \
--learning_rate 2e-5 \
--evaluation_strategy steps \
--eval_steps 100 \
--bf16 True \
--save_strategy epoch \
--logging_steps 10 \
--tf32 True \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--save_total_limit 20 \
--fsdp "full_shard" \
--fsdp_config ./fsdp_config/fsdp_config.json \
--report_to wandb >train_neuron_hien_10w.out 2>&1 &

# NO_SHARD full_shard


torchrun --nproc_per_node=3 --master_port=25647 train_sft.py \
--model_name_or_path /mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf \
--train_file_path /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/news/hien/train.jsonl \
--validate_file_path /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/news/hien/dev_558.jsonl \
--do_train \
--output_dir /mnt/nfs/algo/intern/haoyunx11/models/sft/mt/train_hien_org \
--overwrite_output_dir \
--preprocess_num_workers 8 \
--num_train_epochs 4 \
--learning_rate 2e-5 \
--evaluation_strategy steps \
--eval_steps 100 \
--bf16 True \
--save_strategy epoch \
--logging_steps 10 \
--tf32 True \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--save_total_limit 20 \
--fsdp "full_shard" \
--fsdp_config ./fsdp_config/fsdp_config.json \
--report_to wandb >train_hien_org.out 2>&1 &




torchrun --nproc_per_node=4 --master_port=25648 train_sft.py \
--model_name_or_path /mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf \
--train_file_path /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/news/hizh/train_clean.jsonl \
--validate_file_path /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/news/hizh/dev_435.jsonl \
--do_train \
--output_dir /mnt/nfs/algo/intern/haoyunx11/models/sft/mt/train_hizh_org \
--overwrite_output_dir \
--preprocess_num_workers 8 \
--num_train_epochs 4 \
--learning_rate 2e-5 \
--evaluation_strategy steps \
--eval_steps 100 \
--bf16 True \
--save_strategy epoch \
--logging_steps 10 \
--tf32 True \
--per_device_train_batch_size 3 \
--per_device_eval_batch_size 3 \
--save_total_limit 20 \
--fsdp "full_shard" \
--fsdp_config ./fsdp_config/fsdp_config.json \
--report_to wandb >train_hizh_org.out 2>&1 &


torchrun --nproc_per_node=4 --master_port=25641 train_sft.py \
--model_name_or_path /mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf \
--train_file_path /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/news/hifr/train.jsonl \
--validate_file_path /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/news/hifr/dev.jsonl \
--do_train \
--output_dir /mnt/nfs/algo/intern/haoyunx11/models/sft/mt/train_hifr_org \
--overwrite_output_dir \
--preprocess_num_workers 8 \
--num_train_epochs 4 \
--learning_rate 2e-5 \
--evaluation_strategy steps \
--eval_steps 100 \
--bf16 True \
--save_strategy epoch \
--logging_steps 10 \
--tf32 True \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--save_total_limit 20 \
--fsdp "full_shard" \
--fsdp_config ./fsdp_config/fsdp_config.json \
--report_to wandb >train_hifr_org.out 2>&1 &





torchrun --nproc_per_node=3 --master_port=25640 train_sft.py \
--model_name_or_path /mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf \
--train_file_path /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/news/ende/train_3w.jsonl \
--validate_file_path /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/ende/eval.jsonl \
--do_train \
--output_dir /mnt/nfs/algo/intern/haoyunx11/models/sft/mt/train_ende_org \
--overwrite_output_dir \
--num_train_epochs 1 \
--learning_rate 2e-5 \
--evaluation_strategy steps \
--eval_steps 500 \
--bf16 True \
--save_strategy steps \
--save_steps 500 \
--logging_steps 10 \
--tf32 True \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--save_total_limit 5 \
--fsdp "full_shard" \
--fsdp_config ./fsdp_config/fsdp_config.json \
--report_to wandb >train_ende_org_fsdp.out 2>&1 &




torchrun --nproc_per_node=1 --master_port=25648 train_sft.py \
--model_name_or_path /mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf \
--train_file_path /data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/data/CrossSum/english-french/4096_cut_english-french_train.jsonl \
--validate_file_path /data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/data/CrossSum/english-french/english-french_test.jsonl \
--do_train \
--output_dir /mnt/nfs/algo/intern/haoyunx11/models/sft/summary/train_neuron/enfr/summary_enfr_train_neuron_cos_score_0.9999153 \
--overwrite_output_dir \
--num_train_epochs 1 \
--learning_rate 2e-5 \
--evaluation_strategy steps \
--eval_steps 250 \
--bf16 True \
--save_strategy steps \
--save_steps 250 \
--logging_steps 10 \
--tf32 True \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--save_total_limit 10 \
--fsdp "full_shard auto_wrap offload" \
--fsdp_config ./fsdp_config/fsdp_config.json \
--report_to wandb >summary_enfr_train_neuron_cos_score_0.9999153.out 2>&1 &



torchrun --nproc_per_node=2 --master_port=25647 train_sft.py \
--model_name_or_path /mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf \
--train_file_path /data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/data/CrossSum/english-chinese_simplified/4096_cut_english-chinese_simplified_train.jsonl \
--validate_file_path /data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/data/CrossSum/chinese_simplified-french/4096_cut_chinese_simplified-french_val.jsonl \
--do_train \
--output_dir /mnt/nfs/algo/intern/haoyunx11/models/sft/summary/enfr/train_enfr_org \
--overwrite_output_dir \
--num_train_epochs 4 \
--learning_rate 2e-5 \
--evaluation_strategy steps \
--eval_steps 100 \
--bf16 True \
--save_strategy epoch \
--logging_steps 10 \
--tf32 True \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--save_total_limit 10 \
--fsdp "full_shard" \
--fsdp_config ./fsdp_config/fsdp_config.json \
--report_to wandb >summary_train_enzh_org.out 2>&1 &

torchrun --nproc_per_node=2 --master_port=25646 train_sft.py \
--model_name_or_path /mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf \
--train_file_path /data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/data/CrossSum/english-chinese_simplified/4096_cut_english-chinese_simplified_train.jsonl \
--validate_file_path /data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/data/CrossSum/english-chinese_simplified/4096_cut_english-chinese_simplified_test.jsonl \
--do_train \
--output_dir /mnt/nfs/algo/intern/haoyunx11/models/sft/summary/train_neuron/enzh/summary_enzh_train_neuron_cos_score_0.9995_mt \
--overwrite_output_dir \
--num_train_epochs 1 \
--learning_rate 2e-5 \
--evaluation_strategy steps \
--eval_steps 250 \
--bf16 True \
--save_strategy steps \
--save_steps 250 \
--logging_steps 10 \
--tf32 True \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--save_total_limit 10 \
--fsdp "full_shard" \
--fsdp_config ./fsdp_config/fsdp_config.json \
--report_to wandb >summary_enzh_train_neuron_cos_score_0.9995_mt.out 2>&1 &




full_shard auto_wrap offload"
"""
