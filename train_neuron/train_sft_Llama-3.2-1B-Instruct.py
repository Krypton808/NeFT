import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
os.environ["FSDP_USE_ORIG_PARAMS"] = "true"
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import torch
import transformers

# from trainer_add_grad_mask_load_neuron_dict_Llama3_2_1B_Instruct import Trainer as Trainer_add_grad_mask

from trainer_add_grad_mask_load_neuron_dict_Mistral_7B_v1 import Trainer as Trainer_add_grad_mask

os.environ["WANDB_DISABLED"] = "true"


@dataclass
class SFTConfig:
    model_name_or_path: Optional[str] = field(metadata={"help": "Path to pretrained model checkpoint"})
    dataset_name: Optional[str] = field(default=None, metadata={"help": "Huggingface dataset name"})
    train_file_path: Optional[str] = field(default=None, metadata={"help": "Path to train data file/directory"})
    validate_file_path: Optional[str] = field(default=None, metadata={"help": "Path to validation data file/directory"})
    max_length: int = field(default=8192, metadata={"help": "Max length of input"})
    text_key_name: Optional[str] = field(default="content",
                                         metadata={"help": "key to text field name in train and validation file"})
    preprocess_num_workers: int = field(default=8,
                                        metadata={"help": "The number of processes to use for the preprocessing."})


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
    metric = evaluate.load("/mnt/data1/utils/evaluate_metric/accuracy")
    return metric.compute(predictions=preds, references=labels)


def main():
    transformers.set_seed(1234)
    parser = transformers.HfArgumentParser((SFTConfig, transformers.TrainingArguments))
    sft_config, training_args = parser.parse_args_into_dataclasses()

    print(training_args.report_to)

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

    for k, v in model.named_parameters():
        print(k)
        if 'up_proj' in k or 'down_proj' in k:
            v.requires_grad = True
        else:
            v.requires_grad = False

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
        # Llama-3.2-1B-Instruct
        # Alpaca
        input_s = record['input']
        output_s = record['output']

        tokenized = tokenizer([input_s, output_s], add_special_tokens=False)
        token_ids = [tok_id for tok_ids in tokenized['input_ids'] for tok_id in tok_ids]
        attention_mask = [mask for masks in tokenized['attention_mask'] for mask in masks]
        # print(token_ids)
        # print(attention_mask)

        # if token_ids[-1] != tokenizer.eos_token_id:
        #     token_ids += [tokenizer.eos_token_id]
        #     attention_mask += [1]

        processed_record = {
            "input_ids": token_ids[:sft_config.max_length],
            "attention_mask": attention_mask[:sft_config.max_length],
            "labels": token_ids.copy()[:sft_config.max_length]
        }
        # ignore input label, label is ignored if value is -100
        processed_record["labels"][:min(len(tokenized["input_ids"][0]), sft_config.max_length)] = [-100] * min(
            len(tokenized["input_ids"][0]), sft_config.max_length)

        # print(processed_record["labels"])
        # print('***************************')
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
        preprocess_logits_for_metrics=preprocess_logits_for_metrics

    )

    # trigger Training
    trainer.train()
    trainer.save_model()


if __name__ == '__main__':
    transformers.Trainer = Trainer_add_grad_mask

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





"""

# llama3_2_1B 
torchrun --nproc_per_node=1 --master_port=25646 train_sft_Llama-3.2-1B-Instruct.py \
--model_name_or_path /mnt/data1/models/Llama-3.2-1B \
--train_file_path /mnt/data1/study/data/alpaca/jsonl/split_sys_prompt/alpaca_split_sys_prompt.jsonl \
--validate_file_path /mnt/data1/study/data/alpaca_gpt4/jsonl/split_sys_prompt/alpaca_gpt4_split_sys_prompt_1k.jsonl \
--do_train \
--output_dir /mnt/data1/output/model/llama3_2_1B_alpaca \
--overwrite_output_dir \
--num_train_epochs 5 \
--learning_rate 2e-5 \
--evaluation_strategy steps \
--eval_steps 1300 \
--bf16 True \
--save_strategy steps \
--save_steps 1300 \
--logging_steps 10 \
--tf32 True \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--save_total_limit 50 \
--fsdp "full_shard" \
--fsdp_config ./fsdp_config/fsdp_config.json >llama3_2_1B_alpaca.out 2>&1 &


# neuron mask
torchrun --nproc_per_node=1 --master_port=25647 train_sft_Llama-3.2-1B-Instruct.py \
--model_name_or_path /mnt/data1/models/Llama-3.2-1B \
--train_file_path /mnt/data1/study/data/alpaca/jsonl/split_sys_prompt/alpaca_split_sys_prompt_2.jsonl \
--validate_file_path /mnt/data1/study/data/alpaca_gpt4/jsonl/split_sys_prompt/alpaca_gpt4_split_sys_prompt_1k_2.jsonl \
--do_train \
--output_dir /mnt/data1/output/model/llama3_2_1B_alpaca_neuron_6000_2 \
--overwrite_output_dir \
--num_train_epochs 5 \
--learning_rate 2e-5 \
--evaluation_strategy steps \
--eval_steps 1300 \
--bf16 True \
--save_strategy steps \
--save_steps 1300 \
--logging_steps 10 \
--tf32 True \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--save_total_limit 50 \
--fsdp "full_shard" \
--fsdp_config ./fsdp_config/fsdp_config.json >llama3_2_1B_alpaca_neuron_6000_2.out 2>&1 &



# Mistral_7B_v3
torchrun --nproc_per_node=4 --master_port=25647 train_sft_Llama-3.2-1B-Instruct.py \
--model_name_or_path /mnt/data1/models/Mistral-7B-v0.1 \
--train_file_path /mnt/data1/study/data/alpaca/jsonl/split_sys_prompt/Mistral_7B_Instruct_v3_alpaca_split_sys_prompt.jsonl \
--validate_file_path /mnt/data1/study/data/alpaca_gpt4/jsonl/split_sys_prompt/Mistral_7B_Instruct_v3_alpaca_gpt4_split_sys_prompt_1k.jsonl \
--do_train \
--output_dir /mnt/data1/output/model/Mistral_7B_v1_alpaca_2 \
--overwrite_output_dir \
--num_train_epochs 5 \
--learning_rate 2e-5 \
--evaluation_strategy steps \
--eval_steps 1300 \
--bf16 True \
--save_strategy steps \
--save_steps 6500 \
--logging_steps 10 \
--tf32 True \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--save_total_limit 50 \
--save_only_model True \
--fsdp "full_shard" \
--fsdp_config ./fsdp_config/fsdp_config.json >Mistral_7B_v1_alpaca_2.out 2>&1 &




torchrun --nproc_per_node=2 --master_port=25640 train_sft_Llama-3.2-1B-Instruct.py \
--model_name_or_path /mnt/data1/models/Mistral-7B-v0.1 \
--train_file_path /mnt/data1/study/data/alpaca/jsonl/split_sys_prompt/Mistral_7B_Instruct_v3_alpaca_split_sys_prompt.jsonl \
--validate_file_path /mnt/data1/study/data/alpaca_gpt4/jsonl/split_sys_prompt/Mistral_7B_Instruct_v3_alpaca_gpt4_split_sys_prompt_1k.jsonl \
--do_train \
--output_dir /mnt/data1/output/model/Mistral_7B_v1_alpaca_neuron_150000_2 \
--overwrite_output_dir \
--num_train_epochs 5 \
--learning_rate 2e-5 \
--evaluation_strategy steps \
--eval_steps 1300 \
--bf16 True \
--save_strategy steps \
--save_steps 6500 \
--logging_steps 10 \
--tf32 True \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--save_total_limit 50 \
--save_only_model False \
--fsdp "full_shard" \
--fsdp_config ./fsdp_config/fsdp_config.json >Mistral_7B_v1_alpaca_neuron_150000_2.out 2>&1 &


14336 * 4096 = 58720256

"""

