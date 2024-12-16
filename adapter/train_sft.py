import os
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import torch
import transformers
from modeling import LlamaForCausalLM

# os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_PROJECT"] = "en_adapter_all_layers_2"

@dataclass
class SFTConfig:
    model_name_or_path: Optional[str] = field(metadata={"help": "Path to pretrained model checkpoint"})
    dataset_name: Optional[str] = field(default=None, metadata={"help": "Huggingface dataset name"})
    train_file_path: Optional[str] = field(default=None, metadata={"help": "Path to train data file/directory"})
    validate_file_path: Optional[str] = field(default=None, metadata={"help": "Path to validation data file/directory"})
    max_length: int = field(default=1024, metadata={"help": "Max length of input"})
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
    metric = evaluate.load("accuracy")
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

    model = LlamaForCausalLM.from_pretrained(sft_config.model_name_or_path)

    for k, v in model.named_parameters():
        if "adapter_linear" in k:
            v.requires_grad = True
            print(k)
            print(v)

        else:
            v.requires_grad = False
            print(k)
            print(v)


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
        input_s = record['input']
        output_s = record['output']
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
        processed_record["labels"][:min(len(tokenized["input_ids"][0]), sft_config.max_length)] = [-100] * min(len(tokenized["input_ids"][0]), sft_config.max_length)
        return {k: torch.tensor(v, dtype=torch.int) for k, v in processed_record.items()}

    with training_args.main_process_first(desc="Process supervised dataset"):
        sft_dataset = raw_datasets.map(
            process_supervised,
            batched=False,
            num_proc=sft_config.preprocess_num_workers,
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

    )

    # trigger Training
    trainer.train()
    trainer.save_model()
    trainer.save_state()


if __name__ == '__main__':
    main()
"""
nohup deepspeed \
--include="localhost:6,7" \
./train_sft.py \
--deepspeed ./ds_config/ds_config_zero2.json \
--model_name_or_path /mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf \
--train_file_path /data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/data/xnli/en_standard_prompt_3w.jsonl \
--validate_file_path /data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/data/xnli/en_standard_prompt_last1k_eval.jsonl \
--do_train \
--output_dir /mnt/nfs/algo/intern/haoyunx11/models/sft/xnli/en_adapter_all_layers_2 \
--overwrite_output_dir \
--preprocess_num_workers 8 \
--num_train_epochs 1 \
--max_steps 500 \
--learning_rate 2e-5 \
--evaluation_strategy steps \
--eval_steps 500 \
--bf16 True \
--save_strategy steps \
--save_steps 100 \
--save_total_limit 2 \
--logging_steps 10 \
--tf32 True \
--per_device_train_batch_size 6 \
--per_device_eval_batch_size 6 \
--save_total_limit 5 \
--report_to wandb >sft_adapter_xnli_en_all_layers_2.out 2>&1 &
"""