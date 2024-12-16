import os

os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from typing import Optional
from dataclasses import dataclass, field
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    AutoModelForCausalLM,
    BloomForCausalLM
)
import argparse
from loguru import logger

from os.path import join
import torch
import bitsandbytes as bnb
from collections import defaultdict
import datasets
from transformers import Trainer, DataCollatorForTokenClassification
from transformers.trainer import has_length, is_datasets_available, LengthGroupedSampler, SequentialSampler


def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
    if self.train_dataset is None or not has_length(self.train_dataset):
        return None

    # Build the sampler.
    if self.args.group_by_length:
        if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
            lengths = (
                self.train_dataset[self.args.length_column_name]
                if self.args.length_column_name in self.train_dataset.column_names
                else None
            )
        else:
            lengths = None
        model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
        return LengthGroupedSampler(
            self.args.train_batch_size * self.args.gradient_accumulation_steps,
            dataset=self.train_dataset,
            lengths=lengths,
            model_input_name=model_input_name,
        )

    else:
        return SequentialSampler(self.train_dataset)


@dataclass
class PeftConfig:
    model_name_or_path: Optional[str] = field(metadata={"help": "Path to pretrained model checkpoint"})
    # data_files: Optional[str] = field(default=None, metadata={"help": "Local data files"})

    train_file_path: Optional[str] = field(default=None, metadata={"help": "Local data files"})
    validate_file_path: Optional[str] = field(default=None, metadata={"help": "Local data files"})

    max_length: int = field(default=512, metadata={
        "help": "Maximum source + target sequence length. Sequences will be right padded (and possibly truncated)."}, )
    preprocess_num_workers: int = field(default=4,
                                        metadata={"help": "The number of processes to use for the preprocessing."})
    pad_to_max_length: bool = field(default=True, metadata={
        "help": "Pad all examples to max_length. This is for fair comparison between different batch size"})

    # Lora related param here
    adam8bit: bool = field(default=False, metadata={"help": "Use 8-bit adam."})
    double_quant: bool = field(default=True,
                               metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4",
                            metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    bits: int = field(default=4, metadata={"help": "How many bits to use."})
    lora_r: int = field(default=64, metadata={"help": "Lora R dimension."})
    lora_alpha: float = field(default=16, metadata={"help": " Lora alpha."})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout."})
    full_finetune: bool = field(default=False, metadata={"help": "Finetune the entire model without adapters."})
    do_infer: bool = field(default=False, metadata={"help": "Finetune the entire model without adapters."})


def verify_model_dtype(model):
    """
    查看模型种各种类型的参数的情况
    """
    dtype2param_num = defaultdict(int)  # 每种数据类型的参数量
    dtype2param_name = defaultdict(list)  # 每种数据类型的参数名称
    dtype2trainable_param_num = defaultdict(int)  # 每种数据类型参与训练的参数量
    dtype2trainable_param_name = defaultdict(list)  # 每种数据类型参与训练的参数名称
    for name, p in model.named_parameters():
        dtype = p.dtype
        dtype2param_num[dtype] += p.numel()
        dtype2param_name[dtype].append(name)
        if p.requires_grad:
            dtype2trainable_param_num[dtype] += p.numel()
            dtype2trainable_param_name[dtype].append(name)
    # 统计全部参数中，各种类型参数分布
    total = 0
    print('verify all params of the model')
    for k, v in dtype2param_num.items():
        total += v
    for k, v in dtype2param_num.items():
        print(k, v, v / total)
    for k, v in dtype2trainable_param_name.items():
        print(k, v)

    print()
    # 统计可训练参数中，各种类型参数分布
    print('verify trainable params the model')
    total_trainable = 0
    for k, v in dtype2trainable_param_num.items():
        total_trainable += v
    for k, v in dtype2trainable_param_num.items():
        print(k, v, v / total_trainable)
    for k, v in dtype2trainable_param_num.items():
        print(k, v)


def find_all_linear_names(model):
    """
    找出所有全连接层，为所有全连接添加adapter
    """
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def setup_everything():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_args_file", type=str, default='train_args/baichuan-sft-qlora.json', help="")
    args = parser.parse_args()
    train_args_file = args.train_args_file
    # 读取训练的参数配置
    parser = HfArgumentParser((PeftConfig, TrainingArguments))
    # 解析得到自定义参数，以及自带参数
    args, training_args = parser.parse_json_file(json_file=train_args_file, allow_extra_keys=True)

    # 创建输出目录
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    # logger.add(join(training_args.output_dir, 'train.log'))
    # logger.info("train_args:{}".format(training_args))
    # 设置随机种子
    set_seed(training_args.seed)
    return args, training_args


def init_components(args, training_args):
    """
    初始化各个组件
    """

    tok_ins = "\n\n### Instruction:\n"
    tok_res = "\n\n### Response:\n"
    prompt_input = tok_ins + "{instruction}" + tok_res

    logger.info('Initializing components...')

    training_args.ddp_find_unused_parameters = False
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    device_map = {'': local_rank}

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map=device_map,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        ),
    )
    # 加载tokenzier
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if model.config.model_type == 'llama' else True
    )
    # QWenTokenizer比较特殊，pad_token_id、bos_token_id、eos_token_id均为None。eod_id对应的token为<|endoftext|>
    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id
    # ChatGLMTokenizer不需要设置，仅设置其他tokenizer
    elif tokenizer.__class__.__name__ != 'ChatGLMTokenizer':
        assert tokenizer.eos_token_id is not None
        assert tokenizer.bos_token_id is not None
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id

    # casts all the non int8 modules to full precision (fp32) for stability
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    print(f'memory footprint of model: {model.get_memory_footprint() / (1024 * 1024 * 1024)} GB')
    # 找到所有需要插入adapter的全连接层
    target_modules = find_all_linear_names(model)
    # 初始化lora配置
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    model.config.torch_dtype = torch.float32

    # 查看模型种各种类型的参数的情况
    verify_model_dtype(model)

    raw_datasets = datasets.load_dataset("json", data_files={'train': args.train_file_path,
                                                             'validation': args.validate_file_path})

    def process_supervised(record, prompt='Translate the following text from English to German.', pad_to_max_length=True):
        input_s = prompt + '\n' + record['instruction']

        input_s = prompt_input.format_map({'instruction': input_s})

        output_s = record['output']
        tokenized = tokenizer([input_s, output_s])
        token_ids = [tok_id for tok_ids in tokenized['input_ids'] for tok_id in tok_ids]
        attention_mask = [mask for masks in tokenized['attention_mask'] for mask in masks]
        if token_ids[-1] != tokenizer.eos_token_id:
            token_ids += [tokenizer.eos_token_id]
            attention_mask += [1]
        processed_record = {
            "input_ids": token_ids[:args.max_length],
            "attention_mask": attention_mask[:args.max_length],
            "labels": token_ids.copy()[:args.max_length]
        }
        if pad_to_max_length:
            processed_record = {
                "input_ids": processed_record["input_ids"] + [tokenizer.pad_token_id] * (
                        args.max_length - len(processed_record["input_ids"])),
                "attention_mask": processed_record["attention_mask"] + [0] * (
                        args.max_length - len(processed_record["attention_mask"])),
                "labels": processed_record["labels"] + [-100] * (args.max_length - len(processed_record["labels"]))
            }

        # ignore input label, label is ignored if value is -100
        processed_record["labels"][:min(len(tokenized["input_ids"][0]), args.max_length)] = [-100] * min(
            len(tokenized["input_ids"][0]), args.max_length)

        return {k: torch.tensor(v, dtype=torch.int) for k, v in processed_record.items()}

    with training_args.main_process_first(desc="Process supervised dataset"):
        sft_dataset = raw_datasets.map(
            process_supervised,
            batched=False,
            num_proc=args.preprocess_num_workers,
            remove_columns=raw_datasets["train"].column_names,
            desc="Process supervised dataset"
        )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=sft_dataset["train"],
        eval_dataset=sft_dataset["validation"],
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer, padding="longest",
                                                         max_length=args.max_length,
                                                         label_pad_token_id=-100)
    )

    return trainer


def main():
    # Trainer._get_train_sampler = _get_train_sampler

    # 进行一些配置和检查
    args, training_args = setup_everything()
    # 加载各种组件
    trainer = init_components(args, training_args)
    # 开始训练
    logger.info("*** starting training ***")
    train_result = trainer.train()
    # 保存最好的checkpoint
    final_save_path = join(training_args.output_dir, 'final')
    trainer.save_model(final_save_path)  # Saves the tokenizer too
    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
