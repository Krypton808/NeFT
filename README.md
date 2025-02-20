<h2 align="center"> <a href="https://arxiv.org/abs/2403.11621">[COLING 2025] Let's Focus on Neuron: Neuron-Level Supervised Fine-tuning for Large Language Model </a></h2>

<h5 align="center"> 

If you like our project, please give us a star ⭐ on GitHub for the latest update.  </h2>

</h5>

<div align=center>
  
[![arXiv](https://img.shields.io/badge/Arxiv-2410.23746-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2403.11621)

</div>

<img src = src/main_pic.png>


## 📣 News
* `[2025/01/17]`  ✨ Our paper is accepted by **The 31st International Conference on Computational Linguistics (COLING 2025)**. See you in Abu Dhabi, UAE!


## An application using NeFT
### Model: Mistral-7B-v0.1
### Dataset: alpaca
## Step1: Prepare Full-Finetuned Model (FT-full)

```shell
cd baselines
torchrun --nproc_per_node=4 --master_port=25647 train_sft.py \
--model_name_or_path {model_name_or_path} \
--train_file_path {train_file_path} \
--validate_file_path {validate_file_path} \
--do_train \
--output_dir {output_dir} \
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
--fsdp_config ./fsdp_config/fsdp_config.json
```

## Step2: Find Neurons
```shell
cd find_neuron
python cal_neuron_sim.py
# In this .py file, run the function run_get_Mistral_7B_v1_neuron(), the following variables need to be set:
# 1. model_path_1, the org model {model_name_or_path}
# 2. model_path_2, pick a checkpoint in {output_dir}
# 3. path, path that preserves the similarity scores of neurons and mask files
# 4. neuron_number, number of neurons ready for NeFT
```

## Step3: Train Neurons (NeFT)
```shell
cd train_neuron
torchrun --nproc_per_node=2 --master_port=25640 train_sft_Llama-3.2-1B-Instruct.py \
--model_name_or_path {model_name_or_path} \
--train_file_path {train_file_path} \
--validate_file_path {validate_file_path} \
--do_train \
--output_dir {NeFT_output_dir} \
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
--fsdp_config ./fsdp_config/fsdp_config.json

# Inside "trainer_add_grad_mask_load_neuron_dict_Mistral_7B_v1.py"
# there is a variable "mask_dir", change the value of this variable to your mask file path


```

Possible runtime issues:

I used 2 80G A800 GPUs to train with NeFT, an error occurs when attempting to use more GPUs, indicating a mismatch between the dimensions of the mask and the gradient. This issue may be related to the sharding mechanism of the training framework. I will follow up on this engineering issue later.

## Citation

If you find our paper and code useful in your research, please consider giving a star ⭐ and citation 📝.

```BibTeX
@article{xu2024let,
  title={Let's Focus on Neuron: Neuron-Level Supervised Fine-tuning for Large Language Model},
  author={Xu, Haoyun and Zhan, Runzhe and Wong, Derek F and Chao, Lidia S},
  journal={arXiv preprint arXiv:2403.11621},
  year={2024}
}
```