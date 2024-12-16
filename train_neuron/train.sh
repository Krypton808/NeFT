
echo frzh_50000
torchrun --nproc_per_node=3 train_sft.py \
--model_name_or_path /mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf \
--train_file_path /data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/data/CrossSum/french-chinese_simplified/1024_cut_french-chinese_simplified_train.jsonl \
--validate_file_path /data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/data/CrossSum/french-chinese_simplified/1024_cut_french-chinese_simplified_val.jsonl \
--do_train \
--output_dir /mnt/nfs/algo/intern/haoyunx11/models/sft/final/summary/train_neuron/frzh/38step_100000 \
--overwrite_output_dir \
--num_train_epochs 2 \
--learning_rate 2e-5 \
--evaluation_strategy steps \
--eval_steps 2 \
--bf16 True \
--save_strategy epoch \
--logging_steps 2 \
--tf32 True \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--save_total_limit 10 \
--fsdp "full_shard" \
--fsdp_config ./fsdp_config/fsdp_config.json \
--prompt "Given the below French article, generate a summary in Chinese." + "\nArticle: " \
--


CUDA_VISIBLE_DEVICES=5,6,7 torchrun --nproc_per_node=3 --master_port=25625 train_sft.py \
--model_name_or_path /mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf \
--train_file_path /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/news/hizh/train_clean.jsonl \
--validate_file_path /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/news/hizh/dev_435.jsonl \
--do_train \
--output_dir /mnt/nfs/algo/intern/haoyunx11/models/sft/final/mt/train_neuron/overlap/x-zh/200000_find_118103/hizh \
--overwrite_output_dir \
--preprocess_num_workers 8 \
--num_train_epochs 2 \
--learning_rate 2e-5 \
--evaluation_strategy steps \
--eval_steps 75 \
--bf16 True \
--save_strategy steps \
--save_steps 375 \
--logging_steps 10 \
--tf32 True \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--save_total_limit 20 \
--fsdp "full_shard" \
--fsdp_config ./fsdp_config/fsdp_config.json \
--prompt 'Translate the following text from Hindi to Chinese.'