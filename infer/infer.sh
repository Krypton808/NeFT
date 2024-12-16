

# layer sft infer
#for n in 11 12 13 14 15 16;
#do
#   echo ${n}
#   CUDA_VISIBLE_DEVICES=6,7 python lora_model_infer.py \
#    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/mt/layer/enzh_sft_layer${n}_deepspeed/checkpoint-2500 \
#    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/enzh/eval.jsonl \
#    --prompt 'Translate the following text from English to Chinese.' \
#    --key_name 'instruction' \
#    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/sn/greedy_decode/layer/enzh_sft_layer${n}_deepspeed/2500steps/enzh_output.csv
#done
# 0,1,2,3
# 0,1,2,3

# 改模型路径 输出路径
# MT one model cross language infer
#for n in 750 375;
#for n in 800 1600 2400 3200 4000 4800 5600 6400 7200 8000 8800 9600;
#for n in 375 750 1125 1500 1875 2250;
#for n in 250 500 750 1000;
#for n in 6400 7200 8000 8800 9600;
#for n in 4150 8300 12450 16600;
#for n in 50000 75000 100000 125000 150000;
for n in 1


do
   echo ${n}

#   # enzh
#   CUDA_VISIBLE_DEVICES=3,4,5,7 python lora_model_infer.py \
#    --model_path /mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf \
#    --lora_weight_path /mnt/nfs/algo/intern/haoyunx11/models/sft/final/mt/lora/enzh/10w/rank_${n}/checkpoint-45650/adapter_model \
#    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/enzh/eval.jsonl \
#    --prompt 'Translate the following text from English to Chinese.' \
#    --key_name 'instruction' \
#    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/mt/train_neuron/enzh/lora/10w/rank_${n}/45650step/enzh_output.csv
#
#  # hizh
#   CUDA_VISIBLE_DEVICES=3,4,5,7 python lora_model_infer.py \
#    --model_path /mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf \
#    --lora_weight_path /mnt/nfs/algo/intern/haoyunx11/models/sft/final/mt/lora/enzh/10w/rank_${n}/checkpoint-45650/adapter_model \
#    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/hizh/eval.jsonl \
#    --prompt 'Translate the following text from Hindi to Chinese.' \
#    --key_name 'instruction' \
#    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/mt/train_neuron/enzh/lora/10w/rank_${n}/45650step/hizh_output.csv
#
#  # frzh
#   CUDA_VISIBLE_DEVICES=3,4,5,7 python lora_model_infer.py \
#    --model_path /mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf \
#    --lora_weight_path /mnt/nfs/algo/intern/haoyunx11/models/sft/final/mt/lora/enzh/10w/rank_${n}/checkpoint-45650/adapter_model \
#    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/frzh/eval.jsonl \
#    --prompt 'Translate the following text from French to Chinese.' \
#    --key_name 'instruction' \
#    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/mt/train_neuron/enzh/lora/10w/rank_${n}/45650step/frzh_output.csv



#    # enzh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 python lora_model_infer.py \
#    --model_path /mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf \
#    --lora_weight_path /mnt/nfs/algo/intern/haoyunx11/models/sft/final/mt/lora/enzh/rank_256/checkpoint-${n} \
#    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/enzh/eval.jsonl \
#    --prompt 'Translate the following text from English to Chinese.' \
#    --key_name 'instruction' \
#    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/mt/train_neuron/enzh/lora/rank_256/${n}step/enzh_output.csv
#
#  # hizh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 python lora_model_infer.py \
#    --model_path /mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf \
#    --lora_weight_path /mnt/nfs/algo/intern/haoyunx11/models/sft/final/mt/lora/enzh/rank_256/checkpoint-${n} \
#    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/hizh/eval.jsonl \
#    --prompt 'Translate the following text from Hindi to Chinese.' \
#    --key_name 'instruction' \
#    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/mt/train_neuron/enzh/lora/rank_256/${n}step/hizh_output.csv
#
#  # frzh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 python lora_model_infer.py \
#    --model_path /mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf \
#    --lora_weight_path /mnt/nfs/algo/intern/haoyunx11/models/sft/final/mt/lora/enzh/rank_256/checkpoint-${n} \
#    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/frzh/eval.jsonl \
#    --prompt 'Translate the following text from French to Chinese.' \
#    --key_name 'instruction' \
#    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/mt/train_neuron/enzh/lora/rank_256/${n}step/frzh_output.csv
#
#
#
#
#
## enzh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 python lora_model_infer.py \
#    --model_path /mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf \
#    --lora_weight_path /mnt/nfs/algo/intern/haoyunx11/models/sft/final/mt/lora/frzh/10w/rank_256/checkpoint-${n} \
#    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/enzh/eval.jsonl \
#    --prompt 'Translate the following text from English to Chinese.' \
#    --key_name 'instruction' \
#    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/mt/train_neuron/frzh/lora/10w/rank_256/${n}step/enzh_output.csv
#
#  # hizh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 python lora_model_infer.py \
#    --model_path /mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf \
#    --lora_weight_path /mnt/nfs/algo/intern/haoyunx11/models/sft/final/mt/lora/frzh/10w/rank_256/checkpoint-${n} \
#    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/hizh/eval.jsonl \
#    --prompt 'Translate the following text from Hindi to Chinese.' \
#    --key_name 'instruction' \
#    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/mt/train_neuron/frzh/lora/10w/rank_256/${n}step/hizh_output.csv
#
#  # frzh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 python lora_model_infer.py \
#    --model_path /mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf \
#    --lora_weight_path /mnt/nfs/algo/intern/haoyunx11/models/sft/final/mt/lora/frzh/10w/rank_256/checkpoint-${n} \
#    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/frzh/eval.jsonl \
#    --prompt 'Translate the following text from French to Chinese.' \
#    --key_name 'instruction' \
#    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/mt/train_neuron/frzh/lora/10w/rank_256/${n}step/frzh_output.csv
#
#
#    # enzh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 python lora_model_infer.py \
#    --model_path /mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf \
#    --lora_weight_path /mnt/nfs/algo/intern/haoyunx11/models/sft/final/mt/lora/frzh/rank_256/checkpoint-${n} \
#    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/enzh/eval.jsonl \
#    --prompt 'Translate the following text from English to Chinese.' \
#    --key_name 'instruction' \
#    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/mt/train_neuron/frzh/lora/rank_256/${n}step/enzh_output.csv
#
#  # hizh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 python lora_model_infer.py \
#    --model_path /mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf \
#    --lora_weight_path /mnt/nfs/algo/intern/haoyunx11/models/sft/final/mt/lora/frzh/rank_256/checkpoint-${n} \
#    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/hizh/eval.jsonl \
#    --prompt 'Translate the following text from Hindi to Chinese.' \
#    --key_name 'instruction' \
#    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/mt/train_neuron/frzh/lora/rank_256/${n}step/hizh_output.csv
#
#  # frzh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 python lora_model_infer.py \
#    --model_path /mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf \
#    --lora_weight_path /mnt/nfs/algo/intern/haoyunx11/models/sft/final/mt/lora/frzh/rank_256/checkpoint-${n} \
#    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/frzh/eval.jsonl \
#    --prompt 'Translate the following text from French to Chinese.' \
#    --key_name 'instruction' \
#    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/mt/train_neuron/frzh/lora/rank_256/${n}step/frzh_output.csv
#
#
#
#
#
#
## enzh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 python lora_model_infer.py \
#    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/final/mt/train_neuron/all_layer_in_out/enzh/10w/checkpoint-16600 \
#    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/enzh/eval.jsonl \
#    --prompt 'Translate the following text from English to Chinese.' \
#    --key_name 'instruction' \
#    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/mt/train_neuron/all_layer_in_out/enzh/10w/16600step/enzh_output.csv
#
#  # hizh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 python lora_model_infer.py \
#    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/final/mt/train_neuron/all_layer_in_out/enzh/10w/checkpoint-16600 \
#    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/hizh/eval.jsonl \
#    --prompt 'Translate the following text from Hindi to Chinese.' \
#    --key_name 'instruction' \
#    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/mt/train_neuron/all_layer_in_out/enzh/10w/16600step/hizh_output.csv
#
#  # frzh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 python lora_model_infer.py \
#    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/final/mt/train_neuron/all_layer_in_out/enzh/10w/checkpoint-16600 \
#    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/frzh/eval.jsonl \
#    --prompt 'Translate the following text from French to Chinese.' \
#    --key_name 'instruction' \
#    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/mt/train_neuron/all_layer_in_out/enzh/10w/16600step/frzh_output.csv


## enzh
   CUDA_VISIBLE_DEVICES=0 python lora_model_infer.py \
    --model_path /data2/haoyun/models/sft/final/mt/train_neuron/enzh/not_offload/pearson/150000/checkpoint-3200 \
    --validate_file_path /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/enzh/eval.jsonl \
    --prompt 'Translate the following text from English to Chinese.' \
    --key_name 'instruction' \
    --save_path /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/4_3090/mt/train_neuron/enzh/not_offload/pearson/150000/enzh_output.csv

  # hizh
   CUDA_VISIBLE_DEVICES=0 python lora_model_infer.py \
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/final/mt/train_neuron/hizh/750step_200000/checkpoint-750 \
    --validate_file_path /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/hizh/eval.jsonl \
    --prompt 'Translate the following text from Hindi to Chinese.' \
    --key_name 'instruction' \
    --save_path /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/4_3090/mt/train_neuron/hizh/hizh_output.csv >infer_output.log 2>&1 &

  # frzh
   CUDA_VISIBLE_DEVICES=0 python lora_model_infer.py \
    --model_path /data2/haoyun/models/sft/final/mt/train_neuron/frzh/not_offload/pearson/150000/checkpoint-3200 \
    --validate_file_path /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/frzh/eval.jsonl \
    --prompt 'Translate the following text from French to Chinese.' \
    --key_name 'instruction' \
    --save_path /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/4_3090/mt/train_neuron/frzh/not_offload/pearson/150000/frzh_output.csv


# enzh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 python lora_model_infer.py \
#    --model_path /data/haoyun.xu/models/sft/final/mt/train_neuron/enzh/not_offload/50000+reversed-${n}/checkpoint-3200 \
#    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/enzh/eval.jsonl \
#    --prompt 'Translate the following text from English to Chinese.' \
#    --key_name 'instruction' \
#    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/mt/train_neuron/enzh/not_offload/50000+reversed-${n}/3200step/enzh_output.csv
#
#  # hizh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 python lora_model_infer.py \
#    --model_path /data/haoyun.xu/models/sft/final/mt/train_neuron/enzh/not_offload/50000+reversed-${n}/checkpoint-3200 \
#    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/hizh/eval.jsonl \
#    --prompt 'Translate the following text from Hindi to Chinese.' \
#    --key_name 'instruction' \
#    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/mt/train_neuron/enzh/not_offload/50000+reversed-${n}/3200step/hizh_output.csv
#
#  # frzh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 python lora_model_infer.py \
#    --model_path /data/haoyun.xu/models/sft/final/mt/train_neuron/enzh/not_offload/50000+reversed-${n}/checkpoint-3200 \
#    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/frzh/eval.jsonl \
#    --prompt 'Translate the following text from French to Chinese.' \
#    --key_name 'instruction' \
#    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/mt/train_neuron/enzh/not_offload/50000+reversed-${n}/3200step/frzh_output.csv










#  CUDA_VISIBLE_DEVICES=0,1,2,3 python lora_model_infer.py \
#    --model_path  /data/haoyun.xu/models/llm/Llama-2-7b-chat-hf \
#    --lora_weight_path /data/haoyun.xu/models/sft/final/mt/lora/enmi/rank_16/checkpoint-4800/adapter_model \
#    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/news/enmi/test.jsonl \
#    --prompt 'Translate the following text from English to Maori.' \
#    --key_name 'instruction' \
#    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/mt/train_neuron/enmi/lora/rank_16/4800step/enmi_output.csv


#CUDA_VISIBLE_DEVICES=3,4,5,7 python lora_model_infer.py \
#    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/final/mt/train_neuron/all_layer_in_out/enmi/checkpoint-1600 \
#    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/news/enmi/test.jsonl \
#    --prompt 'Translate the following text from English to Maori.' \
#    --key_name 'instruction' \
#    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/mt/train_neuron/all_layer_in_out/enmi/1600step/enmi_output.csv

#CUDA_VISIBLE_DEVICES=0,1,2,3 python lora_model_infer.py \
#    --model_path /data/haoyun.xu/models/sft/final/mt/embedding/enmi/checkpoint-6400 \
#    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/news/enmi/test.jsonl \
#    --prompt 'Translate the following text from English to Maori.' \
#    --key_name 'instruction' \
#    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/mt/embedding/enmi/enmi_output.csv


#CUDA_VISIBLE_DEVICES=0,1,2,3 python lora_model_infer.py \
#    --model_path /data/haoyun.xu/models/llm/Llama-2-7b-chat-hf \
#    --lora_weight_path /data/haoyun.xu/models/sft/final/mt/lora/enbs/rank_16/checkpoint-4800/adapter_model \
#    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/news/enbs/test.jsonl \
#    --prompt 'Translate the following text from English to Bosnian.' \
#    --key_name 'instruction' \
#    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/mt/train_neuron/enbs/lora/rank_16/4800step/enbs_output.csv

#CUDA_VISIBLE_DEVICES=0,1,2,3 python lora_model_infer.py \
#    --model_path /data/haoyun.xu/models/sft/final/mt/embedding/enbs/checkpoint-6400 \
#    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/news/enbs/test.jsonl \
#    --prompt 'Translate the following text from English to Bosnian.' \
#    --key_name 'instruction' \
#    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/mt/embedding/enbs/6400step/enbs_output.csv

#CUDA_VISIBLE_DEVICES=0,1,2,3 python lora_model_infer.py \
#    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/final/mt/train_neuron/enbs/enbs_100000/checkpoint-${n} \
#    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/news/enbs/test.jsonl \
#    --prompt 'Translate the following text from English to Bosnian.' \
#    --key_name 'instruction' \
#    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/mt/train_neuron/enbs/1600step_100000/${n}step/enbs_output.csv

#    CUDA_VISIBLE_DEVICES=0,1,2,3 python lora_model_infer.py \
#    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/final/mt/org/enzh/checkpoint-3200 \
#    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/enzh/eval.jsonl \
#    --prompt 'Translate the following text from English to Chinese.' \
#    --key_name 'instruction' \
#    --save_path /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/4_3090/mt/org/enzh/3200step_2/enzh_output.csv


#    CUDA_VISIBLE_DEVICES=0,1,2,3 python lora_model_infer.py \
#    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/final/other_2/checkpoint-${n} \
#    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/other/org/test.jsonl \
#    --prompt "" \
#    --key_name 'input' \
#    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/other_2/org/${n}step/output.csv



#    CUDA_VISIBLE_DEVICES=0,1,2,3 python lora_model_infer.py \
#    --model_path /mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf \
#    --lora_weight_path /mnt/nfs/algo/intern/haoyunx11/models/sft/final/lora/rank_8/checkpoint-${n}\
#    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/other/org/test.jsonl \
#    --prompt "" \
#    --key_name 'input' \
#    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/other/lora/${n}step/output.csv


#    CUDA_VISIBLE_DEVICES=0,1,2,3 python lora_model_infer.py \
#    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/final/other/checkpoint-${n} \
#    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/other/org/test.jsonl \
#    --prompt "" \
#    --key_name 'input' \
#    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/other/org/${n}step/output.csv

# enfr
#CUDA_VISIBLE_DEVICES=0,1,2,3 python lora_model_infer.py \
#    --model_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/mt/embedding/enfr/10w/checkpoint-49800 \
#    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/enfr/eval.jsonl \
#    --prompt 'Translate the following text from English to French.' \
#    --key_name 'instruction' \
#    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/mt/embedding/enfr/10w/enfr_output.csv
#
## hifr
#CUDA_VISIBLE_DEVICES=0,1,2,3 python lora_model_infer.py \
#    --model_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/mt/embedding/enfr/10w/checkpoint-49800 \
#    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/hifr/eval.jsonl \
#    --prompt 'Translate the following text from Hindi to French.' \
#    --key_name 'instruction' \
#    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/mt/embedding/enfr/10w/hifr_output.csv

# enfr
#    CUDA_VISIBLE_DEVICES=0,1,2,3 python lora_model_infer.py \
#    --model_path /mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf \
#    --lora_weight_path /mnt/nfs/algo/intern/haoyunx11/models/sft/final/mt/lora/hifr/rank_${n}/checkpoint-2250/adapter_model \
#    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/enfr/eval.jsonl \
#    --prompt 'Translate the following text from English to French.' \
#    --key_name 'instruction' \
#    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/mt/train_neuron/hifr/lora/rank_${n}/2250step/enfr_output.csv
#
## hifr
#    CUDA_VISIBLE_DEVICES=0,1,2,3 python lora_model_infer.py \
#    --model_path /mnt/nfs/algo/intern/haoyunx11/models/llm/llama-2/Llama-2-7b-chat-hf \
#    --lora_weight_path /mnt/nfs/algo/intern/haoyunx11/models/sft/final/mt/lora/hifr/rank_${n}/checkpoint-2250/adapter_model \
#    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/hifr/eval.jsonl \
#    --prompt 'Translate the following text from Hindi to French.' \
#    --key_name 'instruction' \
#    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/mt/train_neuron/hifr/lora/rank_${n}/2250step/hifr_output.csv
#
#
## enfr
#    CUDA_VISIBLE_DEVICES=0,1,2,3 python lora_model_infer.py \
#    --model_path /data/haoyun.xu/models/llm/Llama-2-7b-chat-hf \
#    --lora_weight_path /data/haoyun.xu/models/sft/final/mt/lora/hifr/rank_${n}/checkpoint-3000/adapter_model \
#    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/enfr/eval.jsonl \
#    --prompt 'Translate the following text from English to French.' \
#    --key_name 'instruction' \
#    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/mt/train_neuron/hifr/lora/rank_${n}/3000step/enfr_output.csv
#
## hifr
#    CUDA_VISIBLE_DEVICES=0,1,2,3 python lora_model_infer.py \
#    --model_path /data/haoyun.xu/models/llm/Llama-2-7b-chat-hf \
#    --lora_weight_path /data/haoyun.xu/models/sft/final/mt/lora/hifr/rank_${n}/checkpoint-3000/adapter_model \
#    --validate_file_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/hifr/eval.jsonl \
#    --prompt 'Translate the following text from Hindi to French.' \
#    --key_name 'instruction' \
#    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/mt/train_neuron/hifr/lora/rank_${n}/3000step/hifr_output.csv




# bash infer.sh >infer_output.log 2>&1 &

done
