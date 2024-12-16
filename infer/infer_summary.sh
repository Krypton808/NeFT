
# 4,5,6,7

# 改模型路径 输出路径
# MT one model cross language infer
#for n in 750 375;
for n in 790 395;
do
   echo ${n}

   # org
   # enzh
#  CUDA_VISIBLE_DEVICES=0,1,2,3 python lora_model_infer_summary_task.py \
#    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/final/summary/org/enzh/checkpoint-${n} \
#    --validate_file_path /data/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/data/CrossSum/english-chinese_simplified/1024_cut_english-chinese_simplified_test.jsonl \
#    --prompt "Given the below English article, generate a summary in Chinese.\nArticle: " \
#    --key_name 'text' \
#    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/summary/org/enzh/${n}step/enzh_output.csv
#
#  # frzh
#  CUDA_VISIBLE_DEVICES=0,1,2,3 python lora_model_infer_summary_task.py \
#    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/final/summary/org/enzh/checkpoint-${n} \
#    --validate_file_path /data/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/data/CrossSum/french-chinese_simplified/1024_cut_french-chinese_simplified_test.jsonl \
#    --prompt "Given the below French article, generate a summary in Chinese.\nArticle: " \
#    --key_name 'text' \
#    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/summary/org/enzh/${n}step/frzh_output.csv
#
#  # hizh
#  CUDA_VISIBLE_DEVICES=0,1,2,3 python lora_model_infer_summary_task.py \
#    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/final/summary/org/enzh/checkpoint-${n} \
#    --validate_file_path /data/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/data/CrossSum/hindi-chinese_simplified/1024_cut_hindi-chinese_simplified_test.jsonl \
#    --prompt "Given the below Hindi article, generate a summary in Chinese.\nArticle: " \
#    --key_name 'text' \
#    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/summary/org/enzh/${n}step/hizh_output.csv



  # train neuron
  # enzh
#  CUDA_VISIBLE_DEVICES=4,5,6,7 python lora_model_infer_summary_task.py \
#    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/final/summary/train_neuron/enzh/790step_200000/checkpoint-${n} \
#    --validate_file_path /data/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/data/CrossSum/english-chinese_simplified/1024_cut_english-chinese_simplified_test.jsonl \
#    --prompt "Given the below English article, generate a summary in Chinese.\nArticle: " \
#    --key_name 'text' \
#    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/summary/train_neuron/enzh/790step_200000/${n}step/enzh_output.csv
#
#  # frzh
#  CUDA_VISIBLE_DEVICES=4,5,6,7 python lora_model_infer_summary_task.py \
#    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/final/summary/train_neuron/enzh/790step_200000/checkpoint-${n} \
#    --validate_file_path /data/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/data/CrossSum/french-chinese_simplified/1024_cut_french-chinese_simplified_test.jsonl \
#    --prompt "Given the below French article, generate a summary in Chinese.\nArticle: " \
#    --key_name 'text' \
#    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/summary/train_neuron/enzh/790step_200000/${n}step/frzh_output.csv
#
#  # hizh
#  CUDA_VISIBLE_DEVICES=4,5,6,7 python lora_model_infer_summary_task.py \
#    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/final/summary/train_neuron/enzh/790step_200000/checkpoint-${n} \
#    --validate_file_path /data/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/data/CrossSum/hindi-chinese_simplified/1024_cut_hindi-chinese_simplified_test.jsonl \
#    --prompt "Given the below Hindi article, generate a summary in Chinese.\nArticle: " \
#    --key_name 'text' \
#    --save_path /data/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/summary/train_neuron/enzh/790step_200000/${n}step/hizh_output.csv



  # union
  # enzh
  CUDA_VISIBLE_DEVICES=0,1 python infer.py \
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/final/summary/train_neuron/enzh/790step_50000/checkpoint-${n} \
    --validate_file_path /data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/data/CrossSum/english-chinese_simplified/2048_cut_english-chinese_simplified_test.jsonl \
    --prompt "Given the below English article, generate a summary in Chinese.\nArticle: " \
    --key_name 'text' \
    --save_path /data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/infer/output/final/summary/train_neuron/enzh/790step_50000/${n}step/enzh_output.csv

  # frzh
  CUDA_VISIBLE_DEVICES=0,1 python infer.py \
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/final/summary/train_neuron/enzh/790step_50000/checkpoint-${n} \
    --validate_file_path /data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/data/CrossSum/french-chinese_simplified/2048_cut_french-chinese_simplified_test.jsonl \
    --prompt "Given the below French article, generate a summary in Chinese.\nArticle: " \
    --key_name 'text' \
    --save_path /data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/infer/output/final/summary/train_neuron/enzh/790step_50000/${n}step/frzh_output.csv

  # hizh
  CUDA_VISIBLE_DEVICES=0,1 python infer.py \
    --model_path /mnt/nfs/algo/intern/haoyunx11/models/sft/final/summary/train_neuron/enzh/790step_50000/checkpoint-${n} \
    --validate_file_path /data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/baselines/data/CrossSum/hindi-chinese_simplified/2048_cut_hindi-chinese_simplified_test.jsonl \
    --prompt "Given the below Hindi article, generate a summary in Chinese.\nArticle: " \
    --key_name 'text' \
    --save_path /data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/infer/output/final/summary/train_neuron/enzh/790step_50000/${n}step/hizh_output.csv




done
