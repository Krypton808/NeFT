# comet gpus 设0 用cpu
"""
CUDA_VISIBLE_DEVICES=0 comet-score -s /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/enzh/src.txt \
-t /data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/IWSLT/output/enzh/news_150000/3200/IWSLT_news_150000_3200.csv \
-r /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/enzh/ref.txt \
--model_storage_path /data2/haoyun/models/comet/wmt22-comet-da --model Unbabel/wmt22-comet-da \
--gpus 1

CUDA_VISIBLE_DEVICES=3 comet-score -s /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/frzh/src.txt \
-t /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/4_3090/mt/llama2_chat/frzh_output.csv \
-r /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/frzh/ref.txt \
--model_storage_path /data2/haoyun/models/comet/wmt22-comet-da --model Unbabel/wmt22-comet-da \
--gpus 1

CUDA_VISIBLE_DEVICES=3 comet-score -s /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/hizh/src.txt \
-t /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/4_3090/mt/llama2_chat/hizh_output.csv \
-r /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/hizh/ref.txt \
--model_storage_path /data2/haoyun/models/comet/wmt22-comet-da --model Unbabel/wmt22-comet-da \
--gpus 1



CUDA_VISIBLE_DEVICES=3 comet-score -s /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/enfr/src.txt \
-t /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/4_3090/mt/llama2_chat/enfr_output.csv \
-r /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/enfr/ref.txt \
--model_storage_path /data2/haoyun/models/comet/wmt22-comet-da --model Unbabel/wmt22-comet-da \
--gpus 1

CUDA_VISIBLE_DEVICES=3 comet-score -s /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/hifr/src.txt \
-t /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/4_3090/mt/llama2_chat/hifr_output.csv \
-r /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/hifr/ref.txt \
--model_storage_path /data2/haoyun/models/comet/wmt22-comet-da --model Unbabel/wmt22-comet-da \
--gpus 1

CUDA_VISIBLE_DEVICES=3 comet-score -s /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/news/enmi/test.en \
-t /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/4_3090/mt/llama2_chat/enmi_output.csv \
-r /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/news/enmi/test.mi \
--model_storage_path /data2/haoyun/models/comet/wmt22-comet-da --model Unbabel/wmt22-comet-da \
--gpus 1

CUDA_VISIBLE_DEVICES=3 comet-score -s /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/news/enbs/test.en \
-t /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/4_3090/mt/llama2_chat/enbs_output.csv \
-r /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/news/enbs/test.bs \
--model_storage_path /data2/haoyun/models/comet/wmt22-comet-da --model Unbabel/wmt22-comet-da \
--gpus 1

sacrebleu /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/news/enbs/test.bs \
-l en-bs \
-i /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/4_3090/mt/train_neuron/enbs/lora/rank_256/4000step/enbs_output.csv \
-m bleu \
-w 4


CUDA_VISIBLE_DEVICES=3 comet-score -s /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/dezh/src.txt \
-t /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/sn/greedy_decode/dezh/train_dezh_org/2500steps/dezh_output.csv \
-r /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/dezh/ref.txt \
--model_storage_path /data2/haoyun/models/comet/wmt22-comet-da --model Unbabel/wmt22-comet-da \
--gpus 0


sacrebleu /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/enzh/ref.txt \
-l en-zh \
-i /data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/IWSLT/output/enzh/LoRA/rank_256/IWSLT_rank_256_4800.csv \
-m bleu \
-w 4

/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/IWSLT/output/enzh/LoRA/rank_16/IWSLT_rank_16_3200.csv
/data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/IWSLT/output/enzh/news_150000/3200/IWSLT_news_150000_3200.csv

sacrebleu /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/frzh/ref.txt \
-l fr-zh \
-i /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/4_3090/mt/train_neuron/overlap/x-zh/200000_find_118103/hizh/750step/frzh_output.csv \
-m bleu \
-w 4

sacrebleu /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/hizh/ref.txt \
-l hi-zh \
-i /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/4_3090/mt/train_neuron/overlap/x-zh/200000_find_118103/hizh/750step/hizh_output.csv \
-m bleu \
-w 4



sacrebleu /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/news/enmi/test.mi \
-l en-mi \
-i /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/4_3090/mt/train_neuron/all_layer_in_out/enmi/1600step/enmi_output.csv \
-m bleu \
-w 4


sacrebleu /data5/haoyun.xu/study/MI/MMMI/src/MI/experiment_setup/other/data/ref.txt \
-l en-en \
-i /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/4_3090/other/lora/6250step/output.csv \
-m bleu \
-w 4 \
--lowercase




sacrebleu /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/enfr/ref.txt \
-l en-fr \
-i /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/4_3090/mt/llama2_chat/enfr_output.csv \
-m bleu \
-w 4 \
--lowercase

sacrebleu /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/hifr/ref.txt \
-l hi-fr \
-i /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/4_3090/mt/llama2_chat/hifr_output.csv \
-m bleu \
-w 4 \
--lowercase


sh eval.sh >batchp_eval_500steps.log 2>&1 &


sh eval_bleu.sh >batchp_eval_bleu_500steps.log 2>&1 &

"""