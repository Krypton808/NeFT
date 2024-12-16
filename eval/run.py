import os
cmd = """
sacrebleu /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/enzh/ref.txt \
-l en-zh \
-i /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/4_3090/mt/train_neuron/hizh/lora/rank_256/2250step/enzh_output.csv \
-m bleu \
-w 4

sacrebleu /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/frzh/ref.txt \
-l fr-zh \
-i /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/4_3090/mt/train_neuron/hizh/lora/rank_256/2250step/frzh_output.csv \
-m bleu \
-w 4

sacrebleu /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/hizh/ref.txt \
-l hi-zh \
-i /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/final/4_3090/mt/train_neuron/hizh/lora/rank_256/2250step/hizh_output.csv \
-m bleu \
-w 4


"""
os.system(cmd)
