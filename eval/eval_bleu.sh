for n in 11 12 13 14 15 16;
do
   echo ${n}
   sacrebleu /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/enzh/ref.txt \
    -l en-zh \
    -i /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/sn/greedy_decode/layer/enzh_sft_layer${n}_deepspeed/2500steps/enzh_output.csv \
    -m bleu \
    -w 4 --lowercase
done