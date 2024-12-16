for n in 500 1000 1500 2000 2500;
do
   echo ${n}
   sacrebleu /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/dezh/ref.txt \
    -l de-zh \
    -i /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/sn/greedy_decode/dezh/train_dezh_org/${n}steps/dezh_output.csv \
    -m bleu \
    -w 4 --lowercase
done