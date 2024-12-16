for n in 11 12 13 14 15 16;
do
   echo ${n}
   CUDA_VISIBLE_DEVICES=6 comet-score -s /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/enzh/src.txt \
-t /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/sn/greedy_decode/layer/enzh_sft_layer${n}_deepspeed/2000steps/enzh_output.csv \
-r /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/enzh/ref.txt \
--model_storage_path /mnt/nfs/algo/intern/haoyunx11/models/comet/wmt22-comet-da --model Unbabel/wmt22-comet-da
done