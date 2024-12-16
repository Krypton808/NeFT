for n in 500 1000 1500 2500;
do
   echo ${n}
   CUDA_VISIBLE_DEVICES=7 comet-score -s /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/dezh/src.txt \
-t /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/infer/output/sn/greedy_decode/train_neuron/dezh/enzh_train_neuron_cos_score_0.9995/${n}steps/dezh_output.csv \
-r /data5/haoyun.xu/study/MT/Finetune_LLM_for_MT/train/data/wmt19/flores_101/dezh/ref.txt \
--model_storage_path /mnt/nfs/algo/intern/haoyunx11/models/comet/wmt22-comet-da --model Unbabel/wmt22-comet-da
done