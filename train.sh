for id in 0 1 2
do
    CUDA_VISIBLE_DEVICES=0 python  run.py \
    --bert_model 'bert-base-uncased' \
    --experiment mix \
    --note random$id \
    --idrandom $id \
    --approach taskdrop
done
