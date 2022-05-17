export CUDA_VISIBLE_DEVICES=0

python main.py \
        --dataset coarse-few-nerd \
        --protocol online \
        --model ProtoNet \
        --batch_size 32 \
        --val_iter 10 \
        --lr 2e-5