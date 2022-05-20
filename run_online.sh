export CUDA_VISIBLE_DEVICES=0

python main.py \
        --dataset coarse-few-nerd \
        --setting online \
        --model ProtoNet \
        --batch_size 32 \
        --val_step 10 \
        --lr 2e-5