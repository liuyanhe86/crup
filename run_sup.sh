export CUDA_VISIBLE_DEVICES=0

python main.py \
        --dataset few-nerd \
        --protocol sup \
        --model CPR \
        --use_sgd \
        --lr 1e-2