export CUDA_VISIBLE_DEVICES=2

python main.py \
        --dataset coarse-few-nerd \
        --setting CI \
        --model AddNER  \
        --lr 5e-3