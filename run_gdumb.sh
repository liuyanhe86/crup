export CUDA_VISIBLE_DEVICES=2

python main.py \
        --dataset stackoverflow \
        --setting CI \
        --model GDumb  \
        --lr 2e-5 \
        --gdumb_size 1000