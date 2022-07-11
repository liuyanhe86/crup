export CUDA_VISIBLE_DEVICES=1

python main.py \
        --dataset stackoverflow \
        --setting CI \
        --model ExtendNER  \
        --lr 3e-4