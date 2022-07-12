export CUDA_VISIBLE_DEVICES=2

python main.py \
        --dataset fine-few-nerd \
        --setting CI \
        --model GDumb  \
        --lr 5e-3 \
        --gdumb_size 3000