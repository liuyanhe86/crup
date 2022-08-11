export CUDA_VISIBLE_DEVICES=1

python main.py \
        --dataset coarse-few-nerd \
        --setting sup \
        --model PCP \
        --batch_size 16\
        --lr 2e-5 \
        --optimizer AdamW
        # --augment remove \