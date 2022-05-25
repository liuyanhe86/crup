export CUDA_VISIBLE_DEVICES=2

# python main.py \
#         --dataset few-nerd \
#         --setting sup \
#         --model Bert-Tagger \
#         --batch_size 32 \
#         --use_sgd \
#         --lr 5e-3

# python main.py \
#         --dataset few-nerd \
#         --setting sup \
#         --model ProtoNet \
#         --batch_size 32 \
#         --use_sgd \
#         --lr 2e-5 \
#         --dot

python main.py \
        --dataset few-nerd \
        --setting sup \
        --model PCP \
        --batch_size 32 \
        --augment \
        --lr 2e-5 \
        --temperature 0.1
        # --dot
