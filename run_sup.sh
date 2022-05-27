export CUDA_VISIBLE_DEVICES=1

# python main.py \
#         --dataset few-nerd \
#         --setting sup \
#         --model Bert-Tagger \
#         --lr 2e-5

# python main.py \
#         --dataset few-nerd \
#         --setting sup \
#         --model ProtoNet \
#         --use_sgd \
#         --lr 2e-5 \
#         --dot

python main.py \
        --dataset few-nerd \
        --setting sup \
        --model PCP \
        --augment \
        --lr 2e-5 \
        --temperature 0.1