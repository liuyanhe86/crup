export CUDA_VISIBLE_DEVICES=2

# python main.py \
#         --dataset few-nerd \
#         --setting sup \
#         --model Bert-Tagger

# python main.py \
#         --dataset few-nerd \
#         --setting sup \
#         --model ProtoNet \
#         --use_sgd \
#         --dot

python main.py \
        --dataset few-nerd \
        --setting sup \
        --model PCP \
        --augment \
        --temperature 0.1