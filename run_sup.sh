export CUDA_VISIBLE_DEVICES=2

python main.py \
        --dataset few-nerd \
        --setting sup \
        --model Bert-Tagger \
        --lr 5e-3 \
        --use_sgd \
        --train_encoder

# python main.py \
#         --dataset few-nerd \
#         --setting sup \
#         --model ProtoNet \
#         --lr 2e-5 \
#         --dot

# python main.py \
#         --dataset few-nerd \
#         --setting sup \
#         --model PCP \
#         --augment \
        # --train_encoder