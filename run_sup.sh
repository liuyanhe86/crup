export CUDA_VISIBLE_DEVICES=0

# python main.py \
#         --dataset few-nerd \
#         --protocol sup \
#         --model Bert-Tagger \
#         --use_sgd \
#         --lr 1e-2

python main.py \
        --dataset few-nerd \
        --protocol sup \
        --model ProtoNet \
        --use_sgd \
        --lr 2e-5 \
        # --dot

# python main.py \
#         --dataset few-nerd \
#         --protocol sup \
#         --model CPR \
#         --use_sgd \
#         --lr 0.075 \
        # --only_test