export CUDA_VISIBLE_DEVICES=0

# python main.py \
#         --dataset few-nerd \
#         --protocol sup \
#         --model Bert-Tagger \
#         --batch_size 32 \
#         --use_sgd \
#         --lr 5e-3

python main.py \
        --dataset few-nerd \
        --protocol sup \
        --model ProtoNet \
        --batch_size 32 \
        --use_sgd \
        --lr 2e-5 \
        --dot

python main.py \
        --dataset few-nerd \
        --protocol sup \
        --model CPR \
        --batch_size 32 \
        --use_sgd \
        --lr 5e-3 \
#         --dot