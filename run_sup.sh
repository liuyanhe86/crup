export CUDA_VISIBLE_DEVICES=1

# python main.py \
#         --dataset stackoverflow \
#         --setting sup \
#         --model Bert-Tagger \
#         --lr 5e-3 \
#         --use_sgd \
#         --train_encoder

python main.py \
        --dataset coarse-few-nerd \
        --setting sup \
        --model ProtoNet \
        --proto_update SDC \
        --lr 2e-5 \
        --metric L2

# python main.py \
#         --dataset few-nerd \
#         --setting sup \
#         --model PCP \
#         --augment \
#         --lr 0.1 \
#         --only_train_encoder