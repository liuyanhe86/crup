export CUDA_VISIBLE_DEVICES=2

# python main.py \
#         --dataset fine-few-nerd \
#         --setting sup \
#         --model Bert-Tagger \
#         --lr 2e-5 \
#         --optimizer AdamW

# python main.py \
#         --dataset coarse-few-nerd \
#         --setting sup \
#         --model ProtoNet \
#         --proto_update SDC \
#         --lr 1e-4 \
#         --optimizer AdamW


python main.py \
        --dataset coarse-few-nerd \
        --setting sup \
        --model PCP \
        --augment remove \
        --lr 2e-5 \
        --optimizer AdamW  \
        --embedding_dimension 64
