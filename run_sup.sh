export CUDA_VISIBLE_DEVICES=0

# python main.py \
#         --dataset fine-few-nerd \
#         --setting sup \
#         --model Bert-Tagger \
#         --lr 1e-2

# python main.py \
#         --dataset stackoverflow \
#         --setting sup \
#         --model ProtoNet \
#         --proto_update SDC \
#         --lr 1e-2 \
#         --metric dot

python main.py \
        --dataset coarse-few-nerd \
        --setting sup \
        --model PCP \
        --batch_size 1 \
        --augment remove \
        --lr 0.1 \
        --only_train_encoder \
        --use_sgd