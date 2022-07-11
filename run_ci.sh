export CUDA_VISIBLE_DEVICES=0

# python main.py \
#         --dataset coarse-few-nerd \
#         --setting CI \
#         --model PCP \
#         --dot \
#         --only_test

python main.py \
        --dataset  stackoverflow \
        --setting CI \
        --model ProtoNet \
        --lr 3e-4 \
        --proto_update replace \
        --metric dot

# python main.py \
#         --dataset stackoverflow \
#         --setting CI \ 
#         --model ProtoNet \
#         --dot \