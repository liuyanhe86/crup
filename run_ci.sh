export CUDA_VISIBLE_DEVICES=2

# python main.py \
#         --dataset coarse-few-nerd \
#         --setting CI \
#         --model PCP \
#         --dot \
#         --only_test

python main.py \
        --dataset  coarse-few-nerd \
        --setting CI \
        --model ProtoNet \
        --lr 5e-5 \
        --proto_update replace \
        --metric dot

# python main.py \
#         --dataset stackoverflow \
#         --setting CI \ 
#         --model ProtoNet \
#         --dot \