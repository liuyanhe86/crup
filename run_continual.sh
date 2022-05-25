export CUDA_VISIBLE_DEVICES=0

# python main.py \
#         --dataset coarse-few-nerd \
#         --setting CI \
#         --model PCP \
#         --use_sgd \
#         --lr 5e-2 \
#         --contrast_proto
#         --dot \
#         --only_test
python main.py \
        --dataset  coarse-few-nerd \
        --setting CI \
        --model ProtoNet \
        --dot \
        --use_sgd
# python main.py \
#         --dataset stackoverflow \
#         --setting CI \ 
#         --model ProtoNet \
#         --dot \
#         --use_sgd