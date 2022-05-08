export CUDA_VISIBLE_DEVICES=0

python main.py \
        --dataset coarse-few-nerd \
        --protocol CI \
        --model CPR \
        --use_sgd \
        --lr 1e-2 \
        # --only_test
# python main.py --dataset fine-few-nerd --protocol CI --model ProtoNet --dot --use_sgd
# python main.py --dataset stackoverflow --protocol CI --model ProtoNet --dot --use_sgd