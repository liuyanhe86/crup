export CUDA_VISIBLE_DEVICES=0

python main.py --dataset coarse-few-nerd --protocol multi-task --model ProtoNet --dot --use_sgd
# python main.py --dataset stackoverflow --protocol multi-task --model ProtoNet --dot --use_sgd
# python main.py --dataset fine-few-nerd --protocol multi-task --model ProtoNet --dot --use_sgd