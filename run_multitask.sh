export CUDA_VISIBLE_DEVICES=2

python main.py --dataset few-nerd --protocol multi-task --model ProtoNet --dot --use_sgd
# python main.py --dataset stackoverflow --protocol fine-tune --model ProtoNet --dot --use_sgd