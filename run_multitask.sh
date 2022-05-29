export CUDA_VISIBLE_DEVICES=0

# python main.py --dataset coarse-few-nerd --setting multi-task --model ProtoNet --dot
python main.py --dataset fine-few-nerd --setting multi-task --model ProtoNet --dot
# python main.py --dataset stackoverflow --setting multi-task --model ProtoNet --dot
