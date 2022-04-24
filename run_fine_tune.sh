export CUDA_VISIBLE_DEVICES=0

# python main.py --dataset few-nerd --protocol fine-tune --model ProtoNet --dot --use_sgd
python main.py --dataset few-nerd --protocol fine-tune --model Bert-Tagger --use_sgd --lr 1e-2

# python main.py --dataset few-nerd --protocol fine-tune-fine --model ProtoNet --dot --use_sgd
# python main.py --dataset stackoverflow --protocol fine-tune --model ProtoNet --dot --use_sgd