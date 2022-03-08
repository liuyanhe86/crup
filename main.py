import argparse
import random
import numpy as np
import torch


import config.config as config


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser(description="Continual NER")
    parser.add_argument("--config_file", dest="config_file", type=str, default="config/config.cfg",
            help="config path")
    parser.add_argument("--device", dest="device", type=str, default="cuda:0", 
            help="device['cpu', 'cuda:0', 'cuda:1', ......]")
    parser.add_argument('--model', dest="model", type=str, default='Bert-Tagger',
            help='model name, must be in [BiLSTM-CRF, BiLSTM-CNN-CRF, Bert-Tagger]')
    parser.add_argument('--dataset', dest="dataset", type=str, default='few-nerd',
            help='dataset name, must be in [few-nerd, stackoverflow]')
    
    args = parser.parse_args()
    opt = config.Configurable(args.config_file)
    set_seed(opt.random_seed)
