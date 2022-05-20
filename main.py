import argparse
import datetime
import logging
import os
import random
import numpy as np
import torch
import transformers

from util.settings import CiSetting, MultiTaskSetting, OnlineSetting, SupervisedSetting

def get_args():
    parser = argparse.ArgumentParser(description="Continual NER")
    parser.add_argument('--dataset', dest="dataset", type=str, default='few-nerd',
            help='dataset name, must be in [few-nerd, stackoverflow]')
    parser.add_argument('--setting', dest="setting", type=str, default='sup',
            help='continual learning setting, must be in [sup, CI, online, multi-task]')
    parser.add_argument('--model', dest="model", type=str, default='ProtoNet',
            help='model name, must be in [CPR, ProtoNet, BERT-Tagger]')
    parser.add_argument('--batch_size', default=10, type=int,
            help='batch size')
    parser.add_argument('--train_epoch', default=10, type=int,
            help='num of iters in training')
    parser.add_argument('--val_step', default=2, type=int,
            help='val after training how many iters')
#     parser.add_argument('--warmup_step', default=300, type=int,
#             help='warm up steps before training')
    parser.add_argument('--max_length', default=50, type=int,
            help='max length')
    parser.add_argument('--lr', default=2e-5, type=float,
            help='learning rate')
    parser.add_argument('--load_ckpt', default=None,
            help='load ckpt')
    parser.add_argument('--save_ckpt', default=None,
            help='save ckpt')
    parser.add_argument('--fp16', action='store_true',
            help='use nvidia apex fp16')
    parser.add_argument('--ckpt_name', type=str, default='',
            help='checkpoint name.')
    parser.add_argument('--random_seed', type=int, default=0,
            help='random seed')
    parser.add_argument('--only_test', action='store_true',
            help='only test model with checkpoint')
    parser.add_argument('--start_task', type=int, default=0,
            help='continual task id of beginning task')
    # only for bert / roberta
    parser.add_argument('--pretrain_ckpt', default='bert-base-uncased',
            help='bert / roberta pre-trained checkpoint')
    parser.add_argument('--dot', action='store_true', 
            help='use dot instead of L2 distance for proto')
    # only for CPR
    parser.add_argument('--temperature', type=float, default=0.1, 
            help='temperature for supervised contrastive loss')
    parser.add_argument('--cpl', action='store_true', 
            help='whether use contrastive prototype loss')
    # experiment
    parser.add_argument('--use_sgd', action='store_true',
            help='use SGD instead of AdamW for BERT.')
    
    args = parser.parse_args()
    return args

def init_logging(filename):
    transformers.logging.set_verbosity_error()
    class TimeFilter(logging.Filter):
        def filter(self, record):
            try:
                last = self.last
            except AttributeError:
                last = record.relativeCreated

            delta = record.relativeCreated/1000 - last/1000
            record.relative = "{:.1f}".format(delta)
            record.uptime = str(datetime.timedelta(seconds=record.relativeCreated//1000))
            self.last = record.relativeCreated
            return True
    
    logging_format = "%(asctime)s - %(uptime)s - %(relative)ss - %(levelname)s - %(name)s - %(message)s"
    logging.basicConfig(format=logging_format, filename=filename, filemode='a', level=logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(logging_format))
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)
    for handler in root_logger.handlers:
        handler.addFilter(TimeFilter())

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    

def main():
    args = get_args()
    set_seed(args.random_seed)
    if not os.path.exists('log'):
        os.mkdir('log')
    init_logging(f'log/{args.setting}_{args.dataset}_{args.model}{"_cpl" if args.cpl else ""}{"_dot" if args.dot else ""}.log')   
    logger = logging.getLogger(__name__)
    logger.info(f'PID: {os.getpid()}; PPID: {os.getppid()}')

    logger.info(f'EXP CONFIG: model: {args.model}, dataset: {args.dataset}, setting: {args.setting}, batch_size: {args.batch_size}, train_epoch: {args.train_epoch}, val_step: {args.val_step}, learning_rate: {args.lr}, use_sgd: {args.use_sgd}')
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    ckpt = f'checkpoint/{args.setting}_{args.dataset}_{args.model}{"_cpl" if args.cpl else ""}{"_dot" if args.dot else ""}.pth.tar'
    logger.info(f'model-save-path: {ckpt}')

    if args.setting == 'sup':
        setting = SupervisedSetting()
        setting.execute(args, ckpt=ckpt)
    
    elif args.setting == 'CI':
        setting = CiSetting()
        setting.execute(args, ckpt=ckpt)

    elif args.setting == 'online':
        setting = OnlineSetting()
        setting.execute(args, ckpt=ckpt)

    elif args.setting == 'multi-task':
        setting = MultiTaskSetting()
        setting.execute(args, ckpt=ckpt)
    
    else:
        raise NotImplementedError(f'{args.setting} has not been implemented!')
        

if __name__ == '__main__':
    main()