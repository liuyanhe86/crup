import argparse
import datetime
import logging
import os
import random
import numpy as np
import torch
import transformers

from util.args import TypedArgumentParser
from util.settings import CiSetting, MultiTaskSetting, OnlineSetting, SupervisedSetting


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
    args = TypedArgumentParser().parse_args()
    set_seed(args.random_seed)
    if not os.path.exists('log'):
        os.mkdir('log')
    init_logging(f'log/{args.setting}_{args.dataset}_{args.model}{"_dot" if args.dot else ""}.log')   
    logger = logging.getLogger(__name__)
    logger.info(f'PID: {os.getpid()}; PPID: {os.getppid()}')

    logger.info(f'EXP CONFIG: {args}')
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    ckpt = f'checkpoint/{args.setting}_{args.dataset}_{args.model}{"_dot" if args.dot else ""}.pth.tar'
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