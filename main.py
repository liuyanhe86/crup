import datetime
import logging
import os
import random
import numpy as np
import torch
import transformers

from util.args import TypedArgumentParser
from util.settings import CiSetting, OnlineSetting, SupervisedSetting, GDumb


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
    init_logging(f'log/{args.setting}_{args.dataset}_{args.model}{"_" + args.proto_update if args.model == "ProtoNet" else ""}{"_" + args.metric if args.model == "ProtoNet" else ""}.log')   
    logger = logging.getLogger(__name__)
    logger.info(f'PID: {os.getpid()}; PPID: {os.getppid()}')

    if args.setting == 'sup':
        setting = SupervisedSetting(args)
    elif args.setting == 'CI':
        if args.model == 'GDumb':
            setting = GDumb(args)
        else:
            setting = CiSetting(args)
    elif args.setting == 'online':
        if args.model == 'GDumb':
            setting = GDumb(args)
        else:
            setting = OnlineSetting(args)
    else:
        raise NotImplementedError(f'{args.setting} has not been implemented!')
    setting.run()
        

if __name__ == '__main__':
    main()