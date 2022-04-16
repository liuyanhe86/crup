import argparse
import datetime
import logging
import os
import random
import numpy as np
import torch
import transformers
from transformers import BertTokenizer

import util.datautils as datautils
from util.frameworks import SupNERFramework, ContinualNERFramework
from util.tasks import PROTOCOLS
from model import BERTWordEncoder, NERModel, BertTagger, ProtoNet

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
    logging.basicConfig(format=logging_format, filename=filename, filemode='w', level=logging.INFO)
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
    
def get_model_name(model: NERModel):
    if isinstance(model, BertTagger):
        return 'BERT-Tagger'
    elif isinstance(model, ProtoNet):
        return 'ProtoNet'
    else:
        raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description="Continual NER")
    parser.add_argument('--dataset', dest="dataset", type=str, default='few-nerd',
            help='dataset name, must be in [few-nerd, stackoverflow]')
    parser.add_argument('--protocol', dest="protocol", type=str, default='sup',
            help='continual learning protocol, must be in [sup, fine-tune, CI, online, multi-task]')
    parser.add_argument('--model', dest="model", type=str, default='ProtoNet',
            help='model name, must be in [CPR, ProtoNet, BERT-Tagger]')
    parser.add_argument('--batch_size', default=10, type=int,
            help='batch size')
    parser.add_argument('--train_iter', default=10000, type=int,
            help='num of iters in training')
    parser.add_argument('--val_iter', default=1000, type=int,
            help='num of iters in validation')
    parser.add_argument('--test_iter', default=500, type=int,
            help='num of iters in testing')
    parser.add_argument('--val_step', default=2000, type=int,
           help='val after training how many iters')
    parser.add_argument('--max_length', default=50, type=int,
           help='max length')
    parser.add_argument('--lr', default=2e-5, type=float,
           help='learning rate')
    parser.add_argument('--grad_iter', default=1, type=int,
           help='accumulate gradient every x iterations')
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
    # only for bert / roberta
    parser.add_argument('--pretrain_ckpt', default='bert-base-uncased',
           help='bert / roberta pre-trained checkpoint')
    # only for prototypical networks
    parser.add_argument('--dot', action='store_true', 
           help='use dot instead of L2 distance for proto')
    # experiment
    parser.add_argument('--use_sgd', action='store_true',
           help='use SGD instead of AdamW for BERT.')
    
    args = parser.parse_args()
    set_seed(args.random_seed)
    
    # now = datetime.datetime.now()
    # {str(now.year)[-2:]}_{now.month}_{now.day}_{now.hour}_{now.minute}
    init_logging(f'log/{args.protocol}_{args.dataset}_{args.model}.log')
    logger = logging.getLogger(__name__)
    logger.info(f'PID: {os.getpid()}; PPID: {os.getppid()}')
    logger.info('loading model and tokenizer...')
    pretrain_ckpt = args.pretrain_ckpt
    word_encoder = BERTWordEncoder(pretrain_ckpt)
    tokenizer = BertTokenizer.from_pretrained(pretrain_ckpt)
    
    if args.protocol == 'sup':
        model = ProtoNet(word_encoder, dot=args.dot)
        logger.info('loading data...')
        dataset_path = os.path.join('data', args.dataset)
        
        train_dataset = datautils.NERDataset(os.path.join(dataset_path, 'supervised/train.txt'), tokenizer, max_length=args.max_length)
        valid_dataset = datautils.NERDataset(os.path.join(dataset_path, 'supervised/dev.txt'), tokenizer, max_length=args.max_length)
        test_dataset = datautils.NERDataset(os.path.join(dataset_path, 'supervised/test.txt'), tokenizer, max_length=args.max_length)
        
        train_data_loader = datautils.get_loader(train_dataset, batch_size=args.batch_size)
        val_data_loader = datautils.get_loader(valid_dataset, batch_size=args.batch_size)
        test_data_loader = datautils.get_loader(test_dataset, batch_size=args.batch_size)

        sup_framework = SupNERFramework(train_data_loader, val_data_loader, test_data_loader)

        if not os.path.exists('checkpoint'):
            os.mkdir('checkpoint')
        ckpt = f'checkpoint/{args.protocol}_{args.dataset}_{get_model_name(model)}.pth.tar'
        
        logger.info(f'model-save-path: {ckpt}')

        if torch.cuda.is_available():
            model.cuda()
    
        logger.info(f'TRAIN CONFIG: model: {get_model_name(model)}, dataset: {dataset_path}, batch_size: {args.batch_size}, train_iter: {args.train_iter}, val_iter: {args.val_iter}, val_step: {args.val_step}, learning_rate: {args.lr}, use_sgd: {args.use_sgd}')
        sup_framework.train(model, 
                            get_model_name(model),
                            load_ckpt=args.load_ckpt, 
                            save_ckpt=ckpt,
                            train_iter=args.train_iter,
                            val_iter=args.val_iter,
                            val_step=args.val_step,
                            learning_rate=args.lr,
                            grad_iter=args.grad_iter,
                            fp16=args.fp16,
                            use_sgd_for_bert=args.use_sgd)

        # test
        precision, recall, f1, fp, fn, within, outer = sup_framework.eval(model, eval_iter=args.test_iter, ckpt=ckpt)
        logger.info('RESULT: precision: %.4f, recall: %.4f, f1: %.4f' % (precision, recall, f1))
        logger.info('ERROR ANALYSIS: fp: %.4f, fn: %.4f, within: %.4f, outer: %.4f'%(fp, fn, within, outer))
    
    elif args.protocol == 'fine-tune':
        model = ProtoNet(word_encoder, dot=args.dot)
        logger.info('loading data...')
        dataset_path = os.path.join('data', args.dataset)
        fine_tune_task_pathes = PROTOCOLS[args.protocol + ' ' + args.dataset]
        if not os.path.exists('checkpoint'):
            os.mkdir('checkpoint')
        ckpt = f'checkpoint/{args.protocol}_{args.dataset}_{get_model_name(model)}.pth.tar'
        logger.info(f'model-save-path: {ckpt}')
        task_id = 0
        label_offset = 0
        test_loaders = {}
        
        logger.info(f'TRAIN CONFIG: model: {get_model_name(model)}, dataset: {dataset_path}, batch_size: {args.batch_size}, train_iter: {args.train_iter}, val_iter: {args.val_iter}, val_step: {args.val_step}, learning_rate: {args.lr}, use_sgd: {args.use_sgd}')
        
        result_dict = {task : {'precision': [], 'recall': [], 'f1': [], 'fp_error': [], 'fn_error':[], 'within_error':[], 'outer_error':[]} for task in fine_tune_task_pathes}
        for task in fine_tune_task_pathes:
            logger.info(f'start training [TASK] {task}')
            train_dataset = datautils.ContinualNERDataset(os.path.join(dataset_path, fine_tune_task_pathes[task], 'train.txt'), tokenizer, label_offset=label_offset, max_length=args.max_length)
            valid_dataset = datautils.ContinualNERDataset(os.path.join(dataset_path, fine_tune_task_pathes[task], 'dev.txt'), tokenizer, label_offset=label_offset, max_length=args.max_length)
            test_dataset = datautils.ContinualNERDataset(os.path.join(dataset_path,  fine_tune_task_pathes[task], 'test.txt'), tokenizer, label_offset=label_offset, max_length=args.max_length)
            label_offset += len(train_dataset.classes)

            train_data_loader = datautils.get_loader(train_dataset, batch_size=args.batch_size)
            val_data_loader = datautils.get_loader(valid_dataset, batch_size=args.batch_size)
            test_data_loader = datautils.get_loader(test_dataset, batch_size=args.batch_size)
            test_loaders[task] = test_data_loader
            
            continual_framework = ContinualNERFramework(train_data_loader, val_data_loader, test_loaders)

            if torch.cuda.is_available():
                model.cuda()

            load_ckpt = None
            if task_id > 0:
                load_ckpt = ckpt
            continual_framework.train(model, 
                            get_model_name(model),
                            load_ckpt=load_ckpt, 
                            save_ckpt=ckpt,
                            train_iter=args.train_iter,
                            val_iter=args.val_iter,
                            val_step=args.val_step,
                            learning_rate=args.lr,
                            grad_iter=args.grad_iter,
                            fp16=args.fp16,
                            use_sgd_for_bert=args.use_sgd)

            # test
            continual_framework.eval(model, result_dict, eval_iter=args.test_iter, ckpt=ckpt)
            task_id += 1
        logger.info('Fine-tune finished successfully!')
        for task in result_dict:
            evals = result_dict[task]
            result = '\n'.join([f'{item}:{value}' for item, value in evals.items()])
            logger.info(f'[RESULT] | {task}\n{result}')

    elif args.protocol == 'multi-task':
        model = ProtoNet(word_encoder, dot=args.dot)
        logger.info('loading data...')
        dataset_path = os.path.join('data', args.dataset)
        multi_task_pathes = PROTOCOLS[args.protocol + ' ' + args.dataset]
        if not os.path.exists('checkpoint'):
            os.mkdir('checkpoint')
        ckpt = f'checkpoint/{args.protocol}_{args.dataset}_{get_model_name(model)}.pth.tar'
        
        logger.info(f'model-save-path: {ckpt}')
        logger.info(f'TRAIN CONFIG: model: {get_model_name(model)}, dataset: {dataset_path}, batch_size: {args.batch_size}, train_iter: {args.train_iter}, val_iter: {args.val_iter}, val_step: {args.val_step}, learning_rate: {args.lr}, use_sgd: {args.use_sgd}')
        
        label_offset = 0
        train_dataset = datautils.MultiNERDataset(tokenizer, max_length=args.max_length)
        for task in multi_task_pathes:
            label_offset += train_dataset.append(os.path.join(dataset_path, multi_task_pathes[task], 'train.txt'), label_offset)
        train_data_loader = datautils.get_loader(train_dataset, batch_size=args.batch_size)
        
        label_offset = 0
        valid_dataset = datautils.MultiNERDataset(tokenizer, max_length=args.max_length)
        for task in multi_task_pathes:
            label_offset += valid_dataset.append(os.path.join(dataset_path, multi_task_pathes[task], 'dev.txt'), label_offset)
        val_data_loader = datautils.get_loader(valid_dataset, batch_size=args.batch_size)
        
        task_id = 0
        test_data_loaders = {}
        result_dict = {task : {'precision': [], 'recall': [], 'f1': [], 'fp_error': [], 'fn_error':[], 'within_error':[], 'outer_error':[]} for task in multi_task_pathes}
        for task in multi_task_pathes:
            logger.info(f'start training [EPOCH] {task_id + 1}')
            continual_framework = ContinualNERFramework(train_data_loader, val_data_loader, None)
            if torch.cuda.is_available():
                model.cuda()
            load_ckpt = None
            if task_id > 0:
                load_ckpt = ckpt
            continual_framework.train(model, 
                            get_model_name(model),
                            load_ckpt=load_ckpt, 
                            save_ckpt=ckpt,
                            train_iter=args.train_iter,
                            val_iter=args.val_iter,
                            val_step=args.val_step,
                            learning_rate=args.lr,
                            grad_iter=args.grad_iter,
                            fp16=args.fp16,
                            use_sgd_for_bert=args.use_sgd)
            task_id += 1
        label_offset = 0
        for task in multi_task_pathes:
            logger.info(f'start training [TASK] {task}')
            test_dataset = datautils.ContinualNERDataset(os.path.join(dataset_path,  multi_task_pathes[task], 'test.txt'), tokenizer, label_offset=label_offset, max_length=args.max_length)
            label_offset += len(test_dataset.classes)
            test_data_loader = datautils.get_loader(test_dataset, batch_size=args.batch_size)
            test_data_loaders[task] = test_data_loader
            continual_framework = ContinualNERFramework(None, None, test_data_loaders)
            # test
            continual_framework.eval(model, result_dict, eval_iter=args.test_iter, ckpt=ckpt)
        logger.info('Multi-task finished successfully!')
        for task in result_dict:
            evals = result_dict[task]
            result = '\n'.join([f'{item}:{value}' for item, value in evals.items()])
            logger.info(f'[RESULT] | {task}\n{result}')
        

if __name__ == '__main__':
    main()