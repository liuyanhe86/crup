import datetime
import logging
import os

from transformers import BertTokenizer
from util.args import TypedArgumentParser
from util.datautils import ContinualNerDataset, MultiNerDataset, NerDataset
from util.episode import ContinualNerEpisode, OnlineNerEpisode, SupConNerEpisode, SupNerEpisode
from util.tasks import PROTOCOLS

logger = logging.getLogger(__name__)

def get_dataset_path(dataset):
    if dataset == 'few-nerd':
        return 'data/few-nerd'
    elif dataset == 'coarse-few-nerd':
        return 'data/few-nerd/coarse'
    elif dataset == 'fine-few-nerd':
        return 'data/few-nerd/fine'
    elif dataset == 'stackoverflow':
        return 'data/stackoverflow'
    else:
        raise ValueError(f'Unknown dataset {dataset}!')

class SupervisedSetting:

    def execute(self, args:TypedArgumentParser, ckpt=None):
        tokenizer = BertTokenizer.from_pretrained(args.pretrain_ckpt)
        dataset_path = get_dataset_path(args.dataset)
        logger.info('loading data...')
        train_dataset = NerDataset(os.path.join(dataset_path, 'supervised/train.txt'), tokenizer, max_length=args.max_length)
        val_dataset = NerDataset(os.path.join(dataset_path, 'supervised/dev.txt'), tokenizer, max_length=args.max_length)
        test_dataset = NerDataset(os.path.join(dataset_path, 'supervised/test.txt'), tokenizer, max_length=args.max_length)
        logger.info(f'train size: {len(train_dataset)}, val size: {len(val_dataset)}, test size: {len(test_dataset)}')
        if args.augment:
            sup_episode = SupConNerEpisode(args, train_dataset, val_dataset, test_dataset)
        else:
            sup_episode = SupNerEpisode(args, train_dataset, val_dataset, test_dataset)
        if not args.only_test:
            sup_episode.train(load_ckpt=args.load_ckpt, save_ckpt=ckpt)
        # test
        precision, recall, f1, fp, fn, within, outer = sup_episode.eval(ckpt=ckpt)
        logger.info('RESULT: precision: %.4f, recall: %.4f, f1: %.4f' % (precision, recall, f1))
        logger.info('ERROR ANALYSIS: fp: %.4f, fn: %.4f, within: %.4f, outer: %.4f'%(fp, fn, within, outer))

class CiSetting:
    def execute(self, args: TypedArgumentParser, ckpt=None):
        logger.info('loading data...')
        ci_tasks = PROTOCOLS[args.setting + ' ' + args.dataset]
        task_id = 0
        label_offset = 0
        tokenizer = BertTokenizer.from_pretrained(args.pretrain_ckpt)
        dataset_path = get_dataset_path(args.dataset)
        result_dict = {task : {'precision': [], 'recall': [], 'f1': [], 'fp_error': [], 'fn_error':[], 'within_error':[], 'outer_error':[]} for task in ci_tasks}
        continual_episode = ContinualNerEpisode(args, result_dict)
        for task in ci_tasks:
            train_dataset = ContinualNerDataset(os.path.join(dataset_path, ci_tasks[task], 'train.txt'), tokenizer, augment=args.augment, label_offset=label_offset, max_length=args.max_length)
            val_dataset = ContinualNerDataset(os.path.join(dataset_path, ci_tasks[task], 'dev.txt'), tokenizer, label_offset=label_offset, max_length=args.max_length)
            test_dataset = ContinualNerDataset(os.path.join(dataset_path,  ci_tasks[task], 'test.txt'), tokenizer, label_offset=label_offset, max_length=args.max_length)
            logger.info(f'[TASK {task}] train size: {len(train_dataset)}, val size: {len(val_dataset)}, test size: {len(test_dataset)}')
            label_offset += len(train_dataset.classes)
            continual_episode.append_task(task, train_dataset, val_dataset, test_dataset)
            if not args.only_test and task_id >= args.start_task:
                load_ckpt = None
                if task_id > 0:
                    load_ckpt = ckpt
                continual_episode.train(load_ckpt=load_ckpt, save_ckpt=ckpt)
                # normal test
                continual_episode.eval(ckpt=ckpt)
            else:
                # only test
                continual_episode.eval(ckpt=ckpt)
            task_id += 1
            logger.info(f'[PROGRESS] ({task_id}/{len(ci_tasks)})')
        logger.info('CI finished!')
        output_continual_results(args, result_dict)
    
class OnlineSetting:
    def execute(self, args: TypedArgumentParser, ckpt):
        logger.info('loading data...')
        online_tasks = PROTOCOLS[args.setting + ' ' + args.dataset]
        tokenizer = BertTokenizer.from_pretrained(args.pretrain_ckpt)
        dataset_path = get_dataset_path(args.dataset)
        online_dataset = MultiNerDataset(tokenizer=tokenizer, augment=args.augment, max_length=args.max_length)
        label_offset = 0
        splits = {'train.txt', 'dev.txt', 'test.txt'}
        for task in online_tasks:
            for split in splits:
                file_path = os.path.join(dataset_path,  online_tasks[task], split)
                offset = online_dataset.append(file_path=file_path, label_offset=label_offset)
                label_offset += offset
        logger.info(f'online dataset size: {len(online_dataset)}')
        online_episode = OnlineNerEpisode(args, online_dataset)
        online_episode.learn(save_ckpt=ckpt)
        logger.info('Online finished!')

class MultiTaskSetting:
    def execute(self, args: TypedArgumentParser, ckpt):
        logger.info('loading data...')
        multi_task_pathes = PROTOCOLS[args.setting + ' ' + args.dataset]
        label_offset = 0
        tokenizer = BertTokenizer.from_pretrained(args.pretrain_ckpt)
        dataset_path = get_dataset_path(args.dataset)
        train_dataset = MultiNerDataset(tokenizer, max_length=args.max_length)  
        task_id = 0
        result_dict = {task : {'precision': [], 'recall': [], 'f1': [], 'fp_error': [], 'fn_error':[], 'within_error':[], 'outer_error':[]} for task in multi_task_pathes}
        continual_episode = ContinualNerEpisode(args, result_dict)
        for task in multi_task_pathes:
            label_set_size = train_dataset.append(os.path.join(dataset_path, multi_task_pathes[task], 'train.txt'), label_offset)
            val_dataset = ContinualNerDataset(os.path.join(dataset_path, multi_task_pathes[task], 'dev.txt'), label_offset)
            test_dataset = ContinualNerDataset(os.path.join(dataset_path, multi_task_pathes[task], 'test.txt'), label_offset)
            logger.info(f'[TASK {task}] train size: {len(train_dataset)}, val size: {len(val_dataset)}, test size: {len(test_dataset)}')
            continual_episode.append_task(task, train_dataset, val_dataset, test_dataset)
            if not args.only_test:
                load_ckpt = None
                if task_id > 0:
                    load_ckpt = ckpt
                continual_episode.train(load_ckpt=load_ckpt, save_ckpt=ckpt)
                # test
                continual_episode.eval(ckpt=ckpt)
            else:
                # test
                continual_episode.eval(ckpt=ckpt)
            label_offset += label_set_size
            task_id += 1
            logger.info(f'[PROGRESS] ({task_id}/{len(multi_task_pathes)})')
        logger.info('Multi-task finished!')
        output_continual_results(args, result_dict)

def output_continual_results(args: TypedArgumentParser, result_dict):       
    output_dir = f'output/{args.setting}_{args.dataset}_{args.model}'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    with open(os.path.join(output_dir, 'precision'), 'a') as file:
        file.write(str(datetime.datetime.now()) + '\n')
        file.write(f'learning rate: {args.lr}; use cp loss: {args.contrast_proto}; use proj: {args.use_proj}; use dot: {args.dot};')
        for task in result_dict:
            task_precision = ','.join([str(_) for _ in result_dict[task]['precision']])
            file.write(task_precision + '\n')
    with open(os.path.join(output_dir, 'f1'), 'a') as file:
        file.write(str(datetime.datetime.now()) + '\n')
        file.write(f'learning rate: {args.lr}; use cp loss: {args.contrast_proto}; use proj: {args.use_proj}; use dot: {args.dot};')
        for task in result_dict:
            task_f1 = ','.join([str(_) for _ in result_dict[task]['f1']])
            file.write(task_f1 + '\n')
