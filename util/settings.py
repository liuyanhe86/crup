import datetime
import logging
import os

from transformers import BertTokenizer

from util.args import TypedArgumentParser
from util.datasets import ContinualNerDataset, GDumbSampler, MultiNerDataset, NerDataset, get_loader
from util.episode import ContinualNerEpisode, DistilledContinualNerEpisode, OnlineNerEpisode, SupNerEpisode
from util.tasks import PROTOCOLS

logger = logging.getLogger(__name__)


class Setting:

    def __init__(self, args: TypedArgumentParser) -> None:
        logger.info(f'EXP CONFIG: {args}')
        if not os.path.exists('checkpoint'):
            os.mkdir('checkpoint')
        ckpt = f'checkpoint/{args.setting}_{args.dataset}_{args.model}{"_" + args.proto_update if args.model == "ProtoNet" else ""}{"_" + args.metric if args.model == "ProtoNet" else ""}.pth.tar'
        logger.info(f'model-save-path: {ckpt}')
        self.args = args
        self.ckpt = ckpt

    def run(self):
        raise NotImplementedError

class SupervisedSetting(Setting):

    def run(self):
        tokenizer = BertTokenizer.from_pretrained(self.args.pretrain_ckpt)
        logger.info('loading data...')
        supervised_task = PROTOCOLS[self.args.setting + ' ' + self.args.dataset]
        train_dataset = NerDataset(os.path.join(supervised_task, 'train.txt'), tokenizer, max_length=self.args.max_length, ignore_label_id=self.args.ignore_index)
        val_dataset = NerDataset(os.path.join(supervised_task, 'dev.txt'), tokenizer, max_length=self.args.max_length, ignore_label_id=self.args.ignore_index)
        test_dataset = NerDataset(os.path.join(supervised_task, 'test.txt'), tokenizer, max_length=self.args.max_length, ignore_label_id=self.args.ignore_index)
        logger.info(f'train size: {len(train_dataset)}, val size: {len(val_dataset)}, test size: {len(test_dataset)}')
        sup_episode = SupNerEpisode(self.args, train_dataset, val_dataset, test_dataset)
        sup_episode.initialize_model()
        sup_episode.train(save_ckpt=self.ckpt)
        # test
        precision, recall, f1, fp, fn, within, outer = sup_episode.eval(ckpt=self.ckpt)
        logger.info('RESULT: precision: %.4f, recall: %.4f, f1: %.4f' % (precision, recall, f1))
        logger.info('ERROR ANALYSIS: fp: %.4f, fn: %.4f, within: %.4f, outer: %.4f'%(fp, fn, within, outer))

class CiSetting(Setting):

    def finish_task(self, episode:SupNerEpisode):
        if self.args.model == 'ProtoNet':
            logger.info(f'num of current prototypes: {len(episode.model.global_protos)}')
        elif self.args.model == 'GDumb':
            logger.info(f'gdumb size: {len(self.sampler)}; model output dim: {episode.model.lc.out_features}')

    def run(self):
        ci_tasks = PROTOCOLS[self.args.setting + ' ' + self.args.dataset]
        task_id = 0
        tokenizer = BertTokenizer.from_pretrained(self.args.pretrain_ckpt)
        result_dict = {task : {'precision': [], 'recall': [], 'f1': [], 'fp_error': [], 'fn_error':[], 'within_error':[], 'outer_error':[]} for task in ci_tasks}
        label2tag, tag2label = {0:'O'}, {'O':0}
        if self.args.model in {'AddNER', 'ExtendNER'}:
            continual_episode = DistilledContinualNerEpisode(self.args, result_dict)
        else:
            continual_episode = ContinualNerEpisode(self.args, result_dict)
        for task in ci_tasks:
            logger.info('loading data...')
            train_dataset = ContinualNerDataset(os.path.join(ci_tasks[task], 'train.txt'), tokenizer, augment=self.args.augment, max_length=self.args.max_length, ignore_label_id=self.args.ignore_index)
            val_dataset = ContinualNerDataset(os.path.join(ci_tasks[task], 'dev.txt'), tokenizer, max_length=self.args.max_length, ignore_label_id=self.args.ignore_index)
            test_dataset = ContinualNerDataset(os.path.join(ci_tasks[task], 'test.txt'), tokenizer, max_length=self.args.max_length, ignore_label_id=self.args.ignore_index)
            logger.info(f'[TASK] {task} | train size: {len(train_dataset)}, val size: {len(val_dataset)}, test size: {len(test_dataset)}')
            num_of_existing_labels = len(label2tag)
            label2tag[num_of_existing_labels] = train_dataset.current_class
            tag2label[train_dataset.current_class] = num_of_existing_labels
            train_dataset.set_labelmap(label2tag, tag2label)
            val_dataset.set_labelmap(label2tag, tag2label)
            test_dataset.set_labelmap(label2tag, tag2label)
            load_ckpt = None
            if task_id > 0:
                load_ckpt = self.ckpt
            continual_episode.append_task(task, train_dataset, val_dataset, test_dataset)
            continual_episode.initialize_model(load_ckpt)
            
            if not self.args.only_test and task_id >= self.args.start_task:
                continual_episode.train(save_ckpt=self.ckpt)
                self.finish_task(continual_episode)
                continual_episode.eval(ckpt=self.ckpt)
            else:
                # only test
                continual_episode.eval(ckpt=self.ckpt)
            task_id += 1
            logger.info(f'[PROGRESS] ({task_id}/{len(ci_tasks)})')
        logger.info('CI finished!')
        output_continual_results(self.args, result_dict)
    
class OnlineSetting(Setting):
    def run(self):
        logger.info('loading data...')
        online_tasks = PROTOCOLS[self.args.setting + ' ' + self.args.dataset]
        tokenizer = BertTokenizer.from_pretrained(self.args.pretrain_ckpt)
        online_dataset = MultiNerDataset(tokenizer=tokenizer, augment=self.args.augment, max_length=self.args.max_length, ignore_label_id=self.args.ignore_index)
        label_offset = 0
        splits = {'train.txt', 'dev.txt', 'test.txt'}
        for task in online_tasks:
            for split in splits:
                file_path = os.path.join(online_tasks[task], split)
                offset = online_dataset.append(file_path=file_path, label_offset=label_offset)
                label_offset += offset
        logger.info(f'online dataset size: {len(online_dataset)}')
        online_episode = OnlineNerEpisode(self.args, online_dataset)
        online_episode.learn(save_ckpt=self.ckpt)
        logger.info('Online finished!')


class GDumb(Setting):

    def __init__(self, args: TypedArgumentParser) -> None:
        Setting.__init__(self, args)
        self.sampler = GDumbSampler(size=args.gdumb_size)

    def run(self):
        if self.args.setting == 'CI':
            self.run_ci()
        elif self.args.setting == 'online':
            self.run_online()
        else:
            raise NotImplementedError(f'ERROR: Invalid setting for GDumb - {self.args.setting}')
    
    def finish_task(self, episode:SupNerEpisode):
        if self.args.model == 'ProtoNet':
            logger.info(f'num of current prototypes: {len(episode.model.global_protos)}')
        elif self.args.model == 'GDumb':
            logger.info(f'gdumb size: {len(self.sampler)}; memory used: {self.sampler.samples.__sizeof__() // (8 * 1024)}KB; model output dim: {episode.model.lc.out_features}')

    def run_ci(self):
        ci_tasks = PROTOCOLS[self.args.setting + ' ' + self.args.dataset]
        task_id = 0
        tokenizer = BertTokenizer.from_pretrained(self.args.pretrain_ckpt)
        result_dict = {task : {'precision': [], 'recall': [], 'f1': [], 'fp_error': [], 'fn_error':[], 'within_error':[], 'outer_error':[]} for task in ci_tasks}
        label2tag, tag2label = {0:'O'}, {'O':0}
        continual_episode = ContinualNerEpisode(self.args, result_dict)
        for task in ci_tasks:
            logger.info('loading data...')
            train_dataset = ContinualNerDataset(os.path.join(ci_tasks[task], 'train.txt'), tokenizer, augment=self.args.augment, max_length=self.args.max_length, ignore_label_id=self.args.ignore_index)
            val_dataset = ContinualNerDataset(os.path.join(ci_tasks[task], 'dev.txt'), tokenizer, max_length=self.args.max_length, ignore_label_id=self.args.ignore_index)
            test_dataset = ContinualNerDataset(os.path.join(ci_tasks[task], 'test.txt'), tokenizer, max_length=self.args.max_length, ignore_label_id=self.args.ignore_index)
            logger.info(f'[TASK] {task} | train size: {len(train_dataset)}, val size: {len(val_dataset)}, test size: {len(test_dataset)}')
            num_of_existing_labels = len(label2tag)
            label2tag[num_of_existing_labels] = train_dataset.current_class
            tag2label[train_dataset.current_class] = num_of_existing_labels
            train_dataset.set_labelmap(label2tag, tag2label)
            val_dataset.set_labelmap(label2tag, tag2label)
            test_dataset.set_labelmap(label2tag, tag2label)

            self.sampler.sample_ci(train_dataset)
            continual_episode.append_task(task, self.sampler, val_dataset, test_dataset)
            
            continual_episode.initialize_model()  # train a new model by passing none checkpoint

            continual_episode.train(save_ckpt=self.ckpt)
            self.finish_task(continual_episode)
            continual_episode.eval(ckpt=self.ckpt)
            task_id += 1
            logger.info(f'[PROGRESS] ({task_id}/{len(ci_tasks)})')
        logger.info('CI finished!')
        output_continual_results(self.args, result_dict)

    def run_online(self):
        pass

def output_continual_results(args: TypedArgumentParser, result_dict):       
    output_dir = f'output/{args.setting}_{args.dataset}_{args.model}'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    with open(os.path.join(output_dir, 'precision'), 'a') as file:
        file.write(str(datetime.datetime.now()) + '\n')
        file.write(f'learning rate: {args.lr}; prototype update: {args.proto_update}; use augment: {args.augment}; metric: {args.metric};\n')
        num_task = len(result_dict)
        for task in result_dict:
            precisions = result_dict[task]['precision']
            num_place_holder = num_task - len(precisions)
            place_holders = ','.join('NaN' for _ in range(num_place_holder))
            if num_place_holder > 0:
                place_holders += ','
            task_precision = ','.join([str(_) for _ in precisions])
            file.write(place_holders + task_precision + '\n')
    with open(os.path.join(output_dir, 'f1'), 'a') as file:
        file.write(str(datetime.datetime.now()) + '\n')
        file.write(f'learning rate: {args.lr}; prototype update: {args.proto_update}; use augment: {args.augment}; metric: {args.metric};\n')
        num_task = len(result_dict)
        for task in result_dict:
            f1s = result_dict[task]['f1']
            num_place_holder = num_task - len(precisions)
            place_holders = ','.join('NaN' for _ in range(num_place_holder))
            if num_place_holder > 0:
                place_holders += ','
            task_f1 = ','.join([str(_) for _ in f1s])
            file.write(place_holders + task_f1 + '\n')
