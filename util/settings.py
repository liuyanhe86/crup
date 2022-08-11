import datetime
import logging
import os
import random

from tqdm import tqdm
from transformers import BertTokenizer

from util.args import TypedArgumentParser
from util.datasets import ContinualNerDataset, GDumbSampler, NerDataset
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
        self.tokenizer = BertTokenizer.from_pretrained(self.args.pretrain_ckpt)

    def run(self):
        raise NotImplementedError

class SupervisedSetting(Setting):

    def run(self):
        logger.info('loading data...')
        supervised_task = PROTOCOLS[self.args.setting + ' ' + self.args.dataset]
        train_dataset = NerDataset(os.path.join(supervised_task, 'train.txt'), self.tokenizer, augment=self.args.augment, max_length=self.args.max_length, ignore_label_id=self.args.ignore_index)
        val_dataset = NerDataset(os.path.join(supervised_task, 'dev.txt'), self.tokenizer, max_length=self.args.max_length, ignore_label_id=self.args.ignore_index)
        test_dataset = NerDataset(os.path.join(supervised_task, 'test.txt'), self.tokenizer, max_length=self.args.max_length, ignore_label_id=self.args.ignore_index)
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
        result_dict = {task : {'precision': [], 'recall': [], 'f1': [], 'fp_error': [], 'fn_error':[], 'within_error':[], 'outer_error':[]} for task in ci_tasks}
        label2tag, tag2label = {0:'O'}, {'O':0}
        if self.args.model in {'AddNER', 'ExtendNER'}:
            continual_episode = DistilledContinualNerEpisode(self.args, result_dict)
        else:
            continual_episode = ContinualNerEpisode(self.args, result_dict)
            if self.args.si_c > 0:
                for n, p in continual_episode.model.named_parameters():
                    if p.requires_grad:
                        n = n.replace('.', '__')
                        continual_episode.model.register_buffer('{}_SI_prev_task'.format(n), p.data.clone())
        for task in ci_tasks:
            logger.info('loading data...')
            train_dataset = ContinualNerDataset(os.path.join(ci_tasks[task], 'train.txt'), self.tokenizer, augment=self.args.augment, max_length=self.args.max_length, ignore_label_id=self.args.ignore_index)
            val_dataset = ContinualNerDataset(os.path.join(ci_tasks[task], 'dev.txt'), self.tokenizer, max_length=self.args.max_length, ignore_label_id=self.args.ignore_index)
            test_dataset = ContinualNerDataset(os.path.join(ci_tasks[task], 'test.txt'), self.tokenizer, max_length=self.args.max_length, ignore_label_id=self.args.ignore_index)
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
            continual_episode.move_to(task, train_dataset, val_dataset, test_dataset)
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
        online_task = PROTOCOLS[self.args.setting + ' ' + self.args.dataset]
        train_dataset = NerDataset(os.path.join(online_task, 'train.txt'), self.tokenizer, augment=self.args.augment, max_length=self.args.max_length, ignore_label_id=self.args.ignore_index)
        val_dataset = NerDataset(os.path.join(online_task, 'dev.txt'), self.tokenizer, max_length=self.args.max_length, ignore_label_id=self.args.ignore_index)
        test_dataset = NerDataset(os.path.join(online_task, 'test.txt'), self.tokenizer, max_length=self.args.max_length, ignore_label_id=self.args.ignore_index)
        logger.info(f'train size: {len(train_dataset)}, val size: {len(val_dataset)}, test size: {len(test_dataset)}')
        online_episode = OnlineNerEpisode(self.args, train_dataset, val_dataset, test_dataset)
        result_dict = online_episode.learn(save_ckpt=self.ckpt)
        output_online_results(self.args, result_dict)
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
        label2tag, tag2label = {0:'O'}, {'O':0}
        continual_episode = ContinualNerEpisode(self.args, ci_tasks)
        for task in ci_tasks:
            logger.info('loading data...')
            train_dataset = ContinualNerDataset(os.path.join(ci_tasks[task], 'train.txt'), self.tokenizer, augment=self.args.augment, max_length=self.args.max_length, ignore_label_id=self.args.ignore_index)
            val_dataset = ContinualNerDataset(os.path.join(ci_tasks[task], 'dev.txt'), self.tokenizer, max_length=self.args.max_length, ignore_label_id=self.args.ignore_index)
            test_dataset = ContinualNerDataset(os.path.join(ci_tasks[task], 'test.txt'), self.tokenizer, max_length=self.args.max_length, ignore_label_id=self.args.ignore_index)
            logger.info(f'[TASK] {task} | train size: {len(train_dataset)}, val size: {len(val_dataset)}, test size: {len(test_dataset)}')
            num_of_existing_labels = len(label2tag)
            label2tag[num_of_existing_labels] = train_dataset.current_class
            tag2label[train_dataset.current_class] = num_of_existing_labels
            train_dataset.set_labelmap(label2tag, tag2label)
            val_dataset.set_labelmap(label2tag, tag2label)
            test_dataset.set_labelmap(label2tag, tag2label)

            self.sampler.init_members(train_dataset)
            self.sampler.sample_ci(train_dataset)
            continual_episode.move_to(task, self.sampler, val_dataset, test_dataset)
            
            continual_episode.initialize_model()  # train a new model by passing none checkpoint

            continual_episode.train(save_ckpt=self.ckpt)
            self.finish_task(continual_episode)
            continual_episode.eval(ckpt=self.ckpt)
            task_id += 1
            logger.info(f'[PROGRESS] ({task_id}/{len(ci_tasks)})')
        output_continual_results(self.args, continual_episode.get_results())
        logger.info('CI finished!')

    def run_online(self):
        online_task = PROTOCOLS[self.args.setting + ' ' + self.args.dataset]
        logger.info('loading data...')
        train_dataset = NerDataset(os.path.join(online_task, 'train.txt'), self.tokenizer, augment=self.args.augment, max_length=self.args.max_length, ignore_label_id=self.args.ignore_index)
        test_dataset = NerDataset(os.path.join(online_task, 'test.txt'), self.tokenizer, max_length=self.args.max_length, ignore_label_id=self.args.ignore_index)
        random.shuffle(train_dataset.samples)
        it = 0
        seen_labels = set()
        flag = True
        result_dict = {'precision': [], 'recall': [], 'f1': []}
        all_entities_seen = 0
        for sample in tqdm(train_dataset.samples, desc='online learning progress'):
            self.sampler.sample_online(sample)
            seen_labels = seen_labels.union(sample.tags)
            if (it + 1) % self.args.batch_size == 0:
                batches = (it + 1) // self.args.batch_size
                if batches % self.args.online_check_steps == 0:
                    logger.info(f'[TEST] Onine Check | batches: {batches}, all seen at: {all_entities_seen}')
                    self.sampler.init_members(train_dataset)
                    precision, recall, f1 = self._train_from_scratch(test_dataset)
                    result_dict['precision'].append(precision)
                    result_dict['recall'].append(recall)
                    result_dict['f1'].append(f1)
                    if all_entities_seen != 0 and batches > all_entities_seen + 100:
                        logger.info('100 its after all entities seen!')
                        break
            if len(seen_labels) == len(train_dataset.get_label_set()) and flag:
                logger.info(f'[NOTING] All entities have been seen at ith-batches: {batches}!')
                all_entities_seen = batches
                flag = False
                result_dict['all_entities_seen'] = all_entities_seen
            it += 1
        output_online_results(self.args, result_dict)
        logger.info('Online finished!')

    def _train_from_scratch(self, test_dataset):
        sup_episode = SupNerEpisode(self.args, train_dataset=self.sampler, val_dataset=None, test_dataset=test_dataset)
        sup_episode.initialize_model()
        sup_episode.train(save_ckpt=self.ckpt)
        precision, recall, f1, _, _, _, _ = sup_episode.eval(ckpt=self.ckpt)
        logger.info('RESULT: precision: %.4f, recall: %.4f, f1: %.4f' % (precision, recall, f1))
        return precision, recall, f1


def output_continual_results(args: TypedArgumentParser, result_dict):       
    output_dir = f'output/{args.setting}_{args.dataset}_{args.model}'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for perf in {'precision', 'recall', 'f1'}:
        with open(os.path.join(output_dir, perf), 'a') as file:
            file.write(str(datetime.datetime.now()) + '\n')
            file.write(f'learning rate: {args.lr}; prototype update: {args.proto_update}; use augment: {args.augment}; metric: {args.metric};\n')
            num_task = len(result_dict)
            for task in result_dict:
                perfs = result_dict[task][perf]
                num_place_holder = num_task - len(perfs)
                place_holders = ','.join('NaN' for _ in range(num_place_holder))
                if num_place_holder > 0:
                    place_holders += ','
                task_perfs = ','.join([str(_) for _ in perfs])
                file.write(place_holders + task_perfs + '\n')


def output_online_results(args: TypedArgumentParser, result_dict):       
    output_dir = f'output/{args.setting}_{args.dataset}_{args.model}'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for perf in {'precision', 'recall', 'f1'}:
        with open(os.path.join(output_dir, perf), 'a') as file:
            file.write(str(datetime.datetime.now()) + '\n')
            file.write(f'learning rate: {args.lr}; step when all entities are seen: {result_dict["all_entities_seen"]}\n')
            file.write(','.join([str(_) for _ in result_dict[perf]]) + '\n')
    

