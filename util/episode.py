import copy
import logging
import os

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from torch.optim import AdamW, SGD

from util.args import TypedArgumentParser
from util.datasets import get_loader
from util.evaluator import Evaluater
from model import BERTWordEncoder, BertTagger, PCP, ProtoNet, AddNER, ExtendNER


logger = logging.getLogger(__name__)

def get_model(args: TypedArgumentParser):
    word_encoder = BERTWordEncoder(pretrain_path=args.pretrain_ckpt)
    if args.model == 'ProtoNet':
        model = ProtoNet(word_encoder, proto_update=args.proto_update, metric=args.metric)
    elif args.model == 'Bert-Tagger':
        model = BertTagger(word_encoder)
    elif args.model == 'PCP':
        model = PCP(word_encoder, embedding_dimension=args.embedding_dimension, temperature=args.temperature)
    elif args.model == 'AddNER':
        model = AddNER(word_encoder)
    elif args.model == 'ExtendNER':
        model = ExtendNER(word_encoder)
    else:
        raise NotImplementedError(f'Error: Model {args.model} not implemented!')
    if torch.cuda.is_available():
        model = model.cuda()
    return model

class SupNerEpisode:

    def __init__(self, args: TypedArgumentParser, load_ckpt=None, train_dataset=None, val_dataset=None, test_dataset=None):
        self.args = args
        self.model = get_model(args)
        self.embedding_dim = 768
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.evaluator = Evaluater(ignore_index=args.ignore_index)

    def initialize_model(self):
        if self.args.model == 'Bert-Tagger':
            lc = nn.Linear(in_features=self.embedding_dim, out_features=len(self.train_dataset.get_label_set()))
            if torch.cuda.is_available():
                self.model.add_module('lc', lc.cuda())
            else:
                self.model.add_module('lc', lc)
        elif self.args.model == 'ProtoNet':
            if self.args.proto_update == 'SDC':
                self.model.start_training()
        
    def load_state(self, ckpt):
        state_dict = self.__load_model__(ckpt)['state_dict']
        own_state = self.model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            own_state[name].copy_(param)

    
    def __load_model__(self, ckpt):
        '''
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        '''
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            logger.info("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)
    
    def item(self, x):
        '''
        PyTorch before and after 0.4
        '''
        torch_version = torch.__version__.split('.')
        if int(torch_version[0]) == 0 and int(torch_version[1]) < 4:
            return x[0]
        else:
            return x.item()

    def default_loss(self, logits, label):
        N = logits.size(-1)
        cost = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
        return cost(logits.view(-1, N), label.view(-1))

    def loss(self, batch, label):
        if self.args.augment:
            (mu, sigma), pred = self.model.train_forward(batch)
            pass
        else:
            logits, pred = self.model.train_forward(batch)
            assert logits.shape[0] == label.shape[0]
            return self.default_loss(logits, label), pred

    def train(self, save_ckpt=None):
        logger.info('Start training ...')
        # Init optimizer
        parameters_to_optimize = self.model.get_parameters_to_optimize()
        logger.info('Optimizer: SGD')
        optimizer = SGD(parameters_to_optimize, lr=self.args.lr)
        
        self.model.train()

        # Training
        best_f1 = 0.0
        epoch_loss = 0.0
        epoch_sample = 0
        pred_cnt = 0
        label_cnt = 0
        correct_cnt = 0

        epoch = 0
        best_count = 0
        train_data_loader = get_loader(self.train_dataset, batch_size=self.args.batch_size, num_workers=8)
        while epoch < self.args.train_epoch and best_count <= 3:
            it = 0
            for _, batch in tqdm(enumerate(train_data_loader), desc='train progress', total=len(self.train_dataset) // self.args.batch_size):
                label = torch.cat(batch['label'], 0)
                if torch.cuda.is_available():
                    for k in batch:
                        if k != 'label' and k != 'label2tag':
                            batch[k] = batch[k].cuda()
                    label = label.cuda()

                loss, pred = self.loss(batch, label)
                tmp_pred_cnt, tmp_label_cnt, correct = self.evaluator.metrics_by_entity(pred, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += self.item(loss.data)
                pred_cnt += tmp_pred_cnt
                label_cnt += tmp_label_cnt
                correct_cnt += correct
                epoch_sample += 1
                if (it + 1) % 500 == 0:
                    precision = correct_cnt / pred_cnt
                    recall = correct_cnt / label_cnt
                    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else float('inf')
                    logger.info('[TRAIN] it: {0} | loss: {1:2.6f} | [ENTITY] precision: {2:3.4f}, recall: {3:3.4f}, f1: {4:3.4f}'\
                .format(it + 1, epoch_loss/ epoch_sample, precision, recall, f1) + '\r')
                it += 1
            precision = correct_cnt / pred_cnt
            recall = correct_cnt / label_cnt
            f1 = 2 * precision * recall / (precision + recall)
            logger.info('[TRAIN] epoch: {0} | loss: {1:2.6f} | [ENTITY] precision: {2:3.4f}, recall: {3:3.4f}, f1: {4:3.4f}'\
                .format(epoch + 1, epoch_loss/ epoch_sample, precision, recall, f1) + '\r')

            if (epoch + 1) % self.args.val_step == 0:
                _, _, f1, _, _, _, _ = self.eval()
                self.model.train()
                if f1 > best_f1:
                    logger.info('Best checkpoint')
                    torch.save({'state_dict': self.model.state_dict()}, save_ckpt)
                    best_f1 = f1
                    best_count = 0
                best_count += 1
            epoch_loss = 0.
            epoch_sample = 0.
            pred_cnt = 0
            label_cnt = 0
            correct_cnt = 0    
            epoch += 1
        if epoch + 1 < self.args.train_epoch:
            logger.info('Early stop')
        logger.info(f'Finish training {self.args.model} on {self.args.dataset} in {self.args.setting} setting.')

    def eval(self, ckpt=None): 
        logger.info('Start evaluating ...')
        
        self.model.eval()
        total_eval_epoch = 0
        is_test = False
        if ckpt is None:
            logger.info("Use val dataset")
            eval_data_loader = get_loader(self.val_dataset, batch_size=self.args.batch_size, num_workers=0)
            total_eval_epoch = len(self.val_dataset) // self.args.batch_size
        else:
            logger.info("Use test dataset")
            self.load_state(ckpt)
            eval_data_loader = get_loader(self.test_dataset, batch_size=self.args.batch_size, num_workers=4)
            total_eval_epoch = len(self.test_dataset) // self.args.batch_size
            is_test = True

        pred_cnt = 0 # pred entity cnt
        label_cnt = 0 # true label entity cnt
        correct_cnt = 0 # correct predicted entity cnt

        fp_cnt = 0 # misclassify O as I-
        fn_cnt = 0 # misclassify I- as O
        total_token_cnt = 0 # total token cnt
        within_cnt = 0 # span correct but of wrong fine-grained type 
        outer_cnt = 0 # span correct but of wrong coarse-grained type
        total_span_cnt = 0 # span correct

        with torch.no_grad():
            for _, batch in tqdm(enumerate(eval_data_loader), desc='eval progress', total=total_eval_epoch):
                label = torch.cat(batch['label'], 0)
                if torch.cuda.is_available():
                    for k in batch:
                        if k != 'label' and k != 'label2tag':
                            batch[k] = batch[k].cuda()
                    label = label.cuda()
                batch.pop('label')
                _, pred = self.model(batch)

                tmp_pred_cnt, tmp_label_cnt, correct = self.evaluator.metrics_by_entity(pred, label)
                fp, fn, token_cnt, within, outer, total_span = self.evaluator.error_analysis(pred, label, batch['label2tag'])
                pred_cnt += tmp_pred_cnt
                label_cnt += tmp_label_cnt
                correct_cnt += correct

                fn_cnt += self.item(fn.data)
                fp_cnt += self.item(fp.data)
                total_token_cnt += token_cnt
                outer_cnt += outer
                within_cnt += within
                total_span_cnt += total_span
    
        precision = correct_cnt / pred_cnt
        recall = correct_cnt /label_cnt
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        fp_error = fp_cnt / total_token_cnt
        fn_error = fn_cnt / total_token_cnt
        within_error = within_cnt / total_span_cnt
        outer_error = outer_cnt / total_span_cnt
        if not is_test:
            logger.info('[VAL] | [ENTITY] precision: {0:3.4f}, recall: {1:3.4f}, f1: {2:3.4f}'.format(precision, recall, f1) + '\r')
        return precision, recall, f1, fp_error, fn_error, within_error, outer_error    

class ContinualNerEpisode(SupNerEpisode):
    def __init__(self, args: TypedArgumentParser, result_dict: dict):
        SupNerEpisode.__init__(self, args)
        self.test_datasets = {}
        self.result_dict = result_dict
    
    def initialize_model(self, load_ckpt=None):
        super().initialize_model()
        if load_ckpt:
            self.load_state(load_ckpt)

    def append_task(self, task, train_dataset, val_dataset, test_dataset):
        self.task = task
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_datasets[task] = test_dataset

    def finish_task(self):
        if self.args.model == 'ProtoNet':
            logger.info(f'num of current prototypes: {len(self.model.global_protos)}')

    def eval(self, ckpt=None): 
        logger.info('Start evaluating...')
        
        self.model.eval()
            
        def eval_one_loader(dataloader, total):
            pred_cnt = 0 # pred entity cnt
            label_cnt = 0 # true label entity cnt
            correct_cnt = 0 # correct predicted entity cnt

            fp_cnt = 0 # misclassify O as I-
            fn_cnt = 0 # misclassify I- as O
            total_token_cnt = 0 # total token cnt
            within_cnt = 0 # span correct but of wrong fine-grained type 
            outer_cnt = 0 # span correct but of wrong coarse-grained type
            total_span_cnt = 0 # span correct
            for _, batch in tqdm(enumerate(dataloader), desc='eval progress', total=total):
                label = torch.cat(batch['label'], 0)
                if torch.cuda.is_available():
                    for k in batch:
                        if k != 'label' and k != 'label2tag':
                            batch[k] = batch[k].cuda()
                    label = label.cuda()
                batch.pop('label')
                _, pred = self.model(batch)

                tmp_pred_cnt, tmp_label_cnt, correct = self.evaluator.metrics_by_entity(pred, label)
                fp, fn, token_cnt, within, outer, total_span = self.evaluator.error_analysis(pred, label, batch['label2tag'])
                pred_cnt += tmp_pred_cnt
                label_cnt += tmp_label_cnt
                correct_cnt += correct

                fn_cnt += self.item(fn.data)
                fp_cnt += self.item(fp.data)
                total_token_cnt += token_cnt
                outer_cnt += outer
                within_cnt += within
                total_span_cnt += total_span

            precision = correct_cnt / pred_cnt
            recall = correct_cnt /label_cnt
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
            fp_error = fp_cnt / total_token_cnt
            fn_error = fn_cnt / total_token_cnt
            within_error = within_cnt / total_span_cnt
            outer_error = outer_cnt / total_span_cnt
            
            return precision, recall, f1, fp_error, fn_error, within_error, outer_error
    
        if ckpt is None:
            logger.info("Use val dataset")
            with torch.no_grad():
                val_data_loader = get_loader(self.val_dataset, batch_size=self.args.batch_size, num_workers=0)
                precision, recall, f1, fp, fn, within, outer = eval_one_loader(val_data_loader, total=len(self.val_dataset) // self.args.batch_size)
                logger.info('[VAL] | [ENTITY] precision: {0:3.4f}, recall: {1:3.4f}, f1: {2:3.4f}'.format(precision, recall, f1) + '\r')
                return precision, recall, f1, fp, fn, within, outer
        else:
            logger.info("Use test dataset")
            self.load_state(ckpt)
            with torch.no_grad():
                for task in self.test_datasets:
                    test_data_loader = get_loader(self.test_datasets[task], batch_size=self.args.batch_size, num_workers=4)
                    precision, recall, f1, fp, fn, within, outer = eval_one_loader(test_data_loader, total=len(self.test_datasets[task]) // self.args.batch_size)
                    logger.info('[TEST] %s | RESULT: precision: %.4f, recall: %.4f, f1: %.4f \n ERROR ANALYSIS: fp: %.4f, fn: %.4f, within: %.4f, outer: %.4f' % (task, precision, recall, f1, fp, fn, within, outer))
                    self.result_dict[task]['precision'].append(precision)
                    self.result_dict[task]['recall'].append(recall)
                    self.result_dict[task]['f1'].append(f1)
                    self.result_dict[task]['fp_error'].append(fp)
                    self.result_dict[task]['fn_error'].append(fn)
                    self.result_dict[task]['within_error'].append(within)
                    self.result_dict[task]['outer_error'].append(outer)

class DistilledContinualNerEpisode(ContinualNerEpisode):

    def __init__(self, args: TypedArgumentParser, result_dict: dict):
        ContinualNerEpisode.__init__(self, args, result_dict)
        self.teacher = None       

    def loss(self, batch, label):
        if self.args.model == 'AddNER':
            all_logits, pred = self.model.train_forward(batch)
            label[label > 0] = 1
            ce_loss = self.default_loss(all_logits[-1], label)
            if self.teacher:
                teacher_dist, _ = self.teacher.train_forward(batch)
                student_dist, _ = self.model.train_forward(batch)
                kl_div = F.kl_div(F.log_softmax(student_dist[:-1], dim=-1), F.softmax(teacher_dist, dim=-1), reduction='batchmean')
                return self.args.alpha * ce_loss + self.args.beta * kl_div, pred
            else:
                return ce_loss, pred
        elif self.args.model == 'ExtendNER':
            all_logits, pred = self.model.train_forward(batch)
            ce_loss = self.default_loss(all_logits, label)
            if self.teacher:
                teacher_dist, _ = self.teacher.train_forward(batch)
                student_dist, _ = self.model.train_forward(batch)
                kl_div = F.kl_div(F.log_softmax(student_dist[:,:-1], dim=-1), F.softmax(teacher_dist, dim=-1), reduction='batchmean')
                return self.args.alpha * ce_loss + self.args.beta * kl_div, pred
            else:
                return ce_loss, pred

    def initialize_model(self, load_ckpt=None):
        if load_ckpt:
            self.load_state(load_ckpt)
            self.teacher = copy.deepcopy(self.model)
            self.teacher.freeze()
            if self.args.model == 'AddNER':
                self.model.add_lc()
            elif self.args.model == 'ExtendNER':
                self.model.add_unit()
                
    

class OnlineNerEpisode(SupNerEpisode):
    def __init__(self, args:TypedArgumentParser, dataset=None):
        SupNerEpisode.__init__(self, args)
        self.dataset = dataset
        self.data_loader = get_loader(dataset=dataset, batch_size=args.batch_size, num_workers=8)
        
    def learn(self, save_ckpt):
        logger.info("Start online learning...")
        # Init optimizer
        parameters_to_optimize = self.model.get_parameters_to_optimize()

        if self.args.use_sgd:
            logger.info('Optimizer: SGD')
            optimizer = torch.optim.SGD(parameters_to_optimize, lr=self.args.lr)
        else:
            logger.info('Optimizer: AdamW')
            optimizer = AdamW(parameters_to_optimize, lr=self.args.lr)        

        self.model.train()
        # Training
        epoch_loss = 0.0
        epoch_sample = 0
        pred_cnt = 0
        label_cnt = 0
        correct_cnt = 0
        it = 0
        for _, batch in enumerate(self.data_loader):
            label = torch.cat(batch['label'], 0)
            if torch.cuda.is_available():
                for k in batch:
                    if k != 'label' and k != 'label2tag':
                        batch[k] = batch[k].cuda()
                label = label.cuda()
            logits, pred = self.model(batch)
            assert logits.shape[0] == label.shape[0]
            loss = self.model.loss(logits, label)
            tmp_pred_cnt, tmp_label_cnt, correct = self.evaluator.metrics_by_entity(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += self.item(loss.data)
            pred_cnt += tmp_pred_cnt
            label_cnt += tmp_label_cnt
            correct_cnt += correct
            epoch_sample += 1
            
            precision = correct_cnt / pred_cnt
            recall = correct_cnt / label_cnt
            f1 = 2 * precision * recall / (precision + recall)
            if (it + 1) % self.args.val_step == 0:
                logger.info('epoch: {0:4} | loss: {1:2.6f} | [ENTITY] precision: {2:3.4f}, recall: {3:3.4f}, f1: {4:3.4f}'\
                    .format(it + 1, epoch_loss/ epoch_sample, precision, recall, f1) + '\r')
                epoch_loss = 0.
                epoch_sample = 0.
                pred_cnt = 0
                label_cnt = 0
                correct_cnt = 0
            it += 1
        torch.save({'state_dict': self.model.state_dict()}, save_ckpt)
        logger.info(f'Finish learning {self.args.model}.')
