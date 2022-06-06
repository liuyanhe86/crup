import logging
import os

import torch
from torch import nn
from tqdm import tqdm
from torch.optim import AdamW, SGD

from .args import TypedArgumentParser
from .datautils import get_loader, NerDataset
from model import BERTWordEncoder, BertTagger, PCP, ProtoNet

logger = logging.getLogger(__name__)

def get_model(args: TypedArgumentParser):
    word_encoder = BERTWordEncoder(args.pretrain_ckpt)
    if args.model == 'ProtoNet':
        model = ProtoNet(word_encoder, dot=args.dot)
    elif args.model == 'Bert-Tagger':
        model = BertTagger(word_encoder)
    elif args.model == 'PCP':
        model = PCP(word_encoder, temperature=args.temperature, dot=args.dot)
    else:
        raise NotImplementedError(f'Error: Model {args.model} not implemented!')
    if torch.cuda.is_available():
        model.cuda()
    return model, word_encoder

class SupNerEpisode:

    def __init__(self, args: TypedArgumentParser, train_dataset:NerDataset=None, val_dataset:NerDataset=None, test_dataset:NerDataset=None):
        self.args = args
        self.model, self.word_encoder = get_model(args)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def initialize_model(self):
        lc = nn.Linear(in_features=self.word_encoder.output_dim, out_features=len(self.train_dataset.get_label_set()))
        if torch.cuda.is_available():
            self.model.add_module('lc', lc.cuda())
        else:
            self.model.add_module('lc', lc)
    
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

    def train(self, load_ckpt=None, save_ckpt=None):
        logger.info('Start training...')
        self.initialize_model()
        # Init optimizer
        parameters_to_optimize = self.model.get_parameters_to_optimize()
        # if self.args.use_sgd:
        # logger.info('Optimizer: SGD')
        # optimizer = SGD(parameters_to_optimize, lr=self.args.lr)
        # else:
        logger.info('Optimizer: AdamW')
        optimizer = AdamW(parameters_to_optimize, lr=self.args.classifier_lr)
        # load model
        if load_ckpt:
            state_dict = self.__load_model__(load_ckpt)['state_dict']
            own_state = self.model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    logger.info('ignore {}'.format(name))
                    continue
                own_state[name].copy_(param)
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
        while epoch + 1 < self.args.train_epoch and best_count < 3:
            it = 0
            for _, batch in tqdm(enumerate(train_data_loader), desc='train progress', total=len(self.train_dataset) // self.args.batch_size):
                label = torch.cat(batch['label'], 0)
                if torch.cuda.is_available():
                    for k in batch:
                        if k != 'label' and k != 'label2tag':
                            batch[k] = batch[k].cuda()
                    label = label.cuda()

                logits, pred = self.model(batch)
                assert logits.shape[0] == label.shape[0]
                loss = self.model.loss(logits, label)
                tmp_pred_cnt, tmp_label_cnt, correct = self.model.metrics_by_entity(pred, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += self.item(loss.data)
                pred_cnt += tmp_pred_cnt
                label_cnt += tmp_label_cnt
                correct_cnt += correct
                epoch_sample += 1
                if (it + 1) % 100 == 0:
                    precision = correct_cnt / pred_cnt
                    recall = correct_cnt / label_cnt
                    f1 = 2 * precision * recall / (precision + recall)
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
        logger.info('Start evaluating...')
        
        self.model.eval()
        total_eval_epoch = 0
        is_test = False
        if ckpt is None:
            logger.info("Use val dataset")
            eval_data_loader = get_loader(self.val_dataset, batch_size=self.args.batch_size, num_workers=0)
            total_eval_epoch = len(self.val_dataset) // self.args.batch_size
        else:
            logger.info("Use test dataset")
            if ckpt != 'none':
                state_dict = self.__load_model__(ckpt)['state_dict']
                own_state = self.model.state_dict()
                for name, param in state_dict.items():
                    if name not in own_state:
                        continue
                    own_state[name].copy_(param)
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

                tmp_pred_cnt, tmp_label_cnt, correct = self.model.metrics_by_entity(pred, label)
                fp, fn, token_cnt, within, outer, total_span = self.model.error_analysis(pred, label, batch['label2tag'])
                pred_cnt += tmp_pred_cnt
                label_cnt += tmp_label_cnt
                correct_cnt += correct

                fn_cnt += self.item(fn.data)
                fp_cnt += self.item(fp.data)
                total_token_cnt += token_cnt
                outer_cnt += outer
                within_cnt += within
                total_span_cnt += total_span
    
        if pred_cnt == 0:
            precision = 0
            logger.error('pred_cnt is 0!')
        else:
            precision = correct_cnt / pred_cnt
        
        if label_cnt == 0:
            recall = 0
            logger.error('label_cnt is 0!')
        else:
            recall = correct_cnt /label_cnt
        if precision + recall != 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
            logger.error('P and R is 0! f1 compute failed!')
        if total_token_cnt == 0:
            fp_error = 0
            fn_error = 0
            logger.error('total_token_cnt is 0!')
        else:
            fp_error = fp_cnt / total_token_cnt
            fn_error = fn_cnt / total_token_cnt
        if total_span_cnt == 0:
            within_error = 0
            outer_error = 0
            logger.error('total_span_cnt is 0!')
        else:
            within_error = within_cnt / total_span_cnt
            outer_error = outer_cnt / total_span_cnt
        if is_test:
            logger.info('[TEST] | [ENTITY] precision: {0:3.4f}, recall: {1:3.4f}, f1: {2:3.4f}'.format(precision, recall, f1) + '\r')
        else:
            logger.info('[VAL] | [ENTITY] precision: {0:3.4f}, recall: {1:3.4f}, f1: {2:3.4f}'.format(precision, recall, f1) + '\r')
        return precision, recall, f1, fp_error, fn_error, within_error, outer_error    

class SupConNerEpisode(SupNerEpisode):
    
    def __init__(self, args: TypedArgumentParser, train_dataset: NerDataset = None, val_dataset: NerDataset = None, test_dataset: NerDataset = None):
        SupNerEpisode.__init__(self, args, train_dataset, val_dataset, test_dataset)

    def train(self, load_ckpt=None, save_ckpt=None):
        logger.info('Start training...')
        # Init optimizer
        parameters_to_optimize = self.model.get_parameters_to_optimize()
        logger.info('Optimizer: AdamW')
        optimizer = AdamW(parameters_to_optimize, lr=self.args.lr)
        # load model
        if load_ckpt:
            state_dict = self.__load_model__(load_ckpt)['state_dict']
            own_state = super().model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    logger.info('ignore {}'.format(name))
                    continue
                own_state[name].copy_(param)

        self.model.train()
        epoch_loss = 0.
        epoch_sample = 0
        epoch = 0
        best_count = 0
        best_p = 0.0
        pred_cnt = 0
        label_cnt = 0
        correct_cnt = 0
        self.train_dataset.set_augment(True)
        train_data_loader = get_loader(self.train_dataset, batch_size=self.args.batch_size, num_workers=8)
        while epoch + 1 < self.args.train_epoch and best_count < 3:
            it = 0
            for _, batch in tqdm(enumerate(train_data_loader), desc='train progress', total=len(self.train_dataset) // self.args.batch_size):
                label = torch.cat(batch['label'], 0)
                if torch.cuda.is_available():
                    for k in batch:
                        if k != 'label' and k != 'label2tag':
                            batch[k] = batch[k].cuda()
                    label = label.cuda()

                embedding, pred = self.model(batch)
                loss = self.model.loss(embedding, label)
                tmp_pred_cnt, tmp_label_cnt, correct = self.model.metrics_by_entity(pred, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pred_cnt += tmp_pred_cnt
                label_cnt += tmp_label_cnt
                correct_cnt += correct
                epoch_loss += self.item(loss.data)
                epoch_sample += 1
                if (it + 1) % 100 == 0:
                    precision = correct_cnt / pred_cnt
                    recall = correct_cnt / label_cnt
                    logger.info('[TRAIN] it: {0} | loss: {1:2.6f} | p: {2:2.6f}, r: {3:2.6f}'
                                .format(it + 1, epoch_loss / epoch_sample, precision, recall))
                it += 1
            precision = correct_cnt / pred_cnt
            recall = correct_cnt / label_cnt
            logger.info('[TRAIN] it: {0} | loss: {1:2.6f} | p: {2:2.6f}, r: {3:2.6f}'
                        .format(it + 1, epoch_loss / epoch_sample, precision, recall))
            
            if (epoch + 1) % self.args.val_step == 0:   
                p, _, _, _, _, _, _ = self.eval()
                self.model.train()
                if p > best_p:
                    logger.info('Best checkpoint')
                    torch.save({'state_dict': self.model.state_dict()}, save_ckpt)
                    best_p = p
                    best_count = 0
                best_count += 1
            epoch_loss = 0.
            epoch_sample = 0.
            pred_cnt = 0
            label_cnt = 0
            correct_cnt = 0    
            epoch += 1
        if epoch < self.args.train_epoch:
            logger.info('Early stop.')        
        logger.info('Finish contrastive training.')

class ContinualNerEpisode(SupConNerEpisode):
    def __init__(self, args: TypedArgumentParser, result_dict: dict):
        SupConNerEpisode.__init__(self, args)
        self.test_datasets = {}
        self.result_dict = result_dict
    
    def append_task(self, task, train_dataset: NerDataset, val_dataset: NerDataset, test_dataset: NerDataset):
        self.task = task
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_datasets[task] = test_dataset

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

                tmp_pred_cnt, tmp_label_cnt, correct = self.model.metrics_by_entity(pred, label)
                fp, fn, token_cnt, within, outer, total_span = self.model.error_analysis(pred, label, batch['label2tag'])
                pred_cnt += tmp_pred_cnt
                label_cnt += tmp_label_cnt
                correct_cnt += correct

                fn_cnt += self.item(fn.data)
                fp_cnt += self.item(fp.data)
                total_token_cnt += token_cnt
                outer_cnt += outer
                within_cnt += within
                total_span_cnt += total_span

            if pred_cnt == 0:
                precision = 0
                logger.warning('pred_cnt is 0!')
            else:
                precision = correct_cnt / pred_cnt
            
            if label_cnt == 0:
                recall = 0
                logger.error('label_cnt is 0!')
            else:
                recall = correct_cnt /label_cnt
            if precision + recall != 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
                logger.warning('P and R is 0! f1 compute failed!')
            if total_token_cnt == 0:
                fp_error = 0
                fn_error = 0
                logger.error('total_token_cnt is 0!')
            else:
                fp_error = fp_cnt / total_token_cnt
                fn_error = fn_cnt / total_token_cnt
            if total_span_cnt == 0:
                within_error = 0
                outer_error = 0
                logger.error('total_span_cnt is 0!')
            else:
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
            if ckpt != 'none':
                state_dict = self.__load_model__(ckpt)['state_dict']
                own_state = self.model.state_dict()
                for name, param in state_dict.items():
                    if name not in own_state:
                        continue
                    own_state[name].copy_(param)
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

class OnlineNerEpisode(SupNerEpisode):
    def __init__(self, args, dataset=None):
        SupNerEpisode.__init__(self, args)
        self.dataset = dataset
        self.data_loader = get_loader(dataset=dataset, batch_size=args.batch_size, num_workers=8)
        
    def learn(self, save_ckpt):
        logger.info("Start online learning...")
    
        # Init optimizer
        parameters_to_optimize = self.model.get_parameters_to_optimize()

        # if self.args.use_sgd:
        #     logger.info('Optimizer: SGD')
        #     optimizer = torch.optim.SGD(parameters_to_optimize, lr=self.args.encoder_lr)
        # else:
        logger.info('Optimizer: AdamW')
        optimizer = AdamW(parameters_to_optimize, lr=self.args.encoder_lr, correct_bias=False)        

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
            tmp_pred_cnt, tmp_label_cnt, correct = self.model.metrics_by_entity(pred, label)
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
