import logging
import os
from typing import Dict

import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

from model import NERModel

logger = logging.getLogger(__name__)

def warmup_linear(global_step, warmup_step):
    if global_step < warmup_step:
        return global_step / warmup_step
    else:
        return 1.0

class SupNERFramework:

    def __init__(self, train_data_loader: DataLoader, val_data_loader: DataLoader, test_data_loader: DataLoader):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
    
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

    def train(self,
              model: NERModel,
              model_name: str,
              learning_rate=1e-1,
              train_iter=10000,
              val_iter=1000,
              val_step=2000,
              load_ckpt=None,
              save_ckpt=None,
              warmup_step=300,
              grad_iter=1,
              fp16=False,
              use_sgd_for_bert=True):
        '''
        model: a NERModel instance
        model_name: Name of the model
        ckpt_dir: Directory of checkpoints
        learning_rate: Initial learning rate
        train_iter: Num of iterations of training
        val_iter: Num of iterations of validating
        val_step: Validate every val_step steps
        '''
        logger.info("Start training...")
    
        # Init optimizer
        parameters_to_optimize = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        parameters_to_optimize = [
            {'params': [p for n, p in parameters_to_optimize 
                if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in parameters_to_optimize
                if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        if use_sgd_for_bert:
            logger.info('Optimizer: SGD')
            optimizer = torch.optim.SGD(parameters_to_optimize, lr=learning_rate)
        else:
            logger.info('Optimizer: AdamW')
            optimizer = AdamW(parameters_to_optimize, lr=learning_rate, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=train_iter) 
        
        # load model
        if load_ckpt:
            state_dict = self.__load_model__(load_ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    logger.info('ignore {}'.format(name))
                    continue
                logger.info('load {} from {}'.format(name, load_ckpt))
                own_state[name].copy_(param)

        if fp16:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

        model.train()

        # Training
        best_f1 = 0.0
        iter_loss = 0.0
        iter_sample = 0
        pred_cnt = 0
        label_cnt = 0
        correct_cnt = 0

        it = 0
        while it + 1 < train_iter:
            for _, batch in enumerate(self.train_data_loader):
                label = torch.cat(batch['label'], 0)
                if torch.cuda.is_available():
                    for k in batch:
                        if k != 'label' and k != 'label2tag':
                            batch[k] = batch[k].cuda()
                    label = label.cuda()

                logits, pred = model(batch)
                # logger.info(f'logits.shape: {logits.shape}; label.shape: {label.shape}')
                assert logits.shape[0] == label.shape[0]
                loss = model.loss(logits, label) / float(grad_iter)
                tmp_pred_cnt, tmp_label_cnt, correct = model.metrics_by_entity(pred, label)
                    
                if fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                if it % grad_iter == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                iter_loss += self.item(loss.data)
                #iter_right += self.item(right.data)
                pred_cnt += tmp_pred_cnt
                label_cnt += tmp_label_cnt
                correct_cnt += correct
                iter_sample += 1
                if (it + 1) % 100 == 0 or (it + 1) % val_step == 0:
                    precision = correct_cnt / pred_cnt
                    recall = correct_cnt / label_cnt
                    f1 = 2 * precision * recall / (precision + recall)
                    logger.info('step: {0:4} | loss: {1:2.6f} | [ENTITY] precision: {2:3.4f}, recall: {3:3.4f}, f1: {4:3.4f}'\
                        .format(it + 1, iter_loss/ iter_sample, precision, recall, f1) + '\r')

                if (it + 1) % val_step == 0:
                    _, _, f1, _, _, _, _ = self.eval(model, val_iter)
                    model.train()
                    if f1 > best_f1:
                        logger.info('Best checkpoint')
                        torch.save({'state_dict': model.state_dict()}, save_ckpt)
                        best_f1 = f1
                    iter_loss = 0.
                    iter_sample = 0.
                    pred_cnt = 0
                    label_cnt = 0
                    correct_cnt = 0

                if (it + 1)  == train_iter:
                    break
                it += 1
                
        logger.info(f'Finish training {model_name}.')


    def eval(self,
            model: NERModel,
            eval_iter: int=500,
            ckpt=None): 
        '''
        model: a NERModel instance
        eval_iter: Num of iterations
        ckpt: Checkpoint path. Set as None if using current model parameters.
        return: Accuracy
        '''
        logger.info('Start evaluating...')
        
        model.eval()
        if ckpt is None:
            logger.info("Use val dataset")
            eval_dataset = self.val_data_loader
        else:
            logger.info("Use test dataset")
            if ckpt != 'none':
                state_dict = self.__load_model__(ckpt)['state_dict']
                own_state = model.state_dict()
                for name, param in state_dict.items():
                    if name not in own_state:
                        continue
                    own_state[name].copy_(param)
            eval_dataset = self.test_data_loader

        pred_cnt = 0 # pred entity cnt
        label_cnt = 0 # true label entity cnt
        correct_cnt = 0 # correct predicted entity cnt

        fp_cnt = 0 # misclassify O as I-
        fn_cnt = 0 # misclassify I- as O
        total_token_cnt = 0 # total token cnt
        within_cnt = 0 # span correct but of wrong fine-grained type 
        outer_cnt = 0 # span correct but of wrong coarse-grained type
        total_span_cnt = 0 # span correct

        eval_iter = min(eval_iter, len(eval_dataset))

        with torch.no_grad():
            it = 0
            while it + 1 < eval_iter:
                for _, batch in enumerate(eval_dataset):
                    label = torch.cat(batch['label'], 0)
                    if torch.cuda.is_available():
                        for k in batch:
                            if k != 'label' and k != 'label2tag':
                                batch[k] = batch[k].cuda()
                        label = label.cuda()
                    logits, pred = model(batch)

                    tmp_pred_cnt, tmp_label_cnt, correct = model.metrics_by_entity(pred, label)
                    fp, fn, token_cnt, within, outer, total_span = model.error_analysis(pred, label, batch)
                    pred_cnt += tmp_pred_cnt
                    label_cnt += tmp_label_cnt
                    correct_cnt += correct

                    fn_cnt += self.item(fn.data)
                    fp_cnt += self.item(fp.data)
                    total_token_cnt += token_cnt
                    outer_cnt += outer
                    within_cnt += within
                    total_span_cnt += total_span

                    if it + 1 == eval_iter:
                        break
                    it += 1

            precision = correct_cnt / pred_cnt
            recall = correct_cnt /label_cnt
            f1 = 2 * precision * recall / (precision + recall)
            fp_error = fp_cnt / total_token_cnt
            fn_error = fn_cnt / total_token_cnt
            within_error = within_cnt / total_span_cnt
            outer_error = outer_cnt / total_span_cnt
            logger.info('[EVAL] step: {0:4} | [ENTITY] precision: {1:3.4f}, recall: {2:3.4f}, f1: {3:3.4f}'.format(it + 1, precision, recall, f1) + '\r')
        return precision, recall, f1, fp_error, fn_error, within_error, outer_error    

class ContinualNERFramework:
    def __init__(self, train_data_loader: DataLoader, val_data_loader: DataLoader, test_data_loaders: Dict[str, DataLoader]):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loaders = test_data_loaders
    
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

    def train(self,
              model: NERModel,
              model_name: str,
              learning_rate=1e-1,
              train_iter=10000,
              val_iter=1000,
              val_step=2000,
              load_ckpt=None,
              save_ckpt=None,
              warmup_step=300,
              grad_iter=1,
              fp16=False,
              use_sgd_for_bert=True):
        '''
        model: a NERModel instance
        model_name: Name of the model
        ckpt_dir: Directory of checkpoints
        learning_rate: Initial learning rate
        train_iter: Num of iterations of training
        val_iter: Num of iterations of validating
        val_step: Validate every val_step steps
        '''
        logger.info("Start training...")
    
        # Init optimizer
        parameters_to_optimize = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        parameters_to_optimize = [
            {'params': [p for n, p in parameters_to_optimize 
                if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in parameters_to_optimize
                if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        if use_sgd_for_bert:
            logger.info('Optimizer: SGD')
            optimizer = torch.optim.SGD(parameters_to_optimize, lr=learning_rate)
        else:
            logger.info('Optimizer: AdamW')
            optimizer = AdamW(parameters_to_optimize, lr=learning_rate, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=train_iter) 
        
        # load model
        if load_ckpt:
            state_dict = self.__load_model__(load_ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    continue
                own_state[name].copy_(param)
            logger.info(f'checkpoint {load_ckpt} loaded successfully')

        if fp16:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

        model.train()

        # Training
        best_f1 = 0.0
        iter_loss = 0.0
        iter_sample = 0
        pred_cnt = 0
        label_cnt = 0
        correct_cnt = 0

        it = 0
        while it + 1 < train_iter:
            best_count = 0
            for _, batch in enumerate(self.train_data_loader):
                label = torch.cat(batch['label'], 0)
                if torch.cuda.is_available():
                    for k in batch:
                        if k != 'label' and k != 'label2tag':
                            batch[k] = batch[k].cuda()
                    label = label.cuda()

                logits, pred = model(batch)
                assert logits.shape[0] == label.shape[0]
                loss = model.loss(logits, label) / float(grad_iter)
                tmp_pred_cnt, tmp_label_cnt, correct = model.metrics_by_entity(pred, label)
                    
                if fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                if it % grad_iter == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                iter_loss += self.item(loss.data)
                #iter_right += self.item(right.data)
                pred_cnt += tmp_pred_cnt
                label_cnt += tmp_label_cnt
                correct_cnt += correct
                iter_sample += 1
                if (it + 1) % 100 == 0 or (it + 1) % val_step == 0:
                    precision = correct_cnt / pred_cnt
                    recall = correct_cnt / label_cnt
                    if precision + recall == 0:
                        f1 = 0
                        logger.warning('f1 compute failed!')
                    else:
                        f1 = 2 * precision * recall / (precision + recall)
                    logger.info('step: {0:4} | loss: {1:2.6f} | [ENTITY] precision: {2:3.4f}, recall: {3:3.4f}, f1: {4:3.4f}'\
                        .format(it + 1, iter_loss/ iter_sample, precision, recall, f1) + '\r')

                if (it + 1) % val_step == 0:
                    _, _, f1, _, _, _, _ = self.eval(model, val_iter)
                    model.train()
                    if f1 > best_f1:
                        logger.info('Best checkpoint')
                        torch.save({'state_dict': model.state_dict()}, save_ckpt)
                        best_f1 = f1
                    iter_loss = 0.
                    iter_sample = 0.
                    pred_cnt = 0
                    label_cnt = 0
                    correct_cnt = 0

                if (it + 1)  == train_iter:
                    break
                it += 1
                
        logger.info(f'Finish training {model_name}.')

    def eval(self,
            model: NERModel,
            result_dict,
            eval_iter: int=500,
            ckpt=None): 
        '''
        model: a NERModel instance
        eval_iter: Num of iterations
        ckpt: Checkpoint path. Set as None if using current model parameters.
        return: Accuracy
        '''
        logger.info('Start evaluating...')
        
        model.eval()
            
        def eval_one_loader(dataloader, eval_iter):
            pred_cnt = 0 # pred entity cnt
            label_cnt = 0 # true label entity cnt
            correct_cnt = 0 # correct predicted entity cnt

            fp_cnt = 0 # misclassify O as I-
            fn_cnt = 0 # misclassify I- as O
            total_token_cnt = 0 # total token cnt
            within_cnt = 0 # span correct but of wrong fine-grained type 
            outer_cnt = 0 # span correct but of wrong coarse-grained type
            total_span_cnt = 0 # span correct

            eval_iter = min(eval_iter, len(dataloader))
            it = 0
            while it + 1 < eval_iter:
                for _, batch in enumerate(dataloader):
                    label = torch.cat(batch['label'], 0)
                    if torch.cuda.is_available():
                        for k in batch:
                            if k != 'label' and k != 'label2tag':
                                batch[k] = batch[k].cuda()
                        label = label.cuda()
                    _, pred = model(batch)

                    tmp_pred_cnt, tmp_label_cnt, correct = model.metrics_by_entity(pred, label)
                    fp, fn, token_cnt, within, outer, total_span = model.error_analysis(pred, label, batch)
                    pred_cnt += tmp_pred_cnt
                    label_cnt += tmp_label_cnt
                    correct_cnt += correct

                    fn_cnt += self.item(fn.data)
                    fp_cnt += self.item(fp.data)
                    total_token_cnt += token_cnt
                    outer_cnt += outer
                    within_cnt += within
                    total_span_cnt += total_span

                    if it + 1 == eval_iter:
                        break
                    it += 1

            if pred_cnt == 0:
                precision = 0
                logger.warning('pred_cnt is 0!')
            else:
                precision = correct_cnt / pred_cnt
            
            if label_cnt == 0:
                recall = 0
                logger.warning('label_cnt is 0!')
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
                logger.warning('total_token_cnt is 0!')
            else:
                fp_error = fp_cnt / total_token_cnt
                fn_error = fn_cnt / total_token_cnt
            if total_span_cnt == 0:
                within_error = 0
                outer_error = 0
                logger.warning('total_span_cnt is 0!')
            else:
                within_error = within_cnt / total_span_cnt
                outer_error = outer_cnt / total_span_cnt
            
            return precision, recall, f1, fp_error, fn_error, within_error, outer_error
    
        if ckpt is None:
            logger.info("Use val dataset")
            with torch.no_grad():
                precision, recall, f1, fp, fn, within, outer = eval_one_loader(self.val_data_loader, eval_iter)
                logger.info('[VAL] [ENTITY] precision: {0:3.4f}, recall: {1:3.4f}, f1: {2:3.4f}'.format(precision, recall, f1) + '\r')
                return precision, recall, f1, fp, fn, within, outer
        else:
            logger.info("Use test dataset")
            if ckpt != 'none':
                state_dict = self.__load_model__(ckpt)['state_dict']
                own_state = model.state_dict()
                for name, param in state_dict.items():
                    if name not in own_state:
                        continue
                    own_state[name].copy_(param)
            for task in self.test_data_loaders:
                precision, recall, f1, fp, fn, within, outer = eval_one_loader(self.test_data_loaders[task], eval_iter)
                logger.info('[TEST] %s | RESULT: precision: %.4f, recall: %.4f, f1: %.4f \n ERROR ANALYSIS: fp: %.4f, fn: %.4f, within: %.4f, outer: %.4f' % (task, precision, recall, f1, fp, fn, within, outer))
                result_dict[task]['precision'].append(precision)
                result_dict[task]['recall'].append(recall)
                result_dict[task]['f1'].append(f1)
                result_dict[task]['fp_error'].append(fp)
                result_dict[task]['fn_error'].append(fn)
                result_dict[task]['within_error'].append(within)
                result_dict[task]['outer_error'].append(outer)
                
