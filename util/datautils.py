import logging
import numpy as np
import os
from typing import Dict, List, Tuple
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
import sys
sys.path.append('..')

logger = logging.getLogger(__name__)

class SampleBase:
    '''
    Abstract Class
    DO NOT USE
    Build your own Sample class and inherit from this class
    '''
    def __init__(self):
        self.class_count = {}

    def get_class_count(self) -> Dict[str, int]:
        '''
        return a dictionary of {class_name : count} in format {str : int}
        '''
        return self.class_count
    
    def valid(self, target_classes: List[str]):
        '''
        decide whether classes of sample is subset of `target_classes`
        '''
        return set(self.get_class_count().keys()).issubset(set(target_classes))

class Sample(SampleBase):
    def __init__(self, filelines: List[str]):
        filelines = [line.split('\t') for line in filelines]
        self.words, self.tags = zip(*filelines)
        self.words = [word.lower() for word in self.words]
        self.class_count = {}

    def __count_entities__(self) -> None:
        '''
        count entities and classes in one sample(sentence)
        '''
        current_tag = self.tags[0]
        for tag in self.tags[1:]:
            if tag == current_tag:
                continue
            else:
                if current_tag != 'O':
                    if current_tag in self.class_count:
                        self.class_count[current_tag] += 1
                    else:
                        self.class_count[current_tag] = 1
                current_tag = tag
        if current_tag != 'O':
            if current_tag in self.class_count:
                self.class_count[current_tag] += 1
            else:
                self.class_count[current_tag] = 1

    def get_class_count(self) -> Dict[str, int]:
        '''
        get classes and number of samples per class in sample with format of {class:count}
        '''
        if self.class_count:
            return self.class_count
        else:
            self.__count_entities__()
            return self.class_count

    def get_tag_class(self) -> List[str]:
        '''
        get classes and number of samples per class in sample with format of {class:count}
        '''
        tag_class = list(set(self.tags))
        if 'O' in tag_class:
            tag_class.remove('O')
        return tag_class

    def __str__(self):
        newlines = zip(self.words, self.tags)
        return '\n'.join(['\t'.join(line) for line in newlines])

class NERDataset(Dataset):
    """
    NER Dataset
    """
    def __init__(self, file_path:str, tokenizer, max_length: int, ignore_label_id: int=-1):
        if not os.path.exists(file_path):
            logger.error(f"data file {file_path} does not exist!")
            assert(0)
        self.tokenizer = tokenizer
        self.samples, self.classes = self.__load_data_from_file__(file_path)
        # add 'O' and make sure 'O' is labeled 0
        distinct_tags = ['O'] + list(self.classes)
        self.tag2label = {tag:idx for idx, tag in enumerate(distinct_tags)}
        self.label2tag = {idx:tag for idx, tag in enumerate(distinct_tags)}
        self.max_length = max_length
        self.ignore_label_id = ignore_label_id
    
    def __load_data_from_file__(self, file_path:str) -> Tuple[List[Sample], List[str]]:
        '''
        load data from raw data file in "word\ttag" format
        '''
        samples = []
        classes = []
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines() # sentences split by empty line
        samplelines = [] # (word \t tag) within a sentence
        index = 0 # sentence/sample index
        for line in lines:
            line = line.strip()
            if line: # $line$ is not empty
                samplelines.append(line)
            else: # $line$ is an empty line, finish reading one sentence/sample
                sample = Sample(samplelines)
                samples.append(sample)
                sample_classes = sample.get_tag_class()
                classes += sample_classes
                samplelines = [] # clear for reading next sentence/sample
                index += 1
        if samplelines: # sentence/sample up to file ending
            sample = Sample(samplelines)
            samples.append(sample)
            sample_classes = sample.get_tag_class()
            classes += sample_classes
            samplelines = []
            index += 1
        classes = list(set(classes))
        return samples, classes

    def __get_token_label_list__(self, sample: Sample) -> Tuple[List[str], List[int]]:
        tokens = []
        labels = []
        for word, tag in zip(sample.words, sample.tags):
            word_tokens = self.tokenizer.tokenize(word)  # just tokenize one single word
            if word_tokens:
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                word_labels = [self.tag2label[tag]] + [self.ignore_label_id] * (len(word_tokens) - 1)
                labels.extend(word_labels)
            # else:
            #     raise ValueError(f'Word {word} cannot be tokenized!')
        return tokens, labels

    def __getraw__(self, tokens: List[str], labels: List[int]) -> Tuple[List[List[int]], List[np.ndarray], List[np.ndarray], List[List[int]]]:
        '''
        get tokenized word list, attention mask, text mask (mask [CLS], [SEP] as well), labels
        '''
        # split into chunks of length (max_length - 2)
        # 2 is for special tokens [CLS] and [SEP]
        tokens_list = []
        labels_list = []
        while len(tokens) > self.max_length - 2:
            tokens_list.append(tokens[ : self.max_length - 2])
            tokens = tokens[self.max_length-2:]
            labels_list.append(labels[ : self.max_length - 2])
            labels = labels[self.max_length-2:]
        if tokens:
            tokens_list.append(tokens)
            labels_list.append(labels)

        # add special tokens and get masks
        indexed_tokens_list = []
        attention_mask_list = []
        text_mask_list = []
        for i, tokens in enumerate(tokens_list):
            # tokens -> ids
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        
            # right padding
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)
            indexed_tokens_list.append(indexed_tokens)

            # attention mask
            attention_mask = np.zeros((self.max_length), dtype=np.int32)
            attention_mask[:len(tokens)] = 1
            attention_mask_list.append(attention_mask)

            # text mask, also mask [CLS] and [SEP]
            text_mask = np.zeros((self.max_length), dtype=np.int32)
            text_mask[1:len(tokens)-1] = 1
            text_mask_list.append(text_mask)

            assert len(labels_list[i]) == len(tokens) - 2, print(labels_list[i], tokens)
        return indexed_tokens_list, attention_mask_list, text_mask_list, labels_list

    def __additem__(self, index, dataset, word, attention_mask, text_mask, label) -> None:
        dataset['index'].append(index)
        dataset['sentence'] += word
        dataset['attention_mask'] += attention_mask
        dataset['text_mask'] += text_mask
        dataset['label'] += label

    def __populate__(self, idx: int) -> Dict[str, List]:
        '''
        populate sample into data dict
        set `savelabeldic=True` if you want to save label2tag dict
        '''

        '''
        index: sample index in all samples
        word: tokenized word ids
        attention mask: attention mask in BERT
        text_mask: 0 for special tokens and paddings, 1 for real text
        label: NER labels, List[int]
        '''
        sample = {'index':[], 'sentence': [], 'attention_mask': [], 'text_mask':[], 'label':[]}
        tokens, labels = self.__get_token_label_list__(self.samples[idx])
        word, attention_mask, text_mask, label = self.__getraw__(tokens, labels)
        word = torch.tensor(word).long()
        attention_mask = torch.tensor(np.array(attention_mask)).long()
        text_mask = torch.tensor(np.array(text_mask)).long()
        self.__additem__(idx, sample, word, attention_mask, text_mask, label)
        sample['label2tag'] = self.label2tag
        return sample

    def __getitem__(self, index):
        data_item = self.__populate__(index)
        return data_item
    
    def __len__(self):
        return len(self.samples)

    def get_label_set(self):
        return set(self.label2tag.keys())

class ContinualNERDataset(NERDataset):
    """
    Continual NER Dataset
    """
    def __init__(self, file_path: str, tokenizer, max_length: int, label_offset: int=0, ignore_label_id: int=-1):
        if not os.path.exists(file_path):
            logger.error(f"[ERROR] Data file {file_path} does not exist!")
            assert(0)
        self.tokenizer = tokenizer
        self.samples, self.classes = self.__load_data_from_file__(file_path)
        self.tag2label = {tag:idx + label_offset + 1 for idx, tag in enumerate(list(self.classes))}
        self.tag2label['O'] = 0
        self.label2tag = {idx + label_offset + 1:tag for idx, tag in enumerate(list(self.classes))}
        self.label2tag[0] = 'O'
        self.max_length = max_length
        self.ignore_label_id = ignore_label_id
    
class MultiNERDataset(NERDataset):
    """
    Multi NER Dataset
    """
    def __init__(self, tokenizer, max_length: int, ignore_label_id: int=-1):
        self.tokenizer = tokenizer
        self.samples = []
        self.tag2label, self.label2tag = {}, {}
        self.tag2label['O'] = 0
        self.label2tag[0] = 'O'
        self.max_length = max_length
        self.ignore_label_id = ignore_label_id
    
    
    def append(self, file_path: str, label_offset: int=0):
        '''
        Append new dataset
        return: type set size of appended dataset
        '''
        if not os.path.exists(file_path):
            logger.error(f"[ERROR] Data file {file_path} does not exist!")
            assert(0)
        samples, classes = self.__load_data_from_file__(file_path)
        self.samples.extend(samples)
        for idx, tag in enumerate(list(classes)):
            self.tag2label[tag] = idx + label_offset + 1
            self.label2tag[idx + label_offset + 1] = tag
        return len(classes)
    
def collate_fn(data):
    batch = {'sentence': [], 'attention_mask': [], 'text_mask':[], 'label':[]}
    for i in range(len(data)):
        for k in batch:
            batch[k] += data[i][k]
    for k in batch:
        if k != 'label':
            batch[k] = torch.stack(batch[k], 0)
    batch['label'] = [torch.tensor(tag_list).long() for tag_list in batch['label']]
    batch['label2tag'] = data[0]['label2tag']
    return batch

def get_loader(dataset, batch_size):
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    return data_loader

if __name__ == '__main__':
    file_path='../data/few-nerd/continual/coarse/disjoint/art/train.txt'
    tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
    max_length=100
    label_offset=10
    dataset = ContinualNERDataset(file_path, tokenizer, max_length, label_offset=label_offset)
    train_loader = get_loader(dataset, batch_size=10)
    for i, batch in enumerate(train_loader):
        with open('batch_example_new.log', 'w') as f:
            f.writelines(str(batch))
        break

    