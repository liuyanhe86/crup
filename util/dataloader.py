import random
import torch
import torch.utils.data
import numpy as np
import os

from typing import Dict, List, Tuple

from transformers import BertTokenizer

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

class NERDataset(torch.utils.data.Dataset):
    """
    NER Dataset
    """
    def __init__(self, file_path:str, tokenizer, max_length: int, ignore_label_id: int=-1):
        if not os.path.exists(file_path):
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.class2sampleid = {} # [class:str : sample_with_class_ids:List[int]]
        self.tokenizer = tokenizer
        self.samples, self.classes = self.__load_data_from_file__(file_path)
        distinct_tags = ['O'] + list(self.classes)
        self.tag2label = {tag:idx for idx, tag in enumerate(distinct_tags)}
        self.label2tag = {idx:tag for idx, tag in enumerate(distinct_tags)}
        self.max_length = max_length
        self.ignore_label_id = ignore_label_id

    def __insert_sample__(self, index: int, sample_classes: List[str]) -> None:
        '''
        save class -> [sample ids] map into `self.class2sampleid`
            `index`: sample/sentence id in a file
            `sample_classes`: classes appeared in the `index`-th sample
        '''
        for item in sample_classes:
            if item in self.class2sampleid:
                self.class2sampleid[item].append(index)
            else:
                self.class2sampleid[item] = [index]
    
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
                self.__insert_sample__(index, sample_classes)
                classes += sample_classes
                samplelines = [] # clear for reading next sentence/sample
                index += 1
        if samplelines: # sentence/sample up to file ending
            sample = Sample(samplelines)
            samples.append(sample)
            sample_classes = sample.get_tag_class()
            self.__insert_sample__(index, sample_classes)
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
            else:
                raise ValueError(f'Word {word} cannot be tokenized!')
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

    def __populate__(self, idx: int, savelabeldic=False) -> Dict[str, List]:
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
        # for idx in idx_list:
            # get tokens and labels of idx_th sample
        tokens, labels = self.__get_token_label_list__(self.samples[idx])  
        word, attention_mask, text_mask, label = self.__getraw__(tokens, labels)
        word = torch.tensor(word).long()
        attention_mask = torch.tensor(np.array(attention_mask)).long()
        text_mask = torch.tensor(np.array(text_mask)).long()
        self.__additem__(idx, sample, word, attention_mask, text_mask, label)
        sample['sentence_num'] = [len(sample['sentence'])]
        if savelabeldic:
            sample['label2tag'] = [self.label2tag]
        return sample

    def __getitem__(self, index):
        # add 'O' and make sure 'O' is labeled 0
        dataset = self.__populate__(index)
        return dataset
    
    def __len__(self):
        return len(self.samples)


def collate_fn(data):
    batch = {'sentence': [], 'attention_mask': [], 'text_mask':[], 'label':[]}
    for i in range(len(data)):
        for k in batch:
            batch[k] += data[i][k]
    for k in batch:
        if k != 'label':
            batch[k] = torch.stack(batch[k], 0)
    batch['label'] = [torch.tensor(tag_list).long() for tag_list in batch['label']]
    return batch

def get_loader(file_path: str, tokenizer, batch_size: int, max_length: int, 
        num_workers=8, collate_fn=collate_fn, ignore_index=-1):
    dataset = NERDataset(file_path, tokenizer, max_length, ignore_label_id=ignore_index)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return data_loader


if __name__ == '__main__':
    train_loader = get_loader(file_path='../data/few-nerd/continual/coarse/non-overlapping/art.txt',\
        tokenizer=BertTokenizer.from_pretrained('bert-base-uncased'), batch_size=10, max_length=100)
    # for i, s in enumerate(train_loader):
    #     print(s)
    #     break

    