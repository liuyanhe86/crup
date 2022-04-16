import logging

import torch
from torch import nn

from model import NERModel

logger = logging.getLogger(__name__)        

class BertTagger(NERModel):
    
    def __init__(self, word_encoder, class_num=100):
        NERModel.__init__(self, word_encoder)
        self.drop = nn.Dropout()
        self.lc = nn.Linear(in_features=word_encoder.output_dim, out_features=class_num)
        
    def forward(self, sample):
        sample_emb = self.word_encoder(sample['sentence'], sample['attention_mask'])
        sample_emb = self.drop(sample_emb)  # [batch_size, max_len, 768]
        sample_emb = sample_emb[sample['text_mask']==1].view(-1, sample_emb.size(-1))  # [num_of_tokens, 768]
        logits = self.lc(sample_emb)  # [num_of_tokens, class_num]
        _, pred = torch.max(logits, 1)
        return logits, pred
