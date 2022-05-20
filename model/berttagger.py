import logging

import torch
from torch import nn

from model import NERModel

logger = logging.getLogger(__name__)        

class BertTagger(NERModel):
    
    def __init__(self, word_encoder):
        NERModel.__init__(self, word_encoder)
        self.drop = nn.Dropout()

    def forward(self, batch):
        batch_emb = self.word_encoder(batch['sentence'], batch['attention_mask'])
        batch_emb = self.drop(batch_emb)  # [batch_size, max_len, 768]
        batch_emb = batch_emb[batch['text_mask']==1].view(-1, batch_emb.size(-1))  # [num_of_tokens, 768]
        logits = self.lc(batch_emb)  # [num_of_tokens, class_num]
        _, pred = torch.max(logits, 1)
        return logits, pred
