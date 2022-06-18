import logging

import torch
from torch import embedding_bag, nn

from model import NERModel

logger = logging.getLogger(__name__)        

class BertTagger(NERModel):
    
    def __init__(self, word_encoder):
        NERModel.__init__(self, word_encoder)
        self.drop = nn.Dropout()

    def train_forward(self, x):
        return self.forward(x)

    def forward(self, x):
        embedding = self.encode(x)
        logits = self.lc(embedding)  # [num_of_tokens, class_num]
        _, pred = torch.max(logits, 1)
        return logits, pred
