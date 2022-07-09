import logging

import torch
from torch import nn

from model import NERModel

logger = logging.getLogger(__name__)        

class ExtendNER(NERModel):
    
    def __init__(self, word_encoder, n_class=2):
        NERModel.__init__(self, word_encoder)
        self.lc = nn.Linear(in_features=word_encoder.output_dim, out_features=n_class)
        self.n_class = n_class

    def train_forward(self, x):
        return self.forward(x)

    def forward(self, x):
        embedding = self.encode(x)
        logits = self.lc(embedding)  # [num_of_tokens, class_num]
        _, pred = torch.max(logits, 1)
        return logits, pred

    def add_unit(self):
        new_lc = nn.Linear(in_features=self.lc.in_features, out_features=self.lc.out_features + 1)
        with torch.no_grad():
            new_lc.weight[:self.lc.out_features] = self.lc.weight
            new_lc.bias[:self.lc.out_features] = self.lc.bias
        if torch.cuda.is_available():
            self.lc = new_lc.cuda()
        else:
            self.lc = new_lc

    def freeze(self):
        self.requires_grad_(False)
