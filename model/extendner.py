import logging

import torch
from torch import nn

from model import NERModel

logger = logging.getLogger(__name__)        

class ExtendNER(NERModel):
    
    def __init__(self, word_encoder):
        NERModel.__init__(self, word_encoder)
        self.drop = nn.Dropout()
        self.lc = nn.Linear(in_features=word_encoder.output_dim, out_features=2)

    def train_forward(self, x):
        return self.forward(x)

    def forward(self, x):
        embedding = self.encode(x)
        logits = self.lc(embedding)  # [num_of_tokens, class_num]
        _, pred = torch.max(logits, 1)
        return logits, pred
