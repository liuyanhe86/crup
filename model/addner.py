from ast import Add
import logging
from turtle import forward

import torch
from torch import nn

from model import NERModel

logger = logging.getLogger(__name__)

class ParallelLinears(nn.Module):

    def __init__(self, rep_dim) -> None:
        nn.Module.__init__(self)
        self.rep_dim = rep_dim
        self.lc0 = nn.Linear(in_features=rep_dim, out_features=2)
        self.n_lc = 1
        self.lc_names = ['lc0']
    
    def add_lc(self):
        name = f'lc{self.n_lc}'
        self.add_module(name, nn.Linear(in_features=self.rep_dim, out_features=2))
        self.lc_names.append(name)
        self.n_lc += 1

    def forward(self, rep):
        all_logits = []
        for name in self.lc_names:
            logits = self.get_submodule(name)(rep)
            all_logits.append(logits)
        all_logits = torch.stack(all_logits)  # [n_lc, n_token, n_class(2)]
        return all_logits

class AddNER(NERModel):
    
    def __init__(self, word_encoder, ignore_index=-1):
        NERModel.__init__(self, word_encoder, ignore_index)
        self.drop = nn.Dropout()
        self.linears = ParallelLinears(rep_dim=word_encoder.output_dim)

    def train_forward(self, x):
        return self.forward(x)        

    def forward(self, x):
        rep = self.encode(x)
        all_logits = self.linears(rep)
        n_tokens = all_logits.size()[1]
        pred = []
        for i in range(n_tokens):
            logits_i = all_logits[:, i, :]
            _, pred_i = torch.max(logits_i, 1)
            if torch.all(pred == 0):
                pred.append(0)
            elif len(pred_i[pred_i == 1]) == 1:
                pred.append(torch.where(pred_i == 1)[0].item())
            else:
                candidates = logits_i[torch.where(pred_i == 1)[0]][:, 1]
                _, tmp_pred = torch.max(candidates, 0)
                pred.append(torch.where(pred_i == 1)[tmp_pred])
        pred = torch.as_tensor(pred)
        return all_logits, pred


        

