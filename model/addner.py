import logging

import torch
from torch import nn

from model import NERModel

logger = logging.getLogger(__name__)

class AddNER(NERModel):
    
    def __init__(self, word_encoder):
        NERModel.__init__(self, word_encoder)
        self.lc0 = nn.Linear(in_features=word_encoder.output_dim, out_features=2)
        self.n_lc = 1

    def train_forward(self, x):
        return self.forward(x)        

    def forward(self, x):
        rep = self.encode(x)
        all_logits = []
        for i in range(self.n_lc):
            logits = self.get_submodule(f'lc{i}')(rep)
            all_logits.append(logits)
        all_logits = torch.stack(all_logits)  # [n_lc, n_token, n_class(2)]        
        n_tokens = all_logits.size()[1]
        pred = []
        for i in range(n_tokens):
            logits_i = all_logits[:, i, :]
            _, pred_i = torch.max(logits_i, 1)
            # logger.info(f'logits: {logits_i.shape}, pred: {pred_i}')
            if torch.all(pred_i == 0):
                pred.append(0)
            elif len(pred_i[pred_i == 1]) == 1:
                pred.append(torch.where(pred_i == 1)[0].item() + 1)
            else:
                # logger.info(f'logits_i: {logits_i.shape}, pred_i: {pred_i}')
                candidates = logits_i[torch.where(pred_i == 1)[0]][:, 1]
                _, tmp_pred = torch.max(candidates, 0)
                pred.append(torch.where(pred_i == 1)[0][tmp_pred])
        pred = torch.as_tensor(pred)
        if torch.cuda.is_available():
            pred = pred.cuda()
        # logger.info(f'final pred: {pred.shape}')
        # exit(0)
        return all_logits, pred

    def add_lc(self):
        name = f'lc{self.n_lc}'
        new_lc = nn.Linear(in_features=self.lc0.in_features, out_features=2)
        if torch.cuda.is_available():
            self.add_module(name, new_lc.cuda())
        else:
            self.add_module(name, new_lc)
        self.n_lc += 1

    def freeze(self):
        self.requires_grad_(False)
        
