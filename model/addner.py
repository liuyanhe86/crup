from ast import Add
import logging

import torch
from torch import nn

from model import NERModel

logger = logging.getLogger(__name__)        

class AddNER(NERModel):
    
    def __init__(self, word_encoder, teacher=None, alpha=0.5, beta=0.5):
        NERModel.__init__(self, word_encoder)
        self.drop = nn.Dropout()
        self.seen_types = {}
        self.teacher = teacher
        self.cur_type = None
        self.alpha = alpha
        self.beta = beta

    def forward(self, batch):
        batch_emb = self.word_encoder(batch['sentence'], batch['attention_mask'])
        batch_emb = self.drop(batch_emb)  # [batch_size, max_len, 768]
        batch_emb = batch_emb[batch['text_mask']==1].view(-1, batch_emb.size(-1))  # [num_of_tokens, 768]
        cur_lc = self.get_submodule(f'lc_{self.cur_type}')
        logits = cur_lc(batch_emb)  # [num_of_tokens, class_num]
        pred = self.predict(batch_emb)
        return logits, pred
    
    def loss(self, logits, label):
        return self.alpha * NERModel.loss(logits, label) + self.beta * self._KL(logits, label)
    
    def _KL(self, logits, label):
        pass

    def predict(self, embedding):
        lc_preds = []
        for label in self.seen_labels:
            lc = self.get_submodule(f'lc_{label}')
            lc_pred = torch.max(lc(embedding), 1)
            lc_preds.append(lc_pred)
        lc_preds = torch.stack(lc_preds)
        token_num = int(lc_preds.size(-1))
        final_pred = []
        for i in range(token_num):
            token_pred = lc_preds[:, i].tolist()
            if token_pred.count(0) == len(token_pred):
                final_pred.append(0)
            elif token_pred.count(0) == len(token_pred) - 1:
                final_pred.append(x for x in token_pred if x != 0)
            else:
                pred_set = set(token_pred)
                pred_set.discard(0)
                pred_count_dict = {_ : token_pred.count(_) for _ in pred_set}
                max_prob = max(pred_count_dict.values())
                max_pred = []
                for pred in pred_set:
                    if pred_count_dict[pred] == max_prob:
                        max_pred.append(pred)
                if len(max_pred) == 1:
                    final_pred.append(max_pred[0])
                else:
                    candidates = [p for p in max_pred if i > 0 and final_pred[i - 1] == p]
                    if len(candidates) == 1:
                        final_pred.append(candidates[0])
                    else:
                        final_pred.append(0)

