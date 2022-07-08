import torch
from torch import nn, Tensor

import logging

logger = logging.getLogger(__name__)
class NERModel(nn.Module):
    def __init__(self, my_word_encoder): 
        '''
        word_encoder: Sentence encoder
        
        You need to set self.cost as your own loss function.
        '''
        nn.Module.__init__(self)
        self.word_encoder = nn.DataParallel(my_word_encoder)
        self.drop = nn.Dropout()

    def get_parameters_to_optimize(self):
        parameters_to_optimize = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        parameters_to_optimize = [
            {'params': [p for n, p in parameters_to_optimize 
                if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in parameters_to_optimize
                if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        return parameters_to_optimize

    def encode(self, batch) -> torch.Tensor:
        rep = self.word_encoder(batch['sentence'], batch['attention_mask'])
        rep = self.drop(rep)  # [batch_size, max_len, 768]
        rep = rep[batch['text_mask']==1]
        return rep    
    
    def forward(self, x):
        '''
        x: sentence
        return: logits, pred
        '''
        raise NotImplementedError
    
    def train_forward(self, x):
        '''
        x: sentence
        return: embedding
        '''
        raise NotImplementedError
    
    # def supcon_loss(self, embedding, labels):
    #     embedding = embedding[labels != self.ignore_index]
    #     labels = labels[labels != self.ignore_index]
    #     z_i, z_j = embedding, embedding.T
    #     dot_product_tempered = torch.mm(z_i, z_j) / self.temperature  # z_i dot z_j / tau
    #     # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
    #     exp_dot_tempered = (
    #         torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
    #     )
    #     if torch.cuda.is_available():
    #         mask_similar_class = (labels.unsqueeze(1).repeat(1, labels.shape[0]) == labels).cuda()
    #     else:
    #         mask_similar_class = (labels.unsqueeze(1).repeat(1, labels.shape[0]) == labels)
    #     if torch.cuda.is_available():
    #         mask_anchor_out = 1 - torch.eye(exp_dot_tempered.shape[0]).cuda()
    #     else:
    #         mask_anchor_out = 1 - torch.eye(exp_dot_tempered.shape[0])
    #     mask_combined = mask_similar_class * mask_anchor_out
    #     cardinality_per_batchs = torch.sum(mask_combined, dim=1)
    #     log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered, dim=1, keepdim=True)))
    #     supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_anchor_out, dim=1) / cardinality_per_batchs
    #     supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)
    #     return supervised_contrastive_loss

    def freeze_encoder(self):
        self.word_encoder.requires_grad_(False)

    