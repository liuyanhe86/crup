import logging
import torch
from torch import nn
import torch.nn.functional as F

from model import NERModel

logger = logging.getLogger(__name__)        

class CPR(NERModel):
    
    def __init__(self, word_encoder):
        NERModel.__init__(self, word_encoder)
        self.drop = nn.Dropout()
        self.proj_head = nn.Sequential(
            nn.Linear(in_features=word_encoder.output_dim, out_features=word_encoder.output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=word_encoder.output_dim, out_features=64)
        )

    def forward(self, sample):
        embedding = self.word_encoder(sample['sentence'], sample['attention_mask'])
        embedding = self.drop(embedding)  # [batch_size, max_len, 768]
        pred = self.proj_head(embedding)  # [batch_size, max_len, 64]
        
        pred = F.normalize(pred, p=2, dim=1)
        return embedding, pred
    
    def loss(self, pred, labels, mem):
        return self._sup_contrastive_loss(pred, labels) + self._proto_sample_contrastive_loss(pred, mem) + self._regulization()
    
    def _sup_contrastive_loss(self, sample, labels, pred):
        dot_product_tempered = torch.mm(sample, sample.T) / self.temperature  # n * m
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        exp_dot_tempered = (
            torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0].detach()) + 1e-5
        )
        mask_combined = (pred.unsqueeze(1).repeat(1, pred.shape[0]) == labels) # n*m
        
        cardinality_per_samples = torch.sum(mask_combined, dim=1)

        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered, dim=1, keepdim=True)))
        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples
        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)

        return supervised_contrastive_loss
    
    def _proto_sample_contrastive_loss(self, sample, memory):
        pass
    
    def _regulization(self):
        pass
