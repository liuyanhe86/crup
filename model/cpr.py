import logging
import torch
from torch import nn
import torch.nn.functional as F

from model import NERModel

logger = logging.getLogger(__name__)        

class CPR(NERModel):
    
    def __init__(self,word_encoder, dot=False):
        NERModel.__init__(self, word_encoder)
        self.drop = nn.Dropout()
        self.dot = dot

    def __dist__(self, x, y, dim):
        if self.dot:
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, proto, embedding, text_mask):
        embedding = embedding[text_mask==1].view(-1, embedding.size(-1))
        return self.__dist__(proto.unsqueeze(0), embedding.unsqueeze(1), 2)

    def __get_proto__(self, embedding, tag, text_mask):
        proto = []
        # text_mask.shape: [batch_size, max_len]
        # before: embedding.shape: [batch_size, max_len, 768]
        embedding = embedding[text_mask==1].view(-1, embedding.size(-1))
        # after: embedding.shape: [num_of_tokens_within_batch, 768]
        tag = torch.cat(tag, 0)  # [num_of_tokens_within_batch]
        assert tag.size(0) == embedding.size(0)
        for label in range(torch.max(tag) + 1):
            # logger.info(f'__get_proto__: {label} embeddings shape:{embedding[tag==label].shape}; isnan:{embedding[tag==label].isnan()}')
            mean = torch.mean(embedding[tag==label], 0)
            if not torch.isnan(mean).any():
                # logger.info(f'__get_proto__: mean: {mean}')
                proto.append(mean)
            else:
                proto.append(torch.zeros(embedding.size(-1)).cuda())
        proto = torch.stack(proto)  # [class_num, 768]
        return proto

    def forward(self, sample, memory):
        sample_emb = self.word_encoder(sample['sentence'], sample['attention_mask'])
        sample_emb = self.drop(sample_emb)  # [batch_size, max_len, 768]
        # Prototypical Networks
        
        # Calculate prototype for each class
        proto = self.__get_proto__(sample_emb, sample['label'], sample['text_mask'])  # [class_num, 768]
        # calculate distance to each prototype
        logits = self.__batch_dist__(proto, sample_emb, sample['text_mask'])  # [num_of_tokens, class_num]
        _, pred = torch.max(logits, 1)  # [num_of_tokens]
        # logger.info(f'foward: pred={pred}; pred.shape={pred.shape}')
        logits = F.normalize(logits, p=2, dim=1)
        return logits, pred
    
    def loss(self, sample, pred, labels, memory):
        
        return self._sup_contrastive_loss(sample, pred, labels) + self._proto_sample_contrastive_loss(sample, memory) + self._regulization()
    
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
