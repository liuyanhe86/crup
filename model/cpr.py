import logging
import torch
from torch import nn
import torch.nn.functional as F

from model import NERModel

logger = logging.getLogger(__name__)

class CPR(NERModel):
    
    def __init__(self, word_encoder, protocol='CI', temperature=0.5, embedding_dim=64):
        NERModel.__init__(self, word_encoder)
        self.drop = nn.Dropout()
        self.proj_head = nn.Sequential(
            nn.Linear(in_features=word_encoder.output_dim, out_features=word_encoder.output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=word_encoder.output_dim, out_features=64)
        )
        self.global_protos = {}
        self.protocol = protocol
        self.temperature = temperature
        self.embedding_dim = embedding_dim
    
    def add_heads(self, label_set):
        for label in label_set:
            if label not in self.global_protos:
                self.add_module(f'shift_head_{label}', nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim).cuda())

    def __dist__(self, x, y, dim):
        return -(torch.pow(x - y, 2)).sum(dim)
        # return (x * y).sum(dim)

    def _batch_dist(self, protos, embeddings):
        return self.__dist__(protos.unsqueeze(0), embeddings.unsqueeze(1), 2)

    def _update_protos(self, embedding, sample):
        tag = sample['label']
        tag = torch.cat(tag, 0)
        tag_set = set([int(_) for _ in tag.tolist()]) - {self.ignore_index}
        local_proto = []
        index2tag = {}
        index = 0
        assert tag.size(0) == embedding.size(0)
        for label in tag_set:
            mean = torch.mean(embedding[tag==label], dim=0)
            if label not in self.global_protos:
                self.global_protos[label] = mean
            else:
                new_proto = torch.mean(torch.stack((self.global_protos[label], mean), dim=0),dim=0)
                # new_proto = F.normalize(new_proto, p=2, dim=0)
                self.global_protos[label] = new_proto
            local_proto.append(mean)
            index2tag[index] = label
            index += 1
        local_proto = torch.stack(local_proto, 0)
        if self.protocol == 'CI' or self.protocol == 'sup':
            return index2tag, local_proto
        else:
            return index2tag, self.global_protos

    def forward(self, sample):
        embedding = self.word_encoder(sample['sentence'], sample['attention_mask'])
        embedding = self.drop(embedding)  # [batch_size, max_len, 768]
        embedding = embedding[sample['text_mask']==1].view(-1, embedding.size(-1))
        index2tag, proto = self._update_protos(embedding, sample)
        logits = self._batch_dist(proto, embedding)  # [num_of_tokens, class_num]
        _, pred = torch.max(logits, 1)
        for index in index2tag:
            pred[pred == index] = index2tag[index]
        embedding = self.proj_head(embedding)  # [batch_size, max_len, 64]
        embedding = F.normalize(embedding, p=2, dim=1)
        # logger.info(f'embedding.shape: {embedding.shape}; logits.shape: {logits.shape}; pred.shape: {pred.shape}')
        return embedding, pred
    
    def loss(self, embedding, labels):
        return self._sup_contrastive_loss(embedding, labels)
    
    def _sup_contrastive_loss(self, embedding, labels):
        z_i, z_j = embedding.view(-1, embedding.size(-1)), embedding.view(-1, embedding.size(-1))
        dot_product_tempered = torch.mm(z_i, z_j.T) / self.temperature  # z_i dot z_j / tau
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        exp_dot_tempered = (
            torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0].detach()) + 1e-5
        )
        mask_combined = (labels.unsqueeze(1).repeat(1, labels.shape[0]) == labels) # n*m        
        cardinality_per_samples = torch.sum(mask_combined, dim=1)

        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered, dim=1, keepdim=True)))
        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples
        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)

        return supervised_contrastive_loss
    
    def _proto_contrastive_loss(self, embedding, labels):
        z_i, z_j = embedding.view(-1, embedding.size(-1)), embedding.view(-1, embedding.size(-1))
        dot_product_tempered = torch.mm(z_i, z_j.T) / self.temperature  # z_i dot z_j / tau
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        exp_dot_tempered = (
            torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0].detach()) + 1e-5
        )
        mask_combined = (labels.unsqueeze(1).repeat(1, labels.shape[0]) == labels) # n*m
        
        cardinality_per_samples = torch.sum(mask_combined, dim=1)

        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered, dim=1, keepdim=True)))
        proto_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples
        proto_contrastive_loss = torch.mean(proto_contrastive_loss_per_sample)

        return proto_contrastive_loss
    
    def _regulization(self):
        pass
