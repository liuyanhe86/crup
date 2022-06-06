import logging
import torch
from torch import nn
import torch.nn.functional as F

from model import NERModel

logger = logging.getLogger(__name__)

class PCP(NERModel):
    
    def __init__(self, 
                word_encoder, 
                setting='CI', 
                temperature=0.5,
                dot=False):
        NERModel.__init__(self, word_encoder)
        self.drop = nn.Dropout()
        self.proj_head = nn.Sequential(
            nn.Linear(in_features=word_encoder.output_dim, out_features=word_encoder.output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=word_encoder.output_dim, 
                      out_features=word_encoder.output_dim)
        )
        self.global_protos = {}
        self.setting = setting
        self.temperature = temperature
        self.dot = dot
    
    def __dist__(self, x, y, dim):
        if self.dot:
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)

    def _batch_dist(self, protos, embeddings):
        return self.__dist__(protos.unsqueeze(0), embeddings.unsqueeze(1), 2)

    def _get_local_protos(self, embedding, batch):
        tag = batch['label']
        tag = torch.cat(tag, 0)
        tag_set = set([int(_) for _ in tag.tolist()])
        tag_set.discard(self.ignore_index)
        index2tag = {}
        index = 0
        assert tag.size(0) == embedding.size(0)
        local_protos = []
        for label in tag_set:
            mean = torch.mean(embedding[tag==label], dim=0)
            if label not in self.global_protos:
                self.global_protos[label] = mean
            else:
                # !!! novel
                new_proto = torch.mean(torch.stack((self.global_protos[label], mean), dim=0),dim=0)
                self.global_protos[label] = new_proto
            # if label in self.global_protos:
            #     # dist = torch.pow(mean - self.global_protos[label], 2).sum()                
            #     # dist = (mean * self.global_protos[label]).sum()
            #     # if label not in self.proto_dist or dist > self.proto_dist[label]:
            #     #     self.proto_dist[label] = dist
            #     #     self.global_protos[label] = mean
                
            #     self.cached_mean[label] += 1
            #     if self.cached_mean[label] == 10:
            #         self.global_protos[label] = mean
            # else:
            # self.global_protos[label] = mean
                # self.cached_mean[label] = 1
            local_protos.append(mean)
            index2tag[index] = label
            index += 1
        local_protos = torch.stack(local_protos)
        return index2tag, local_protos
    
    def _get_global_protos(self):
        index2label, protos = {}, []
        index = 0
        for label in self.global_protos:
            protos.append(self.global_protos[label])
            index2label[index] = label
            index += 1
        protos = torch.stack(protos)
        return index2label, protos


    def forward(self, batch):
        embedding = self.word_encoder(batch['sentence'], batch['attention_mask'])
        embedding = self.drop(embedding)  # [batch_size, max_len, 768]
        embedding = embedding[batch['text_mask']==1]  #.view(-1, embedding.size(-1))
        if 'label' in batch:
            index2tag, protos = self._get_local_protos(embedding, batch)
        else:
            index2tag, protos = self._get_global_protos()
        logits = self._batch_dist(protos, embedding)  # [num_of_tokens, class_num]
        _, pred = torch.max(logits, 1)
        for index in index2tag:
            pred[pred == index] = index2tag[index]
        embedding = F.normalize(embedding, p=2, dim=1)
        embedding = self.proj_head(embedding)  # [num_tokens, contrastive_embed_dim]
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding, pred
    
    def loss(self, embedding, labels):
        return self._supcon_loss(embedding, labels)
    
    def _supcon_loss(self, embedding, labels):
        embedding = embedding[labels != self.ignore_index]
        labels = labels[labels != self.ignore_index]
        z_i, z_j = embedding, embedding.T
        dot_product_tempered = torch.mm(z_i, z_j) / self.temperature  # z_i dot z_j / tau
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        exp_dot_tempered = (
            torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        )
        mask_similar_class = (labels.unsqueeze(1).repeat(1, labels.shape[0]) == labels)
        if torch.cuda.is_available():
            mask_similar_class.cuda()
        if torch.cuda.is_available():
            mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0]).cuda())
        else:
            mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0]))
        mask_combined = mask_similar_class * mask_anchor_out
        cardinality_per_batchs = torch.sum(mask_combined, dim=1)
        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered, dim=1, keepdim=True)))
        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_anchor_out, dim=1) / cardinality_per_batchs
        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)
        return supervised_contrastive_loss
    
    def _regulization(self):
        pass
