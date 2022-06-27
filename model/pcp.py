import logging
import torch
from torch import nn
import torch.nn.functional as F

from model import NERModel

logger = logging.getLogger(__name__)

class PCP(NERModel):
    
    def __init__(self, word_encoder, temperature=0.1, metric='dot'):
        NERModel.__init__(self, word_encoder)
        self.drop = nn.Dropout()
        self.proj_head = nn.Sequential(
            nn.Linear(in_features=word_encoder.output_dim, out_features=word_encoder.output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=word_encoder.output_dim, 
                      out_features=word_encoder.output_dim)
        )
        self.global_protos = {}
        self.index2label = None
        self.protos = None
        self.temperature = temperature
        self.metric = metric

    def __dist__(self, x, y, dim):
        if self.metric == 'dot':
            return (x * y).sum(dim)
        elif self.metric == 'L2':
            return -(torch.pow(x - y, 2)).sum(dim)
        else:
            raise NotImplementedError(f'ERROR: Invalid metric - {self.metric}')

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
    
    def update_global_protos(self, proto_dict):
        index2label, protos = {}, []
        index = 0
        for label in proto_dict:
            protos.append(proto_dict[label])
            index2label[index] = label
            index += 1
        protos = torch.stack(protos)
        self.index2label, self.protos = index2label, protos

    def _update_protos(self):
        self.word_encoder.requires_grad_(False)
        Z_c = self.encode(self.current_batch)
        self.word_encoder.requires_grad_(True)

        tag = self.current_batch['label']
        tag = torch.cat(tag, 0)
        tag_set = set([int(_) for _ in tag.tolist()])
        tag_set.discard(self.ignore_index)
        for i in tag_set:
            Z_c_i = Z_c[tag==i]
            if i not in self.global_protos:
                self.global_protos[i] = torch.mean(Z_c_i, dim=0)
            else:
                mu_i = self.global_protos[i]
                dist = torch.sum(torch.pow(Z_c - mu_i, 2), dim=1).mean()
                # dist_mean = 
                pass

    def train_forward(self, batch):
        embedding = self.encode(batch)

        self.current_batch = batch
        self.Z_p = embedding
        index2tag, protos = self._get_local_protos(embedding, batch)
        
        logits = self._batch_dist(protos, embedding)  # [num_of_tokens, class_num]
        _, pred = torch.max(logits, 1)
        for index in index2tag:
            pred[pred == index] = index2tag[index]
        
        embedding = F.normalize(embedding, p=2, dim=1)
        embedding = self.proj_head(embedding)  # [num_tokens, contrastive_embed_dim]
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding, pred

    def forward(self, x):
        embedding = self.encode(x)
        embedding = F.normalize(embedding, p=2, dim=1)
        embedding = self.proj_head(embedding)  # [num_tokens, contrastive_embed_dim]
        embedding = F.normalize(embedding, p=2, dim=1)
        logits = self._batch_dist(self.protos, embedding)  # [num_of_tokens, class_num]
        _, pred = torch.max(logits, 1)
        for index in self.index2label:
            pred[pred == index] = self.index2label[index]
        return embedding, pred

    def reset_protos(self):
        self.index2label = None
        self.protos = None
    
    def _regulization(self):
        pass
