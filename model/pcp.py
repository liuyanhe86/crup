import logging
import torch
from torch import nn
import torch.nn.functional as F

from model import NERModel

logger = logging.getLogger(__name__)

class PCP(NERModel):
    
    def __init__(self, word_encoder, embedding_dimension=64, temperature=0.1, metric='dot', ignore_index=-1):
        NERModel.__init__(self, word_encoder, ignore_index)
        self.drop = nn.Dropout()
        self.embedding_dimension = embedding_dimension
        
        self.f_mu = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.word_encoder.output_dim,
                      self.embedding_dimension)
        )
        self.f_Sigma = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.word_encoder.output_dim,
                      self.embedding_dimension)
        )
        self.global_protos = {}
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

    def _semantic_drift_compensation(self, x):
        self.word_encoder.requires_grad_(False)
        Z_c = self.encode(x)
        self.word_encoder.requires_grad_(True)
        tag = torch.cat(x['label'], 0)
        for i in range(torch.max(tag) + 1):
            if i not in self.global_protos:
                proto_i = torch.mean(Z_c[tag == i], dim=0)
                if not torch.isnan(proto_i).any():
                    self.global_protos[i] = proto_i
                else:
                    self.global_protos[i] = torch.zeros_like(proto_i)
            else:
                if self.global_protos[i].any():
                    proto_i = self.global_protos[i]
                    dist = torch.sqrt(torch.sum(torch.pow(self.Z_p - proto_i, 2), dim=1))
                    dist_mu = torch.mean(dist)
                    dist_variance = torch.mean(torch.pow(dist - dist_mu, 2))
                    delta = Z_c - self.Z_p
                    w = torch.exp(-torch.pow(dist, 2) / (2 * dist_variance)) + 1e-5
                    Delta = torch.sum(torch.unsqueeze(w, 1) * delta, dim=0) / torch.sum(w)
                    self.global_protos[i] = proto_i + Delta
                else:
                    proto_i = torch.mean(Z_c[tag==i], dim=0)
                    if not torch.isnan(proto_i).any():
                        self.global_protos[i] = proto_i

    def _get_global_protos(self):
        index2label, protos = {}, []
        index = 0
        for label in self.global_protos:
            protos.append(self.global_protos[label])
            index2label[label] = index
            index += 1
        protos = torch.stack(protos)
        return index2label, protos

    def train_forward(self, x):
        if len(self.global_protos) == 0:  # initialize
            self._semantic_drift_compensation(x)
        else:
            self._semantic_drift_compensation(self.current_batch)
        rep = self.encode(x)
        self.current_batch, self.Z_p = x, rep.clone()
        self.Z_p.detach_()

        # index2label, protps = self._get_local_protos(embedding, x)
        index2label, protos = self._get_global_protos()
        
        logits = self._batch_dist(protos, rep)  # [num_of_tokens, class_num]
        _, pred = torch.max(logits, 1)
        for index in index2label:
            pred[pred == index] = index2label[index]
        
        # rep = F.normalize(rep, p=2, dim=1)
        # rep = self.proj_head(rep)  # [num_tokens, contrastive_embed_dim]
        # rep = F.normalize(rep, p=2, dim=1)

        original_embedding_mu = self.f_mu(rep)
        original_embedding_sigma = F.elu(self.f_Sigma(rep)) + 1 + 1e-14

        
        return rep, pred

    def forward(self, x):
        rep = self.encode(x)
        # embedding = F.normalize(embedding, p=2, dim=1)
        # embedding = self.proj_head(embedding)  # [num_tokens, contrastive_embed_dim]
        # embedding = F.normalize(embedding, p=2, dim=1)
        index2label, protos = self._get_global_protos()
        logits = self._batch_dist(protos, rep)  # [num_of_tokens, class_num]
        _, pred = torch.max(logits, 1)
        for index in index2label:
            pred[pred == index] = index2label[index]
        return rep, pred

    def reset_protos(self):
        self.index2label = None
        self.protos = None
    
    def _regulization(self):
        pass
