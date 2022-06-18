import logging
import torch
torch.set_printoptions(profile='full')
from torch import nn
import torch.nn.functional as F

from model import NERModel

logger = logging.getLogger(__name__)        

class ProtoNet(NERModel):
    
    def __init__(self,word_encoder, dot=True):
        NERModel.__init__(self, word_encoder)
        self.drop = nn.Dropout()
        self.dot = dot
        self.global_protos = {}
        # self.proj = nn.Linear(in_features=2 * word_encoder.output_dim, out_features=word_encoder.output_dim)

    def __dist__(self, x, y, dim):
        if self.dot:
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, proto, embedding):
        return self.__dist__(proto.unsqueeze(0), embedding.unsqueeze(1), 2)
    
    def _get_local_protos(self, embedding, batch):
        tag = batch['label']
        tag = torch.cat(tag, 0)
        assert tag.size(0) == embedding.size(0)
        local_protos = []
        for label in range(torch.max(tag) + 1):
            mean = torch.mean(embedding[tag==label], dim=0)
            if label not in self.global_protos:
                if not torch.isnan(mean).any():
                    self.global_protos[label] = mean
                    local_protos.append(mean)
                else:
                    self.global_protos[label] = torch.zeros_like(mean)
                    local_protos.append(torch.zeros_like(mean))
            else:
                if not torch.isnan(mean).any():
                # !!! novel
                    new_proto = torch.mean(torch.stack((self.global_protos[label], mean), dim=0),dim=0)
                    # new_proto = self.proj(torch.cat((self.global_protos[label], mean)))
                    self.global_protos[label] = new_proto
                    # self.global_protos[label] = mean
                    local_protos.append(mean)
                else:
                    local_protos.append(torch.zeros_like(mean))
        local_protos = torch.stack(local_protos)
        return local_protos
                
    
    def _get_global_protos(self):
        protos = []
        for label in self.global_protos:
            protos.append(self.global_protos[label])
        protos = torch.stack(protos)
        return protos

    def _update_protos(self, x):
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
                    w = torch.exp(-torch.pow(dist, 2) / (2 * dist_variance))
                    Delta = torch.sum(torch.unsqueeze(w, 1) * delta, dim=0) / torch.sum(w)
                    self.global_protos[i] = proto_i + Delta
                else:
                    proto_i = torch.mean(Z_c[tag==i], dim=0)
                    if not torch.isnan(proto_i).any():
                        self.global_protos[i] = proto_i

    def train_forward(self, x):
        if len(self.global_protos) == 0:
            logger.info('init prototypes')
            self._update_protos(x)
        else:
            logger.info('update prototypes')
            self._update_protos(self.current_batch)
        embedding = self.encode(x)
        self.current_batch,self.Z_p = x, embedding
        protos = self._get_global_protos()
        # calculate distance to each prototype
        logits = self.__batch_dist__(protos, embedding)
        _, pred = torch.max(logits, 1)  # [num_of_tokens]
        logger.info(f'min pred: {torch.min(pred)}; max pred: {torch.max(pred)}; min labels: {torch.min(torch.cat(x["label"]))}; max labels: {torch.max(torch.cat(x["label"]))}')
        return logits, pred
    
    def forward(self, x):
        embedding = self.encode(x)
        index2tag, protos = self._get_global_protos()
        # calculate distance to each prototype
        logits = self.__batch_dist__(protos, embedding)
        _, pred = torch.max(logits, 1)  # [num_of_tokens]
        for index in index2tag:
            pred[pred == index] = index2tag[index]
        return logits, pred
