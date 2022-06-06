import logging
import torch
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
        for _ in self.global_protos:
            protos.append(self.global_protos[_])
        protos = torch.stack(protos)
        return protos


    def forward(self, batch):
        batch_emb = self.word_encoder(batch['sentence'], batch['attention_mask'])
        batch_emb = self.drop(batch_emb)  # [batch_size, max_len, 768]
        # Prototypical Networks
        
        # Calculate prototype for each class
        batch_emb = batch_emb[batch['text_mask']==1].view(-1, batch_emb.size(-1))
        if 'label' in batch:
            protos = self._get_local_protos(batch_emb, batch)
        else:
            protos = self._get_global_protos()
        # calculate distance to each prototype
        logits = self.__batch_dist__(protos, batch_emb)
        _, pred = torch.max(logits, 1)  # [num_of_tokens]
        return logits, pred
