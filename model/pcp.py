import logging
import torch
torch.set_printoptions(profile='full')
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
                    # out_features=384)
        )
        self.global_protos = {}
        # self.proto_dist = {}
        # self.cached_mean = {}
        self.setting = setting
        self.temperature = temperature
        self.dot = dot
    
    def add_heads(self, label_set):
        for label in label_set:
            if label not in self.global_protos:
                self.add_module(f'shift_head_{label}', nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim).cuda())

    def __dist__(self, x, y, dim):
        if self.dot:
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)

    def _batch_dist(self, protos, embeddings):
        return self.__dist__(protos.unsqueeze(0), embeddings.unsqueeze(1), 2)

    def _update_protos(self, embedding, batch):
        tag = batch['label']
        tag = torch.cat(tag, 0)
        tag_set = set([int(_) for _ in tag.tolist()])
        tag_set.discard(self.ignore_index)
        index2tag = {}
        index = 0
        assert tag.size(0) == embedding.size(0)
        for label in tag_set:
            mean = torch.mean(embedding[tag==label], dim=0)
            # if label not in self.global_protos:
            #     self.global_protos[label] = mean
            # else:
            #     # !!! novel
            #     # new_proto = torch.mean(torch.stack((self.global_protos[label], mean), dim=0),dim=0)
            #     self.global_protos[label] = new_proto
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
            self.global_protos[label] = mean
                # self.cached_mean[label] = 1
            index2tag[index] = label
            index += 1
    
    def _get_all_protos(self):
        index2label, protos = {}, []
        index = 0
        for label in self.global_protos:
            protos.append(self.global_protos[label])
            index2label[index] = label
            index += 1
        protos = torch.stack(protos)
        return index2label, protos


    def contrastive_forward(self, batch):
        embedding = self.word_encoder(batch['sentence'], batch['attention_mask'])
        embedding = self.drop(embedding)  # [batch_size, max_len, 768]
        embedding = embedding[batch['text_mask']==1].view(-1, embedding.size(-1))
        embedding = F.normalize(embedding, p=2, dim=1)
        # if 'label' in batch:
        #     self._update_protos(embedding, batch)
        # index2tag, proto = self._get_all_protos()
        # logits = self._batch_dist(proto, embedding)  # [num_of_tokens, class_num]
        # _, pred = torch.max(logits, 1)
        # for index in index2tag:
        #     pred[pred == index] = index2tag[index]
        embedding = self.proj_head(embedding)  # [batch_size, max_len, contrastive_embed_dim]
        embedding = F.normalize(embedding, p=2, dim=1)
        # _, pred = torch.max(self.lc(embedding), 1)
        return embedding
    
    def forward(self, batch):
        embedding = self.word_encoder(batch['sentence'], batch['attention_mask'])
        embedding = self.drop(embedding)  # [batch_size, max_len, 768]
        embedding = embedding[batch['text_mask']==1].view(-1, embedding.size(-1))
        logits = self.lc(embedding)
        _, pred = torch.max(logits, 1)
        return logits, pred
    
    def _regulization(self):
        pass
