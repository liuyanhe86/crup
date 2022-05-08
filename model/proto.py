import logging
import torch
from torch import nn

from model import NERModel

logger = logging.getLogger(__name__)        

class ProtoNet(NERModel):
    
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
        tag = torch.cat(tag, 0)
        tag_set = set([int(_) for _ in tag.tolist()]) - {self.ignore_index}
        proto = []
        index2tag = {}
        index = 0
        assert tag.size(0) == embedding.size(0)
        for label in tag_set:
            mean = torch.mean(embedding[tag==label], dim=0)
            proto.append(mean)
            index2tag[index] = label
            index += 1
        proto = torch.stack(proto, 0)
        return 

    def forward(self, sample):
        sample_emb = self.word_encoder(sample['sentence'], sample['attention_mask'])
        sample_emb = self.drop(sample_emb)  # [batch_size, max_len, 768]
        # Prototypical Networks
        
        # Calculate prototype for each class
        proto = self.__get_proto__(sample_emb, sample['label'], sample['text_mask'])  # [class_num, 768]
        # calculate distance to each prototype
        logits = self.__batch_dist__(proto, sample_emb, sample['text_mask'])  # [num_of_tokens, class_num]
        _, pred = torch.max(logits, 1)  # [num_of_tokens]
        # logger.info(f'foward: pred={pred}; pred.shape={pred.shape}')
        return logits, pred
