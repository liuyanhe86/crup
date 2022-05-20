import logging
import torch
from torch import nn
import torch.nn.functional as F

from model import NERModel

logger = logging.getLogger(__name__)

class CPR(NERModel):
    
    def __init__(self, 
                word_encoder, 
                protocol='CI', 
                temperature=0.5,
                dot=False,
                cpl=False, 
                embedding_dim=64):
        NERModel.__init__(self, word_encoder)
        self.drop = nn.Dropout()
        self.proj_head = nn.Sequential(
            nn.Linear(in_features=word_encoder.output_dim, out_features=word_encoder.output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=word_encoder.output_dim, out_features=embedding_dim)
        )
        self.global_protos = {}
        self.protocol = protocol
        self.temperature = temperature
        self.embedding_dim = embedding_dim
        self.dot = dot
        self.cpl = cpl
    
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
            if label not in self.global_protos:
                self.global_protos[label] = mean
            else:
                # !!! novel
                new_proto = torch.mean(torch.stack((self.global_protos[label], mean), dim=0),dim=0)
                self.global_protos[label] = new_proto
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


    def forward(self, batch):
        embedding = self.word_encoder(batch['sentence'], batch['attention_mask'])
        embedding = self.drop(embedding)  # [batch_size, max_len, 768]
        embedding = embedding[batch['text_mask']==1].view(-1, embedding.size(-1))
        if 'label' in batch:
            self._update_protos(embedding, batch)
        index2tag, proto = self._get_all_protos()
        logits = self._batch_dist(proto, embedding)  # [num_of_tokens, class_num]
        _, pred = torch.max(logits, 1)
        for index in index2tag:
            pred[pred == index] = index2tag[index]
        embedding = self.proj_head(embedding)  # [batch_size, max_len, 64]
        embedding = F.normalize(embedding, p=2, dim=1)
        # logger.info(f'embedding.shape: {embedding.shape}; logits.shape: {logits.shape}; pred.shape: {pred.shape}')
        return embedding, pred
    
    def loss(self, embedding, labels):
        if self.cpl:
            return self._sup_contrastive_loss(embedding, labels) + self._proto_contrastive_loss(embedding, labels)
        else:
            return self._sup_contrastive_loss(embedding, labels)
    
    def _sup_contrastive_loss(self, embedding, labels):
        z_i, z_j = embedding.view(-1, embedding.size(-1)), embedding.view(-1, embedding.size(-1))
        dot_product_tempered = torch.mm(z_i, z_j.T) / self.temperature  # z_i dot z_j / tau
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        exp_dot_tempered = (
            torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0].detach()) + 1e-5
        )
        mask_combined = (labels.unsqueeze(1).repeat(1, labels.shape[0]) == labels) # n*m        
        cardinality_per_batchs = torch.sum(mask_combined, dim=1)

        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered, dim=1, keepdim=True)))
        supervised_contrastive_loss_per_batch = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_batchs
        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_batch)

        return supervised_contrastive_loss
    
    def _proto_contrastive_loss(self, embedding, labels):
        protos, proto_labels = [], []
        for label in self.global_protos:
            protos.append(self.global_protos[label])
            proto_labels.append(label)
        protos = torch.stack(protos)
        proto_labels = torch.Tensor(proto_labels)
        if torch.cuda.is_available():
            protos.cuda()
            proto_labels.cuda()
        z_i, z_j = embedding.view(-1, embedding.size(-1)), protos.view(-1, embedding.size(-1))
        dot_product_tempered = torch.mm(z_i, z_j.T) / self.temperature  # z_i dot z_j / tau
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        exp_dot_tempered = (
            torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0].detach()) + 1e-5
        )
        mask_combined = (labels.unsqueeze(1).repeat(1, labels.shape[0]) == proto_labels) # n*m        
        cardinality_per_batchs = torch.sum(mask_combined, dim=1)

        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered, dim=1, keepdim=True)))
        proto_contrastive_loss_per_batch = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_batchs
        proto_contrastive_loss = torch.mean(proto_contrastive_loss_per_batch)

        return proto_contrastive_loss
    
    def _regulization(self):
        pass
