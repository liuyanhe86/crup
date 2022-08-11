import logging
import torch

from model import NERModel

logger = logging.getLogger(__name__)    

class ProtoNet(NERModel):
    
    def __init__(self, word_encoder, proto_update='SDC', metric='dot'):
        NERModel.__init__(self, word_encoder)
        self.proto_update = proto_update
        self.metric = metric
        self.global_protos = {}

    def __dist__(self, x, y, dim):
        if self.metric == 'dot':
            return (x * y).sum(dim)
        elif self.metric == 'L2':
            return -(torch.pow(x - y, 2)).sum(dim)
        else:
            raise NotImplementedError(f'ERROR: Invalid metric - {self.metric}')

    def __batch_dist__(self, proto, embedding):
        return self.__dist__(proto.unsqueeze(0), embedding.unsqueeze(1), 2)
    
    def _replace_or_mean(self, x):
        self.word_encoder.requires_grad_(False)
        rep = self.encode(x)
        self.word_encoder.requires_grad_(True)
        tag = torch.cat(x['label'], 0)
        assert tag.size(0) == rep.size(0)
        for label in range(torch.max(tag) + 1):
            mean = torch.mean(rep[tag==label], dim=0)
            if label not in self.global_protos:
                if not torch.isnan(mean).any():
                    self.global_protos[label] = mean
                else:
                    self.global_protos[label] = torch.zeros_like(mean)
            else:
                if not torch.isnan(mean).any():
                    # !!! novel
                    if self.proto_update == 'replace':
                        new_proto = mean
                    else:
                        new_proto = torch.mean(torch.stack((self.global_protos[label], mean), dim=0),dim=0)
                    self.global_protos[label] = new_proto

    def _semantic_drift_compensation(self):
        self.word_encoder.requires_grad_(False)
        z_c = self.encode(self.current_batch)
        self.word_encoder.requires_grad_(True)
        tag = torch.cat(self.current_batch['label'], 0)
        for i in range(torch.max(tag) + 1):
            if i not in self.global_protos:
                proto_i = torch.mean(z_c[tag == i], dim=0)
                if not torch.isnan(proto_i).any():
                    self.global_protos[i] = proto_i
                else:
                    self.global_protos[i] = torch.zeros_like(proto_i)
            else:
                if self.global_protos[i].any():
                    proto_i = self.global_protos[i]
                    dist = torch.sqrt(torch.sum(torch.pow(self.z_p - proto_i, 2), dim=1))
                    dist_mu = torch.mean(dist)
                    dist_variance = torch.mean(torch.pow(dist - dist_mu, 2))
                    assert z_c.shape[0] == self.z_p.shape[0], logger.info(f'Z_c:{z_c.shape}, Z_p:{self.z_p.shape}')
                    delta = z_c - self.z_p
                    w = torch.exp(-torch.pow(dist, 2) / (2 * dist_variance)) + 1e-5
                    Delta = torch.sum(torch.unsqueeze(w, 1) * delta, dim=0) / torch.sum(w)
                    self.global_protos[i] = proto_i + Delta
                else:
                    proto_i = torch.mean(z_c[tag==i], dim=0)
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

    def _init_protos(self, x):
        self.word_encoder.requires_grad_(False)
        z_c = self.encode(x)
        self.word_encoder.requires_grad_(True)
        tag = torch.cat(x['label'], 0)
        for i in range(torch.max(tag) + 1):
            if i not in self.global_protos:
                proto_i = torch.mean(z_c[tag == i], dim=0)
                if not torch.isnan(proto_i).any():
                    self.global_protos[i] = proto_i
                else:
                    self.global_protos[i] = torch.zeros_like(proto_i)

    def start_training(self):
        self.first_batch = True

    def _get_rep_logits_pred(self, x, mode='train'):
        if mode == 'train':
            if self.proto_update == 'replace' or self.proto_update == 'mean':
                self._replace_or_mean(x)
                rep = self.encode(x)
            elif self.proto_update == 'SDC':
                if self.first_batch:  # initialize
                    self._init_protos(x)
                    self.first_batch = False
                else:
                    self._semantic_drift_compensation()
                rep = self.encode(x)
                self.current_batch, self.z_p = x, rep.clone().detach_()
            else:
                raise NotImplementedError(f'ERROR: Invalid prototype update - {self.proto_update}')
        elif mode == 'test':
            rep = self.encode(x)
            
        index2label, protos = self._get_global_protos()
        logits = self.__batch_dist__(protos, rep)  # [num_of_tokens, class_num]
        _, pred = torch.max(logits, 1)
        for index in index2label:
            pred[pred == index] = index2label[index]
        return rep, logits, pred

    def train_forward(self, x):
        _, logits, pred = self._get_rep_logits_pred(x, mode='train')
        return logits, pred
    
    def forward(self, x):
        _, logits, pred = self._get_rep_logits_pred(x, mode='test')
        return logits, pred

    def finish_task(self):
        logger.info(f'current prototypes: {self.global_protos}')