import logging
import torch
from torch import nn
import torch.nn.functional as F

from model import ProtoNet

logger = logging.getLogger(__name__)

class PCP(ProtoNet):
    
    def __init__(self, word_encoder, proto_update='SDC', metric='dot', embedding_dimension=64):
        ProtoNet.__init__(self, word_encoder, proto_update=proto_update, metric=metric)
        # self.proj_head = nn.Sequential(
        #     nn.Linear(in_features=word_encoder.output_dim, out_features=word_encoder.output_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(in_features=word_encoder.output_dim, out_features=embedding_dimension)
        # )
        self.f_mu = nn.Sequential(
            nn.ReLU(),
            nn.Linear(word_encoder.output_dim,
                      embedding_dimension)
        )
        self.f_sigma = nn.Sequential(
            nn.ReLU(),
            nn.Linear(word_encoder.output_dim,
                      embedding_dimension)
        )

    def train_forward(self, x):
        rep, _, pred = self._get_rep_logits_pred(x, mode='train')
        mu = self.f_mu(rep)
        sigma = F.elu(self.f_sigma(rep)) + 1 + 1e-14

        return mu, sigma, pred

    # def train_forward(self, x):
    #     rep, _, pred = self._get_rep_logits_pred(x, mode='train')
    #     embedding = F.normalize(rep, p=2, dim=1)
    #     embedding = self.proj_head(embedding)  # [num_tokens, contrastive_embed_dim]
    #     embedding = F.normalize(embedding, p=2, dim=1)
    #     return embedding, pred

    def reset_protos(self):
        self.index2label = None
        self.protos = None
    
    def _regulization(self):
        pass
