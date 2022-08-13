import logging
from torch import nn
import torch.nn.functional as F

from model import ProtoNet

logger = logging.getLogger(__name__)

class CRUP(ProtoNet):
    
    def __init__(self, word_encoder, proto_update='SDC', metric='dot', embedding_dimension=64):
        ProtoNet.__init__(self, word_encoder, proto_update=proto_update, metric=metric)
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
