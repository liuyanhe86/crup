import torch
import torch.nn as nn
from transformers import BertModel

class BERTWordEncoder(nn.Module):

    def __init__(self, pretrain_path): 
        nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.output_dim = 768

    def forward(self, sentences, attention_masks):
        outputs = self.bert(sentences, attention_mask=attention_masks, output_hidden_states=True, return_dict=True)
            
        # use the sum of the last 4 layers
        last_four_hidden_states = torch.cat([hidden_state.unsqueeze(0) for hidden_state in outputs['hidden_states'][-4:]], 0)
        del outputs
        word_embeddings = torch.sum(last_four_hidden_states, 0) # [num_sent, number_of_tokens, 768]
        
        return word_embeddings
