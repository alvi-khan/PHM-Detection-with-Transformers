import torch
import torch.nn as nn
from transformers import AutoModel
import params
import utils


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained(params.pretrained_model.path)
        self.drop1 = nn.Dropout(params.dropout)
        self.linear = nn.Linear(params.pretrained_model.value.output_units, params.hidden_units)
        self.batch_norm = nn.LayerNorm(params.hidden_units)
        self.drop2 = nn.Dropout(params.dropout)
        self.out = nn.Linear(params.hidden_units, 8)

        utils.log(f'Model Parameter Count: {utils.count_model_parameters(self)}')

    def forward(self, data):
        output = self.model(**data)
        output = output['pooler_output'] if 'pooler_output' in output else torch.mean(output['last_hidden_state'], 1)
        
        output = self.drop1(output)
        output = self.linear(output)
        output = self.batch_norm(output)
        output = nn.Tanh()(output)
        output = self.drop2(output)
        output = self.out(output)
        
        return output
