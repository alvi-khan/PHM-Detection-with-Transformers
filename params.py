from enum import Enum
from types import DynamicClassAttribute


class Model:
    def __init__(self, path, output_units):
        self.path = path
        self.output_units = output_units


class models(Enum):
    bert = Model(path="bert-base-uncased", output_units=768)
    roberta = Model(path="roberta-base", output_units=768)
    xlnet = Model(path="xlnet-base-cased", output_units=768)

    @DynamicClassAttribute
    def path(self):
        return self.value.path
    
    @DynamicClassAttribute
    def output_units(self):
        return self.value.output_units


class devices(str, Enum):
    cpu='cpu'
    cuda0='cuda:0'
    cuda1='cuda:1'


max_length = 128
batch_size = 16
learning_rate = 1e-05
epochs = 15
dropout = 0.4
hidden_units = 64
weight_decay = 1e-02
epsilon = 1e-08
pretrained_model = models.bert
device = devices.cuda0
data_path = "./data.txt"
output_folder = None
