import torch
from transformers import AutoTokenizer
import params


class Dataset:
    def __init__(self, text, target):
        self.text = text
        self.tokenizer = AutoTokenizer.from_pretrained(params.pretrained_model.path)
        self.target = target
    
    def __len__(self):
        return len(self.text)
    
    def get_tokens(self, text):
        inputs = self.tokenizer.encode_plus(
            text=text,
            padding="max_length",
            truncation=True,
            max_length=params.max_length
        )

        tokens = {}
        for key in inputs:
            tokens[key] = torch.tensor(inputs[key], dtype=torch.long)
        return tokens
    
    def __getitem__(self, item):
        text = str(self.text[item])
        text = " ".join(text.split())
        tokens = self.get_tokens(text)
        tokens['target'] = torch.tensor(self.target[item], dtype=torch.long)
        return tokens
