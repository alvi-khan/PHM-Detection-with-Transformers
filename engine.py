import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score


def loss_fn(output, target):
    return nn.CrossEntropyLoss()(output, target)


def classify(output):
    output = torch.log_softmax(output, dim=1)
    probabilities = output
    output = torch.argmax(output, dim=1)
    return probabilities, output


def process(data_loader, model, device, optimizer=None, scheduler=None, calculate_loss=True):
    progress_bar = tqdm(data_loader, total=len(data_loader))
    losses = []
    targets = []
    outputs = []
    probabilities = []

    def get_loss(output, target):
        loss = loss_fn(output, target)
        losses.append(loss.item())
        if model.training:
            progress_bar.set_postfix(loss=np.mean(losses))
        return loss
    
    def backprop(loss):
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
    
    def update_stats(output, target, probability):
        targets.extend(target.cpu().detach().numpy().tolist())
        outputs.extend(output.cpu().detach().numpy().tolist())
        probabilities.extend(probability.cpu().detach().numpy().tolist())
    
    def get_stats():
        if not calculate_loss:
            return outputs, targets, probabilities
        
        f1 = f1_score(targets, outputs, average='weighted')
        f1 = np.round(f1.item(), 4)
        return f1, np.mean(losses)

    for data in progress_bar:
        for key in data:
            data[key] = data[key].to(device, dtype=torch.long)
        target = data.pop('target')
        
        model.zero_grad()
        output = model(data)
        if calculate_loss:
            loss = get_loss(output, target)
        probability, output = classify(output)
        if model.training:
            backprop(loss)
        update_stats(output, target, probability)

    return get_stats()


def train(data_loader, model, device, optimizer, scheduler):
    model.train()
    return process(data_loader, model, device, optimizer, scheduler)


def evaluate(data_loader, model, device):
    model.eval()
    with torch.no_grad():
        stats = process(data_loader, model, device)
    return stats


def test(data_loader, model, device):
    model.eval()
    with torch.no_grad():
        stats = process(data_loader, model, device, calculate_loss=False)
    return stats
