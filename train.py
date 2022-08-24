import pandas as pd
import torch
from torch.optim import AdamW
import utils
from model import Model
from transformers import get_scheduler
from collections import defaultdict
import engine
from visualize import save_acc_curves, save_loss_curves
import params
import os
from evaluate import evaluate


def get_optimizer(model):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        params=optimizer_parameters,
        lr=params.learning_rate,
        weight_decay=params.weight_decay,
        eps=params.epsilon
    )

    return optimizer


def run():
    train, valid, _ = utils.read_data()
    train_data_loader = utils.get_dataloader(train)
    val_data_loader = utils.get_dataloader(valid)

    device = utils.set_device()
    model = Model()
    model = model.to(device)
    optimizer = get_optimizer(model)

    num_train_steps = int(len(train) / params.batch_size * params.epochs)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_train_steps * 0.2,
        num_training_steps=num_train_steps
    )

    utils.log("")
    utils.log("##################################### Training ############################################")

    history = defaultdict(list)
    best_acc = 0.0

    for epoch in range(params.epochs):
        train_acc, train_loss = engine.train(train_data_loader, model, device, optimizer, scheduler)
        utils.log(f'Epoch {epoch + 1} --- Training loss: {train_loss} Training accuracy: {train_acc}')
        
        val_acc, val_loss = engine.evaluate(val_data_loader, model, device)
        utils.log(f'Epoch {epoch + 1} --- Validation loss: {val_loss} Validation accuracy: {val_acc}')
        
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        
        if val_acc > best_acc:
            old_model = params.output_folder + f"/Model.bin"
            if os.path.exists(old_model):
                os.remove(old_model)
            torch.save(model.state_dict(), params.output_folder + f"/Model.bin")
            best_acc = val_acc

    pd.DataFrame(history).to_csv(params.output_folder + "/History.csv")
    save_acc_curves(history)
    save_loss_curves(history)

    del train_data_loader, val_data_loader

    utils.log("")
    evaluate(model)


if __name__ == "__main__":
    params.max_length = 128
    params.batch_size = 16
    params.learning_rate = 1e-05
    params.epochs = 15
    params.dropout = 0.4
    params.hidden_units = 64
    params.weight_decay = 1e-02
    params.epsilon = 1e-08
    params.pretrained_model = params.models.bert
    params.device = params.devices.cuda0
    params.data_path = "./data.txt"
    params.output_folder = None

    run()
