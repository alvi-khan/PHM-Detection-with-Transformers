import torch
from dataset import Dataset
from torch.utils.data import DataLoader
import params
import datetime
import os
from sklearn.preprocessing import LabelEncoder
import io
import pandas as pd
import numpy as np


def read_data():
    file = open(params.data_path, 'r', encoding="utf8")
    data = file.read()
    file.close()

    data = data.replace('\t', ',|')
    df = pd.read_csv(filepath_or_buffer=io.StringIO(data), names=["domain", "label", "text"], sep='\,\|', engine='python', header=None)
    
    df["target"] = df["domain"] + df["label"].replace({0: "N", 1: "N", 2: "Y", 3: "Y"})

    le = LabelEncoder()
    le.fit(df["target"])
    df["target"] = le.transform(df["target"])

    return np.split(df.sample(frac=1, random_state=42), [int(.6 * len(df)), int(.8 * len(df))])


def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_device():
    if params.device == params.devices.cpu:
        return "cpu"
    
    device = torch.device(params.device.value if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        log("GPU not available.")
    
    return device


def get_dataloader(df):
    dataset = Dataset(df.text.values, df.target.values)

    return DataLoader(
        dataset=dataset,
        batch_size=params.batch_size,
        shuffle=False
    )


def create_output_folder():
    date_time = datetime.datetime.now().strftime("%Y-%m-%d %I %M %p ")
    model_name = params.pretrained_model.name
    params.output_folder = date_time + model_name
    if not os.path.exists(params.output_folder):
        os.makedirs(params.output_folder)


def log(text, filename='Logs.txt', display=True):
    if params.output_folder is None:
        create_output_folder()
    if display:
        print(text)
    file = open(f"{params.output_folder}/{filename}", 'a+')
    file.write(str(text) + "\n")
    file.close()
