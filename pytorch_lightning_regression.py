import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import matplotlib.pyplot as plt

data_path = "./data/"


class ImageDataset(Dataset):
    """Tabular and Image dataset."""

    def __init__(self, pickle_file, image_dir):
        self.image_dir = image_dir
        self.pickle_file = pickle_file

        self.tabular = pd.read_pickle(pickle_file)

        print(self.tabular)

    def __len__(self):
        return len(self.tabular)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        tabular = self.tabular.iloc[idx, 0:]

        y = tabular["unformattedPrice"]

        image = Image.open(f"{self.image_dir}/{tabular['zpid']}.png")
        image = np.array(image)
        image = image[..., :3]

        image = transforms.functional.to_tensor(image)

        tabular = tabular[["latLong_latitude", "latLong_longitude", "beds", "baths", "area"]]
        tabular = tabular.tolist()
        tabular = torch.FloatTensor(tabular)

        return image, y


def conv_block(input_size, output_size):
    block = nn.Sequential(
        nn.Conv2d(input_size, output_size, (3, 3)), nn.ReLU(), nn.BatchNorm2d(output_size), nn.MaxPool2d((2, 2)),
    )

    return block


class LitClassifier(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.conv1 = conv_block(3, 16)
        self.conv2 = conv_block(16, 32)
        self.conv3 = conv_block(32, 64)
        # conv2d -> -2 pixels
        # max pool -> pixels/2
        # remainder will be dropped
        self.ln1 = nn.Linear(64 * 26 * 26, 16)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(16)
        self.dropout = nn.Dropout2d(0.5)
        self.ln2 = nn.Linear(16, 4)
        self.ln3 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.reshape(x.shape[0], -1)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = self.ln2(x)
        x = self.relu(x)
        # x = self.ln3(x)
        # print(x)
        return self.ln3(x)

    def train_dataloader(self):
        return DataLoader(image_data, batch_size=32)

    def training_step(self, batch, batch_nb):
        x, y = batch
        # print(x)
        # print(y)
        # print(self(x))
        # print(y)
        # print(torch.flatten(self(x)))
        criterion = torch.nn.L1Loss()
        y_pred = torch.flatten(self(x))
        y_pred = y_pred.double()
        # loss =  torch.sqrt(criterion(y_pred, y))
        loss = criterion(y_pred, y)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=(self.lr))


if __name__ == "__main__":
    image_data = ImageDataset(pickle_file=f"{data_path}df.pkl", image_dir=f"{data_path}processed_images/")

    model = LitClassifier()
    # mlflow_logger = pl_loggers.MLFlowLogger("logs/")
    trainer = pl.Trainer(gpus=1)

    lr_finder = trainer.lr_find(model)
    print(lr_finder.results)
    fig = lr_finder.plot(suggest=True, show=True)

    new_lr = lr_finder.suggestion()
    print(new_lr)
    model.hparams.lr = new_lr  # 1e-2

    trainer.fit(model)
