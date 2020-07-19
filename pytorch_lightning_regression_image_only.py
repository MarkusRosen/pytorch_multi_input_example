import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

data_path = "./data/"


class ImageDataset(Dataset):
    """Tabular and Image dataset."""

    def __init__(self, pickle_file, image_dir):
        self.image_dir = image_dir
        self.pickle_file = pickle_file

        self.tabular = pd.read_pickle(pickle_file)

    def __len__(self):
        return len(self.tabular)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        tabular = self.tabular.iloc[idx, 0:]

        y = tabular["price"]

        image = Image.open(f"{self.image_dir}/{tabular['zpid']}.png")
        image = np.array(image)

        image = image[..., :3]

        image = transforms.functional.to_tensor(image)

        tabular = tabular[["latitude", "longitude", "beds", "baths", "area"]]
        tabular = tabular.tolist()
        tabular = torch.FloatTensor(tabular)

        return image, y


def conv_block(input_size, output_size):
    block = nn.Sequential(
        nn.Conv2d(input_size, output_size, (3, 3)), nn.ReLU(), nn.BatchNorm2d(output_size), nn.MaxPool2d((2, 2)),
    )

    return block


class LitClassifier(pl.LightningModule):
    def __init__(
        self, lr: float = 1e-3, num_workers: int = 4, batch_size: int = 32,
    ):
        super().__init__()
        self.lr = lr
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.conv1 = conv_block(3, 16)
        self.conv2 = conv_block(16, 32)
        self.conv3 = conv_block(32, 64)

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

        x = self.dropout(x)
        x = self.ln2(x)
        x = self.relu(x)

        return self.ln3(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        criterion = torch.nn.L1Loss()
        y_pred = torch.flatten(self(x))
        y_pred = y_pred.double()

        loss = criterion(y_pred, y)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch

        criterion = torch.nn.L1Loss()
        y_pred = torch.flatten(self(x))
        y_pred = y_pred.double()

        val_loss = criterion(y_pred, y)

        return {"val_loss": val_loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, y = batch

        criterion = torch.nn.L1Loss()
        y_pred = torch.flatten(self(x))
        y_pred = y_pred.double()

        test_loss = criterion(y_pred, y)

        return {"test_loss": test_loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        logs = {"test_loss": avg_loss}
        return {"test_loss": avg_loss, "log": logs, "progress_bar": logs}

    def setup(self, stage):

        image_data = ImageDataset(pickle_file=f"{data_path}df.pkl", image_dir=f"{data_path}processed_images/")

        train_size = int(0.80 * len(image_data))
        val_size = int((len(image_data) - train_size) / 2)
        test_size = int((len(image_data) - train_size) / 2)

        self.train_set, self.val_set, self.test_set = random_split(image_data, (train_size, val_size, test_size))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=(self.lr))

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)


if __name__ == "__main__":
    logger = TensorBoardLogger("lightning_logs", name="image_only")
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=5000, patience=7, verbose=False, mode="min")

    model = LitClassifier()
    trainer = pl.Trainer(gpus=1, logger=logger, early_stop_callback=early_stop_callback)

    lr_finder = trainer.lr_find(model)
    fig = lr_finder.plot(suggest=True, show=True)
    new_lr = lr_finder.suggestion()
    print(new_lr)
    model.hparams.lr = new_lr

    trainer.fit(model)
    trainer.test(model)
