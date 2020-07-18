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


class ImageNet(nn.Module):
    def __init__(self):
        super().__init__()
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
        x = self.ln3(x)
        # print(x)
        return x


def train(model, device, train_loader):
    start_time = datetime.now()
    model.train()
    running_loss = 0.0
    for i_batch, local_batch in enumerate(train_loader):
        local_batch_X = local_batch[0]
        local_batch_y = local_batch[1]

        local_batch_X, local_batch_y = (
            local_batch_X.to(device),
            local_batch_y.to(device),
        )

        optimizer.zero_grad()
        y_pred = model(local_batch_X.float())

        loss = criterion(torch.flatten(y_pred), local_batch_y.float())

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    end_time = datetime.now() - start_time
    return epoch_loss


if __name__ == "__main__":
    torch.manual_seed(42)
    random_seed = 42
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    image_data = ImageDataset(pickle_file=f"{data_path}df.pkl", image_dir=f"{data_path}processed_images/")
    params = {"batch_size": 4, "shuffle": True, "num_workers": 4}
    max_epochs = 200

    train_loader = DataLoader(image_data, **params)
    # for x in train_loader:
    #    print(x)
    model = ImageNet()
    model = model.to(device)
    print(model)
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-3 / 200)

    for epoch in range(max_epochs):
        start_time = datetime.now()
        training_loss = train(model, device, train_loader)
        print(training_loss)

# TODO: boxplots, remove outliers
# TODO: validation loop
# TODO: add tabular data network -> check dimensions (before was 4, now should be 5)
# TODO: print nicer
# TODO: add simple plot
