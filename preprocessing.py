from os import remove
import pandas as pd
import os
from typing import List, Dict
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from torch import float32

data_path = "./data/"


def select_rows_with_images(images: List[str], df: pd.DataFrame) -> pd.DataFrame:
    dicts: List[Dict] = []
    for index, row in df.iterrows():
        image_file = row["zpid"] + ".png"
        if image_file in images:
            data_row = row.to_dict()
            dicts.append(data_row)

    df_new = pd.DataFrame(dicts)

    df = df_new[["zpid", "unformattedPrice", "latLong_latitude", "latLong_longitude", "beds", "baths", "area"]]

    return df


def resize_images(images: List[str]):
    for i in images:
        image = Image.open(f"{data_path}images/{i}")
        new_image = image.resize((224, 224))
        new_image.save(f"{data_path}processed_images/{i}")


def remove_outliers(df: pd.DataFrame, col: str):
    q_low = df[col].quantile(0.02)
    q_hi = df[col].quantile(0.98)
    df_filtered = df[(df[col] < q_hi) & (df[col] > q_low)]

    return df_filtered


images = os.listdir(f"{data_path}images")
df = pd.read_pickle(f"{data_path}ny_dataframe.pkl")
# print(images)


df = select_rows_with_images(images, df)
# df = df.iloc[0:800]

df["unformattedPrice"] = df["unformattedPrice"].astype(float)
df["latLong_latitude"] = df["latLong_latitude"].astype(float)
df["latLong_longitude"] = df["latLong_longitude"].astype(float)
df["beds"] = df["beds"].astype(float)
df["baths"] = df["baths"].astype(float)
df["area"] = df["area"].astype(float)

# df["unformattedPrice"] = df["unformattedPrice"]/df["area"]
df.columns = ["zpid", "price", "latitude", "longitude", "beds", "baths", "area"]
print(df.describe())
ax = sns.boxplot(x=df["price"])
# plt.show()
print(df.dtypes)

# for col in df.columns[1:]:
#    df = remove_outliers(df, col)

df = remove_outliers(df, "price")
df = remove_outliers(df, "beds")
df = remove_outliers(df, "baths")
df = remove_outliers(df, "area")
# dataset has to be divisble by 0.8!

df = df.iloc[3:]
print(df.describe())
ax = sns.boxplot(x=df["price"])
# plt.show()
print(df)
df.to_pickle(f"{data_path}df.pkl")
# print(list(df["zpid"] + ".png"))
# resize_images(list(df["zpid"] + ".png"))
