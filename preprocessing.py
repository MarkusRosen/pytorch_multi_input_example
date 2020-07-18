import pandas as pd
import os
from typing import List, Dict
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt

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


images = os.listdir(f"{data_path}images")
df = pd.read_pickle(f"{data_path}ny_dataframe.pkl")


df = select_rows_with_images(images, df)
df = df.iloc[0:800]
print(df.describe())
ax = sns.boxplot(x=df["baths"])
plt.show()

df.to_pickle(f"{data_path}df.pkl")
# resize_images(images)
