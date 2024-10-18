import pandas as pd
from videoreader import VideoReader
import os
import os.path


test = pd.read_csv("data/splits/test.csv")
train = pd.read_csv("data/splits/train.csv")
val = pd.read_csv("data/splits/val.csv")
print(f"total number of videos: {len(test) + len(train) + len(val)}")

v = VideoReader()
for file in os.scandir("data/videos"):
    v.read_video(f"data/videos/{file.name}", True)