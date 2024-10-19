import os

import numpy as np
import pandas as pd

from videoreader import VideoReader


test = pd.read_csv("data/splits/test.csv")
train = pd.read_csv("data/splits/train.csv")
val = pd.read_csv("data/splits/val.csv")
print(f"total number of videos: {len(test) + len(train) + len(val)}")

test_videos = np.array(test["Video file"])
train_videos = np.array(train["Video file"])
val_videos = np.array(val["Video file"])

v = VideoReader()
for file in os.scandir("data/videos"):
    # check to see whether we've already read data for this video
    # and written it to disk
    if (os.path.isfile(f"data/csv/test/{file.name}") or
        os.path.isfile(f"data/csv/train/{file.name}") or
        os.path.isfile(f"data/csv/val/{file.name}")):
        continue

    # otherwise, read data from the video
    data: np.ndarray = v.read_video(f"data/videos/{file.name}", True)

    # determine whether the video should be in test, train or split
    if file.name in test_videos:
        split = "test"
    elif file.name in train_videos:
        split = "train"
    else:
        split = "val"
    
    v.write_data(data, file.name, f"data/csv/{split}")