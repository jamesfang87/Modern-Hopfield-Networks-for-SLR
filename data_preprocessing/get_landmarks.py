import os

import numpy as np
import pandas as pd

from videoreader import VideoReader

test = pd.read_csv("asl_citizen/splits/test.csv")
train = pd.read_csv("asl_citizen/splits/train.csv")
val = pd.read_csv("asl_citizen/splits/val.csv")
print(f"total number of videos: {len(test) + len(train) + len(val)}")

test_videos = np.array(test["Video file"])
train_videos = np.array(train["Video file"])
val_videos = np.array(val["Video file"])

v = VideoReader()
for file in os.scandir("../asl_citizen/videos"):
    # check to see whether we've already read asl_citizen for this video
    # and written it to disk
    data_file_name = file.name.replace(".mp4", ".npy")
    if (os.path.isfile(f"asl_citizen/npy/test/{data_file_name}") or
            os.path.isfile(f"asl_citizen/npy/train/{data_file_name}") or
            os.path.isfile(f"asl_citizen/npy/val/{data_file_name}")):
        print(f"skipping video named {file.name}")
        continue

    # otherwise, read asl_citizen from the video
    print(f"processing video {file.name}")
    v.read_video(f"asl_citizen/videos/{file.name}", False)

    # determine whether the video should be in test, train or split
    if file.name in test_videos:
        split = "test"
    elif file.name in train_videos:
        split = "train"
    else:
        split = "val"

    v.write_data(data_file_name, f"asl_citizen/npy/{split}")
