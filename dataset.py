import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from transformations.normalizations import upsample_sign, downsample_sign


class ASLCitizen(Dataset):
    def __init__(self, annotations_file: str, data_dir: str, sign_length = 128, transforms=None):
        """
        :param annotations_file: the path of the .csv file containing file paths and labels
        :param data_dir: the path of the directory holding the .npy files that represent the videos
        :param transforms: transforms on the asl_citizen, defaulted to None
        """
        self.annotations = pd.read_csv(annotations_file)
        self.data_dir = data_dir
        self.transforms = transforms

        self.sign_length = sign_length

        self.file_names = self.annotations["Video file"].to_list()
        self.labels = self.annotations["Label"].to_list()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        data_path = os.path.join(self.data_dir, self.file_names[idx].replace(".mp4", ".npy"))
        sign = np.load(data_path)
        label = self.labels[idx]

        # only look at x and y coordinates of body landmarks for now
        sign = sign[:, :, :2]

        # shift coordinate system so that [0.5, 0.5] is now the origin
        sign -= 0.5

        # normalize the length of all signs to be self.sign_length
        if len(sign) > self.sign_length:
            sign = downsample_sign(sign, self.sign_length)
        elif len(sign) < self.sign_length:
            sign = upsample_sign(sign, self.sign_length)

        # "flatten" out all other dimensions except for the temporal dimension
        sign = sign.reshape(sign.shape[0], -1)

        if self.transforms:
            for transform in self.transforms:
                sign = transform(sign)

        return sign, label