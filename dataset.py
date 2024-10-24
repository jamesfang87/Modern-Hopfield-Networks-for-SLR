import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class ASLCitizen(Dataset):
    def __init__(self, annotations_file: str, data_dir: str, transforms=None):
        """
        :param annotations_file: the path of the .csv file containing file paths and labels
        :param data_dir: the path of the directory holding the .npy files that represent the videos
        :param transforms: transforms on the data, defaulted to None
        """
        self.annotations = pd.read_csv(annotations_file)
        self.data_dir = data_dir
        self.transforms = transforms

        self.file_names = self.annotations["Video file"].to_list()
        self.labels = self.annotations["Label"].to_list()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        data_path = os.path.join(self.data_dir, self.file_names[idx].replace(".mp4", ".npy"))
        data = np.load(data_path)
        label = self.labels[idx]

        if self.transforms:
            for transform in self.transforms:
                data = transform(data)

        return data, label