import torch.nn
from torch.nn import Flatten, Linear, Sequential
from torch.optim import AdamW
from torch.utils.data import DataLoader

from model_trainer import ModelTrainer
from dataset import ASLCitizen

print(torch.__version__)

train = DataLoader(ASLCitizen("asl_citizen/splits/train.csv", "asl_citizen/npy/train"), batch_size=64)
val = DataLoader(ASLCitizen("asl_citizen/splits/val.csv", "asl_citizen/npy/val"), batch_size=64)


T = 128  # temporal dimension, number of frames
different_signs = 2731

# create model
network = Sequential()
optimizer = AdamW(params=network.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

translator = ModelTrainer(network, optimizer, loss_fn, train, val)
results = translator.train_model(50)
