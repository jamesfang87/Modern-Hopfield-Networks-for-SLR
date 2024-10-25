import torch.nn
from torch.nn import Flatten, Linear, Sequential
from torch.optim import AdamW
from torch.utils.data import DataLoader

from hflayers import Hopfield, HopfieldPooling, HopfieldLayer

from model import Model
from dataset import ASLCitizen


train = DataLoader(ASLCitizen("data/splits/train", "data/npy/train"), batch_size=64)
val = DataLoader(ASLCitizen("data/splits/val", "data/npy/val"), batch_size=64)


hopfield = Hopfield(
    input_size= (42 + 17) * 2
)

T = 111  # temporal dimension, number of frames
different_signs = 2731

output_projection = Linear(in_features=hopfield.output_size * T, out_features=different_signs)


network = Sequential(hopfield, Flatten(), output_projection)
optimizer = AdamW(params=network.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss

translator = Model(network, optimizer, train, val, loss_fn)
results = translator.train_model(50)