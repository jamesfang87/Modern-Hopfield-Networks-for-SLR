# Import general modules used e.g. for plotting.
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import sys
import torch

# Importing Hopfield-specific modules.
from hflayers import Hopfield, HopfieldPooling, HopfieldLayer
from hflayers.auxiliary.data import LatchSequenceSet

# Import auxiliary modules.
from distutils.version import LooseVersion
from typing import List, Tuple

# Importing PyTorch specific modules.
from torch import Tensor
from torch.nn import Flatten, Linear, LSTM, Module, Sequential
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# Set plotting style.
sns.set()


python_check = '(\u2713)' if sys.version_info >= (3, 8) else '(\u2717)'
pytorch_check = '(\u2713)' if torch.__version__ >= LooseVersion(r'1.5') else '(\u2717)'

print(f'Installed Python version:  {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} {python_check}')
print(f'Installed PyTorch version: {torch.__version__} {pytorch_check}')


latch_sequence_set = LatchSequenceSet(
    num_samples=4096,
    num_instances=32,
    num_characters=20)

log_dir = f'resources/'
os.makedirs(log_dir, exist_ok=True)


device = torch.device(r'cuda:0' if torch.cuda.is_available() else r'cpu')
print(f"Using {device}")

# Create data loader of training set.
sampler_train = SubsetRandomSampler(list(range(512, 4096 - 512)))
data_loader_train = DataLoader(dataset=latch_sequence_set, batch_size=32, sampler=sampler_train)

# Create data loader of validation set.
sampler_eval = SubsetRandomSampler(list(range(512)) + list(range(4096 - 512, 4096)))
data_loader_eval = DataLoader(dataset=latch_sequence_set, batch_size=32, sampler=sampler_eval)

def train_epoch(network: Module,
                optimiser: AdamW,
                data_loader: DataLoader,
                loss_fn
               ) -> Tuple[float, float]:
    """
    Execute one training epoch.
    
    :param network: network instance to train
    :param optimiser: optimiser instance responsible for updating network parameters
    :param data_loader: data loader instance providing training data
    :return: tuple comprising training loss as well as accuracy
    """
    network.train()
    losses, accuracies = [], []
    for sample_data in data_loader:
        data, target = sample_data[r'data'], sample_data[r'target']
        data, target = data.to(device=device), target.to(device=device)

        # Process data by Hopfield-based network.
        model_output = network.forward(input=data)
        # Update network parameters.
        optimiser.zero_grad()
        loss = loss_fn(model_output, target.to(dtype=torch.int64))
        loss.backward()
        clip_grad_norm_(parameters=network.parameters(), max_norm=1.0, norm_type=2)
        optimiser.step()

        # Compute performance measures of current model.
        accuracy = (torch.argmax(model_output.sigmoid(), dim=1) == target).to(dtype=torch.float32).mean()
        accuracies.append(accuracy.detach().item())
        losses.append(loss.detach().item())
    
    # Report progress of training procedure.
    return (sum(losses) / len(losses), sum(accuracies) / len(accuracies))


def eval_iter(network: Module,
              data_loader: DataLoader,
              loss_fn
             ) -> Tuple[float, float]:
    """
    Evaluate the current model.
    
    :param network: network instance to evaluate
    :param data_loader: data loader instance providing validation data
    :return: tuple comprising validation loss as well as accuracy
    """
    network.eval()
    with torch.no_grad():
        losses, accuracies = [], []
        for sample_data in data_loader:
            data, target = sample_data[r'data'], sample_data[r'target']
            data, target = data.to(device=device), target.to(device=device)

            # Process data by Hopfield-based network.
            model_output = network.forward(input=data)
            loss = loss_fn(model_output, target.to(dtype=torch.int64))

            # Compute performance measures of current model.
            accuracy = (torch.argmax(model_output.sigmoid(), dim=1) == target).to(dtype=torch.float32).mean()
            accuracies.append(accuracy.detach().item())
            losses.append(loss.detach().item())

        # Report progress of validation procedure.
        return (sum(losses) / len(losses), sum(accuracies) / len(accuracies))


def operate(network: Module,
            optimiser: AdamW,
            data_loader_train: DataLoader,
            data_loader_eval: DataLoader,
            num_epochs: int = 1
           ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train the specified network by gradient descent using backpropagation.
    
    :param network: network instance to train
    :param optimiser: optimiser instance responsible for updating network parameters
    :param data_loader_train: data loader instance providing training data
    :param data_loader_eval: data loader instance providing validation data
    :param num_epochs: amount of epochs to train
    :return: data frame comprising training as well as evaluation performance
    """
    losses, accuracies = {r'train': [], r'eval': []}, {r'train': [], r'eval': []}
    for epoch in range(num_epochs):
        
        # Train network.
        performance = train_epoch(network, optimiser, data_loader_train, torch.nn.CrossEntropyLoss())
        losses[r'train'].append(performance[0])
        accuracies[r'train'].append(performance[1])
        
        # Evaluate current model.
        performance = eval_iter(network, data_loader_eval, torch.nn.CrossEntropyLoss())
        losses[r'eval'].append(performance[0])
        accuracies[r'eval'].append(performance[1])
    
    # Report progress of training and validation procedures.
    return pd.DataFrame(losses), pd.DataFrame(accuracies)

def set_seed(seed: int = 42) -> None:
    """
    Set seed for all underlying (pseudo) random number sources.
    
    :param seed: seed to be used
    :return: None
    """
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_performance(loss: pd.DataFrame,
                     accuracy: pd.DataFrame,
                     log_file: str
                    ) -> None:
    """
    Plot and save loss and accuracy.
    
    :param loss: loss to be plotted
    :param accuracy: accuracy to be plotted
    :param log_file: target file for storing the resulting plot
    :return: None
    """
    fig, ax = plt.subplots(1, 2, figsize=(20, 7))
    
    loss_plot = sns.lineplot(data=loss, ax=ax[0])
    loss_plot.set(xlabel=r'Epoch', ylabel=r'Cross-entropy Loss')
    
    accuracy_plot = sns.lineplot(data=accuracy, ax=ax[1])
    accuracy_plot.set(xlabel=r'Epoch', ylabel=r'Accuracy')
    
    ax[1].yaxis.set_label_position(r'right')
    fig.tight_layout()
    fig.savefig(log_file)
    #plt.show(fig)

set_seed()
hopfield = Hopfield(
    input_size=latch_sequence_set.num_characters)

output_projection = Linear(in_features=hopfield.output_size * latch_sequence_set.num_instances, out_features=10)
network = Sequential(hopfield, Flatten(), output_projection).to(device=device)
optimiser = AdamW(params=network.parameters(), lr=1e-3)

losses, accuracies = operate(
    network=network,
    optimiser=optimiser,
    data_loader_train=data_loader_train,
    data_loader_eval=data_loader_eval,
    num_epochs=40)

plot_performance(loss=losses, accuracy=accuracies, log_file=f'{log_dir}/hopfield_base.pdf')
