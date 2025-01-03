import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm


class ModelTrainer:
    def __init__(self, model, optimizer, loss_fn, train_dataloader, val_dataloader):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=0.01, steps_per_epoch=len(self.train_dataloader), epochs=50)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(device=self.device)

    # code based on https://github.com/ml-jku/hopfield-layers/tree/master
    def train_model(self, num_epochs) -> tuple[dict, dict]:
        losses, accuracies = {"train": [], "val": []}, {"train": [], "val": []}
        for epoch in range(num_epochs):
            train_loss, train_accuracy = self.__train_epoch()
            val_loss, val_accuracy = self.__eval()

            losses["train"].append(train_loss), losses["val"].append(val_loss)
            accuracies["train"].append(train_accuracy), accuracies["val"].append(val_accuracy)

            print(f"epoch {epoch + 1}:")
            print(f"\ttrain loss: {train_loss}, train accuracy: {train_accuracy}")
            print(f"\tval loss: {val_loss}, val accuracy: {val_accuracy}")

        return losses, accuracies

    # code based on https://github.com/ml-jku/hopfield-layers/tree/master
    def __train_epoch(self) -> tuple[float, float]:
        self.model.train()
        losses, accuracies = [], []
        for batch in tqdm(self.train_dataloader):
            data, labels = batch
            data, labels = data.to(self.device), labels.to(self.device)

            # Model forward propagation
            model_output = self.model.forward(input=data.to(dtype=torch.float64))

            # Update model parameters
            self.optimizer.zero_grad()
            loss = self.loss_fn(model_output, labels.to(dtype=torch.int64))
            loss.backward()
            clip_grad_norm_(parameters=self.model.parameters(), max_norm=1.0, norm_type=2)
            self.optimizer.step()
            self.scheduler.step()

            # Compute performance measures of current model.
            accuracy = (torch.argmax(model_output.sigmoid().round(), dim=1) == labels).to(dtype=torch.float32).mean()
            accuracies.append(accuracy.detach().item())
            losses.append(loss.detach().item())

        # Report progress of training procedure
        return sum(losses) / len(losses), sum(accuracies) / len(accuracies)

    # code based on https://github.com/ml-jku/hopfield-layers/tree/master
    def __eval(self) -> tuple[float, float]:
        self.model.eval()
        with torch.no_grad():
            losses, accuracies = [], []
            for batch in tqdm(self.val_dataloader):
                data, labels = batch
                data, labels = data.to(self.device), labels.to(self.device)

                # Model forward propagation
                model_output = self.model.forward(input=data.to(dtype=torch.float64))
                loss = self.loss_fn(model_output, labels.to(dtype=torch.int64))

                # Compute performance measures of current model
                accuracy = (torch.argmax(model_output.sigmoid().round(), dim=1) == labels).to(dtype=torch.float32).mean()
                accuracies.append(accuracy.detach().item())
                losses.append(loss.detach().item())

            # Report results on validation set
            return sum(losses) / len(losses), sum(accuracies) / len(accuracies)