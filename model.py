import torch
from torch.nn.utils import clip_grad_norm_


class Model:
    def __init__(self, model, optimizer, dataloader, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.loss_fn = loss_fn

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_model(self, num_epochs) -> tuple[dict, dict]:
        losses, accuracies = {"train": [], "val": []}, {"train": [], "val": []}
        for epoch in range(num_epochs):
            train_loss, train_accuracy = self.__train_epoch()
            val_loss, val_accuracy = self.__eval()

            losses["train"].append(train_loss), losses["val"].append(val_loss)
            accuracies["train"].append(train_accuracy), accuracies["val"].append(val_accuracy)

            print(f"train loss: {train_loss}, train accuracy: {train_accuracy}")
            print(f"val loss: {val_loss}, val accuracy: {val_accuracy}")

        return losses, accuracies

    def __train_epoch(self) -> tuple[float, float]:
        self.model.train()
        losses, accuracies = [], []
        for batch in self.dataloader:
            data, labels = batch
            data, labels = data.to(self.device), labels.to(self.device)

            # Model forward propagation
            model_output = self.model.forward(input=data)

            # Update model parameters
            self.optimizer.zero_grad()
            loss = self.loss_fn()  asdfasdf# remember to fix
            loss.backward()
            clip_grad_norm_(parameters=self.model.parameters(), max_norm=1.0, norm_type=2)
            self.optimizer.step()

            # Compute performance measures of current model.
            accuracy = (model_output.sigmoid().round() == labels).to(dtype=torch.float32).mean()
            accuracies.append(accuracy.detach().item())
            losses.append(loss.detach().item())

        # Report progress of training procedure.
        return sum(losses) / len(losses), sum(accuracies) / len(accuracies)

    def __eval(self) -> tuple[float, float]:
        self.model.eval()
        with torch.no_grad():
            losses, accuracies = [], []
            for batch in self.dataloader:
                data, labels = batch
                data, labels = data.to(self.device), labels.to(self.device)

                # Process data by Hopfield-based network.
                model_output = self.model.forward(input=data)
                loss = binary_cross_entropy_with_logits(input=model_output, target=labels, reduction=r'mean')

                # Compute performance measures of current model.
                accuracy = (model_output.sigmoid().round() == labels).to(dtype=torch.float32).mean()
                accuracies.append(accuracy.detach().item())
                losses.append(loss.detach().item())

            # Report results on validation set
            return sum(losses) / len(losses), sum(accuracies) / len(accuracies)
