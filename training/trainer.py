import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm


class Stats:
    """Accumulates loss and ranking metrics (Recall@1, Recall@5,
    Recall@10, MRR, DCG) over an epoch and reports their averages."""

    def __init__(self):
        self.losses = []
        self.recall_at_1 = []  # Recall@1
        self.recall_at_5 = []
        self.recall_at_10 = []
        self.mrr = []
        self.dcg = []

    def update(self, loss: float, model_output: torch.Tensor, labels: torch.Tensor):
        # 1-indexed rank of the correct label within the sorted predictions
        ranking = torch.argsort(model_output, dim=1, descending=True)
        ranks = (ranking == labels.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1
        ranks = ranks.to(dtype=torch.float32)

        self.losses.append(loss)
        self.recall_at_1.append((ranks <= 1).to(dtype=torch.float32).mean().item())
        self.recall_at_5.append((ranks <= 5).to(dtype=torch.float32).mean().item())
        self.recall_at_10.append((ranks <= 10).to(dtype=torch.float32).mean().item())
        self.mrr.append((1.0 / ranks).mean().item())
        self.dcg.append((1.0 / torch.log2(ranks + 1)).mean().item())

    def compute(self) -> dict[str, float]:
        return {
            "loss": sum(self.losses) / len(self.losses),
            "recall@1": sum(self.recall_at_1) / len(self.recall_at_1),
            "recall@5": sum(self.recall_at_5) / len(self.recall_at_5),
            "recall@10": sum(self.recall_at_10) / len(self.recall_at_10),
            "mrr": sum(self.mrr) / len(self.mrr),
            "dcg": sum(self.dcg) / len(self.dcg),
        }


class ModelTrainer:
    def __init__(self, model, optimizer, loss_fn, train_dataloader, val_dataloader):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=0.01,
            steps_per_epoch=len(self.train_dataloader),
            epochs=50,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(device=self.device)

    # code based on https://github.com/ml-jku/hopfield-layers/tree/master
    def train_model(self, num_epochs) -> tuple[dict, dict]:
        train_history, val_history = {}, {}
        for epoch in range(num_epochs):
            train_stats = self.__train_epoch()
            val_stats = self.__eval()

            for key, value in train_stats.items():
                train_history.setdefault(key, []).append(value)
            for key, value in val_stats.items():
                val_history.setdefault(key, []).append(value)

            print(f"epoch {epoch + 1}:")
            print(f"\ttrain: {train_stats}")
            print(f"\tval: {val_stats}")

        return train_history, val_history

    # code based on https://github.com/ml-jku/hopfield-layers/tree/master
    def __train_epoch(self) -> dict[str, float]:
        self.model.train()
        stats = Stats()
        for batch in tqdm(self.train_dataloader):
            data, labels = batch
            data, labels = data.to(self.device), labels.to(self.device)

            # Model forward propagation
            model_output = self.model.forward(input=data.to(dtype=torch.float64))

            # Update model parameters
            self.optimizer.zero_grad()
            loss = self.loss_fn(model_output, labels.to(dtype=torch.int64))
            loss.backward()
            clip_grad_norm_(
                parameters=self.model.parameters(), max_norm=1.0, norm_type=2
            )
            self.optimizer.step()
            self.scheduler.step()

            # Compute performance measures of current model.
            stats.update(loss.detach().item(), model_output.detach(), labels)

        # Report progress of training procedure
        return stats.compute()

    # code based on https://github.com/ml-jku/hopfield-layers/tree/master
    def __eval(self) -> dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            stats = Stats()
            for batch in tqdm(self.val_dataloader):
                data, labels = batch
                data, labels = data.to(self.device), labels.to(self.device)

                # Model forward propagation
                model_output = self.model.forward(input=data.to(dtype=torch.float64))
                loss = self.loss_fn(model_output, labels.to(dtype=torch.int64))

                # Compute performance measures of current model
                stats.update(loss.detach().item(), model_output.detach(), labels)

            # Report results on validation set
            return stats.compute()
