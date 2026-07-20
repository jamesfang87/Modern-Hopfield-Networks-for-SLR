import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm


class Stats:
    """Accumulates loss and ranking metrics (Recall@1, Recall@5,
    Recall@10, MRR, DCG) over an epoch and reports their averages.
    """

    def __init__(self):
        self.losses = []
        self.recall_at_1 = []
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

    def compute(self) -> dict:
        return {
            "loss": sum(self.losses) / len(self.losses),
            "recall@1": sum(self.recall_at_1) / len(self.recall_at_1),
            "recall@5": sum(self.recall_at_5) / len(self.recall_at_5),
            "recall@10": sum(self.recall_at_10) / len(self.recall_at_10),
            "mrr": sum(self.mrr) / len(self.mrr),
            "dcg": sum(self.dcg) / len(self.dcg),
        }


class ModelTrainer:
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        train_dataloader,
        val_dataloader,
        max_lr,
        epochs,
        mean,
        std,
        checkpoint_path=None,
        use_amp=True,
        grad_clip_norm=1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.mean = mean
        self.std = std
        self.checkpoint_path = checkpoint_path
        self.grad_clip_norm = grad_clip_norm

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device=self.device)

        self.use_amp = use_amp and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=max_lr,  # float or list matching optimizer.param_groups order
            steps_per_epoch=len(self.train_dataloader),
            epochs=epochs,
        )

        self.best_recall_at_1 = 0.0

    def _prepare_batch(self, batch):
        pixel_values = batch["pixel_values"].to(self.device)
        labels = batch["label"].to(self.device)

        mean_t = torch.tensor(self.mean, device=self.device).view(1, 3, 1, 1, 1)
        std_t = torch.tensor(self.std, device=self.device).view(1, 3, 1, 1, 1)
        pixel_values = (
            pixel_values - mean_t
        ) / std_t  # normalize, still (B, C, T, H, W)
        pixel_values = pixel_values.permute(
            0, 2, 1, 3, 4
        ).contiguous()  # -> (B, T, C, H, W)

        return pixel_values, labels

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

            if (
                self.checkpoint_path is not None
                and val_stats["recall@1"] > self.best_recall_at_1
            ):
                self.best_recall_at_1 = val_stats["recall@1"]
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(
                    f"\tnew best val recall@1 ({self.best_recall_at_1:.4f}), saved to {self.checkpoint_path}"
                )

        return train_history, val_history

    def __train_epoch(self) -> dict:
        self.model.train()
        stats = Stats()
        for batch in tqdm(self.train_dataloader):
            pixel_values, labels = self._prepare_batch(batch)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                model_output = self.model(pixel_values=pixel_values).logits
                loss = self.loss_fn(model_output, labels.to(dtype=torch.int64))

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(
                parameters=self.model.parameters(),
                max_norm=self.grad_clip_norm,
                norm_type=2,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            stats.update(loss.detach().item(), model_output.detach(), labels)

        return stats.compute()

    def __eval(self) -> dict:
        self.model.eval()
        with torch.no_grad():
            stats = Stats()
            for batch in tqdm(self.val_dataloader):
                pixel_values, labels = self._prepare_batch(batch)

                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    model_output = self.model(pixel_values=pixel_values).logits
                    loss = self.loss_fn(model_output, labels.to(dtype=torch.int64))

                stats.update(loss.detach().item(), model_output.detach(), labels)

            return stats.compute()
