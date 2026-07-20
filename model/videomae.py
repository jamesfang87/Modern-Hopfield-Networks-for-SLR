import torch

from transformers import (
    VideoMAEConfig,
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
    VideoMAEModel,
    AutoConfig,
    AutoModel,
)


def load_videomae_v2(checkpoint: str = "OpenGVLab/VideoMAEv2-Base"):
    processor = VideoMAEImageProcessor.from_pretrained(checkpoint)
    config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    encoder = AutoModel.from_pretrained(
        checkpoint, config=config, trust_remote_code=True
    )
    return processor, encoder


class FineTuningClassifier(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, hidden_size: int, num_signs: int):
        super().__init__()
        self.encoder = encoder
        self.head = torch.nn.Linear(hidden_size, num_signs)

    def forward(self, pixel_values):
        # pixel_values: (B, T, C, H, W) -> processor handles this shape
        outputs = self.encoder(pixel_values)
        # mean-pool over the token/sequence dimension
        pooled = outputs.last_hidden_state.mean(dim=1)
        return self.head(pooled)


def load_model():
    _, encoder_v2 = load_videomae_v2("OpenGVLab/VideoMAEv2-Base")
    return FineTuningClassifier(
        encoder_v2, hidden_size=encoder_v2.config.hidden_size, num_signs=2731
    )
