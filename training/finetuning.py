"""
Fine-tuning script for isolated sign language recognition (ASL Citizen)
using a pretrained VideoMAE encoder + classification head.
"""

import json
import os

import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    VideoMAEConfig,
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
)

from dataset import ASL_Citizen
from trainer import ModelTrainer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRAIN_H5 = "/kaggle/working/h5/train.h5"
VAL_H5 = "/kaggle/working/h5/val.h5"
LABEL_MAP_PATH = "/kaggle/working/cache/label_map.json"
CHECKPOINT_DIR = "/kaggle/working/checkpoints"

CHECKPOINT_NAME = "MCG-NJU/videomae-base-finetuned-ssv2"

NUM_FRAMES = 32
FRAME_STEP = 2
CROP_SIZE = 224
BATCH_SIZE = 4
NUM_WORKERS = 4

FREEZE_EPOCHS = 3  # phase 1: head-only warmup
FINETUNE_EPOCHS = 15  # phase 2: full fine-tune
HEAD_LR = 1e-3
ENCODER_LR = 5e-6
WEIGHT_DECAY = 0.05

USE_AMP = True  # mixed precision
USE_GRADIENT_CHECKPOINTING = True
GRAD_CLIP_NORM = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
def load_label_map() -> dict:
    with open(LABEL_MAP_PATH) as f:
        return json.load(f)


def list_video_ids(h5_path: str) -> list:
    with h5py.File(h5_path, "r") as f:
        return list(f.keys())


def build_model(num_classes: int) -> VideoMAEForVideoClassification:
    config = VideoMAEConfig.from_pretrained(CHECKPOINT_NAME)
    config.num_labels = num_classes
    model = VideoMAEForVideoClassification.from_pretrained(
        CHECKPOINT_NAME,
        config=config,
        ignore_mismatched_sizes=True,  # pretrained head was a different size
    )
    if USE_GRADIENT_CHECKPOINTING and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model


def set_encoder_trainable(model: nn.Module, trainable: bool):
    for name, param in model.named_parameters():
        if "classifier" not in name:  # everything except the head
            param.requires_grad = trainable


def build_optimizer(
    model: nn.Module, encoder_lr: float, head_lr: float, weight_decay: float
):
    """Returns the optimizer plus the max_lr list in the same order as its
    param_groups, so ModelTrainer's OneCycleLR can use per-group peak LRs."""
    encoder_params = [
        p
        for n, p in model.named_parameters()
        if "classifier" not in n and p.requires_grad
    ]
    head_params = [
        p for n, p in model.named_parameters() if "classifier" in n and p.requires_grad
    ]

    param_groups = [{"params": head_params, "lr": head_lr}]
    max_lr = [head_lr]
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": encoder_lr})
        max_lr.append(encoder_lr)

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    return optimizer, max_lr


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    label_map = load_label_map()
    num_classes = len(label_map)
    print(f"num_classes = {num_classes}")

    processor = VideoMAEImageProcessor.from_pretrained(CHECKPOINT_NAME)
    mean, std = processor.image_mean, processor.image_std

    train_ids = list_video_ids(TRAIN_H5)
    val_ids = list_video_ids(VAL_H5)

    train_dataset = ASL_Citizen(
        h5_path=TRAIN_H5,
        video_ids=train_ids,
        num_frames=NUM_FRAMES,
        frame_step=FRAME_STEP,
        crop_size=CROP_SIZE,
        is_train=True,
    )
    val_dataset = ASL_Citizen(
        h5_path=VAL_H5,
        video_ids=val_ids,
        num_frames=NUM_FRAMES,
        frame_step=FRAME_STEP,
        crop_size=CROP_SIZE,
        is_train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )

    model = build_model(num_classes).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")

    # Phase 1: freeze encoder, train head only.
    print("=== Phase 1: linear probe (head only) ===")
    set_encoder_trainable(model, trainable=False)
    optimizer, max_lr = build_optimizer(
        model, encoder_lr=0.0, head_lr=HEAD_LR, weight_decay=WEIGHT_DECAY
    )
    trainer = ModelTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        max_lr=max_lr,
        epochs=FREEZE_EPOCHS,
        mean=mean,
        std=std,
        checkpoint_path=checkpoint_path,
        use_amp=USE_AMP,
        grad_clip_norm=GRAD_CLIP_NORM,
    )
    trainer.train_model(FREEZE_EPOCHS)
    best_recall_at_1 = trainer.best_recall_at_1

    # Phase 2: unfreeze encoder, fine-tune everything with discriminative LRs.
    # Re-wrap in a fresh ModelTrainer since OneCycleLR is built per-phase
    # (different epoch count / step count than phase 1), and the optimizer
    # needs a new param group now that the encoder is trainable again.
    print("=== Phase 2: full fine-tune (discriminative LR) ===")
    set_encoder_trainable(model, trainable=True)
    optimizer, max_lr = build_optimizer(
        model, encoder_lr=ENCODER_LR, head_lr=HEAD_LR, weight_decay=WEIGHT_DECAY
    )
    trainer = ModelTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        max_lr=max_lr,
        epochs=FINETUNE_EPOCHS,
        mean=mean,
        std=std,
        checkpoint_path=checkpoint_path,
        use_amp=USE_AMP,
        grad_clip_norm=GRAD_CLIP_NORM,
    )
    # carry over phase 1's best score so phase 2 only overwrites the
    # checkpoint if it actually beats it
    trainer.best_recall_at_1 = best_recall_at_1
    trainer.train_model(FINETUNE_EPOCHS)

    print(
        f"training complete. best val recall@1 (top-1 accuracy): {trainer.best_recall_at_1:.4f}"
    )


if __name__ == "__main__":
    main()
