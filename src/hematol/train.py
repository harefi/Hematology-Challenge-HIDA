#!/usr/bin/env python
"""train.py — Baseline ResNet‑18 training script for the Helmholtz HIDA Hematology Data‑Challenge.

Run from the project root:

    python scripts/train.py --data_root ./Source --epochs 30

This script will:
1. (Optionally) extract the zipped datasets if they have not been unpacked yet.
2. Build a combined PyTorch `Dataset` from Acevedo_20 and Matek_19.
3. Create stratified train/val splits.
4. Train a ResNet‑18 (pre‑trained on ImageNet) using GPU(s) if available.
5. Log accuracy, loss, F1‑macro per epoch to CSV.
6. Save best model weights (highest F1‑macro) to <output_dir>/best_model.pth
"""

import argparse
from pathlib import Path
import random
import csv

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from .data import build_datasets, unpack_if_needed
from .transforms import train_tfms, val_tfms
from .models import build_baseline_model


def worker_init_fn(worker_id):
    """Ensure each worker has a different random seed."""
    np.random.seed(torch.initial_seed() % 2**32 + worker_id)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, running_correct, n_samples = 0.0, 0, 0
    all_preds, all_targets = [], []
    for inputs, targets in tqdm(loader, leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_correct += torch.sum(preds == targets).item()
        n_samples += inputs.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / n_samples
    epoch_acc = running_correct / n_samples
    epoch_f1 = f1_score(all_targets, all_preds, average="macro")
    return epoch_loss, epoch_acc, epoch_f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, running_correct, n_samples = 0.0, 0, 0
    all_preds, all_targets = [], []
    for inputs, targets in tqdm(loader, leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_correct += torch.sum(preds == targets).item()
        n_samples += inputs.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / n_samples
    epoch_acc = running_correct / n_samples
    epoch_f1 = f1_score(all_targets, all_preds, average="macro")
    return epoch_loss, epoch_acc, epoch_f1


def main():
    parser = argparse.ArgumentParser(description="Baseline training script")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("./Source"),
        help="Folder containing the extracted datasets OR the original zip files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./results"),
        help="Where to save checkpoints & logs",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        type=Path,
        help="Path to a checkpoint (.pth) to resume training from",
    )

    args = parser.parse_args()

    # Tensorboard Hook
    from torch.utils.tensorboard import SummaryWriter

    tb_writer = SummaryWriter(log_dir=args.output_dir / "tb")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Unpack archives if needed
    archives = {
        "Acevedo_20.zip": "Acevedo_20",
        "Matek_19.zip": "Matek_19",
    }
    for zip_name, dir_name in archives.items():
        zip_path = args.data_root / zip_name
        dest_dir = args.data_root / dir_name
        if zip_path.exists():
            unpack_if_needed(zip_path, dest_dir)

    # 2. Build datasets
    full_dataset, class_to_idx, targets = build_datasets(args.data_root)
    num_classes = len(class_to_idx)
    print(f"* Dataset ready — {len(full_dataset)} images across {num_classes} classes.")

    indices = np.arange(len(full_dataset))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, stratify=targets, random_state=args.seed
    )

    # Wrap transforms
    for ds in full_dataset.datasets:
        ds.transform = train_tfms  # default to train; will be overwritten for val

    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)

    # enforce transform difference
    def set_transform(subset, transform):
        subset.dataset.transform = transform
        return subset

    train_subset = set_transform(train_subset, train_tfms)
    val_subset = set_transform(val_subset, val_tfms)

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 3. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Using device: {device}")

    # 4. Model
    model = build_baseline_model(num_classes)
    if torch.cuda.device_count() > 1:
        print(f"--> DataParallel on {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_epoch = 1
    best_f1 = 0.0

    if args.resume and not args.resume.exists():
        raise FileNotFoundError(f"--resume file not found: {args.resume}")

    if args.resume and args.resume.exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_f1 = ckpt.get("val_f1", 0.0)
        print(
            f"--> Resumed from {args.resume}  (epoch {ckpt['epoch']}, val-F1 {best_f1:.4f})"
        )

    log_path = args.output_dir / "training_log.csv"
    # ---------------------------------------------------------------- log header
    if not log_path.exists():
        with log_path.open("w", newline="") as f:
            csv.writer(f).writerow(["epoch", "phase", "loss", "accuracy", "f1_macro"])
            # from here on we *always* append inside the epoch loop

    # for epoch in range(1, args.epochs + 1):
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)

        print(
            f"  train — loss {train_loss:.4f}  acc {train_acc:.4f}  f1 {train_f1:.4f}"
        )
        print(f"  val   — loss {val_loss:.4f}  acc {val_acc:.4f}  f1 {val_f1:.4f}")
        # ---- append to CSV log ----
        with log_path.open("a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch, "train", train_loss, train_acc, train_f1])
            writer.writerow([epoch, "val", val_loss, val_acc, val_f1])
            csvfile.flush()  # make sure it's written immediately

        # Tensorboard Hook
        tb_writer.add_scalar("Loss/train", train_loss, epoch)
        tb_writer.add_scalar("Loss/val", val_loss, epoch)
        tb_writer.add_scalar("Acc/train", train_acc, epoch)
        tb_writer.add_scalar("Acc/val", val_acc, epoch)
        tb_writer.add_scalar("F1/train", train_f1, epoch)
        tb_writer.add_scalar("F1/val", val_f1, epoch)

        # ---- keep best model according to val F1 ----
        if val_f1 > best_f1:
            best_f1 = val_f1
            ckpt_path = args.output_dir / "best_model.pth"

            torch.save(
                {
                    "epoch": epoch,
                    "val_f1": val_f1,
                    "model_state_dict": (
                        model.module.state_dict()
                        if isinstance(model, nn.DataParallel)
                        else model.state_dict()
                    ),
                    "class_to_idx": class_to_idx,
                },
                ckpt_path,
            )
            print(f"* New best model saved to {ckpt_path}  (F1 {val_f1:.4f})")

        # ---- always save the most recent weights ----
        # unconditional save – put it right *after* the best-model block
        torch.save(
            {
                "epoch": epoch,
                "val_f1": val_f1,
                "model_state_dict": (
                    model.module.state_dict()
                    if isinstance(model, nn.DataParallel)
                    else model.state_dict()
                ),
                "class_to_idx": class_to_idx,
            },
            args.output_dir / "last_model.pth",
        )

    tb_writer.close()


if __name__ == "__main__":
    main()
