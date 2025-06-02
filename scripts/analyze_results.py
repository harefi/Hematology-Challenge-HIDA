# scripts/analyze_results.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from pathlib import Path
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Subset
import torch
import torch.nn as nn

# ─── Absolute imports from src/hematol ────────────────────────────────────
# Requires: run with PYTHONPATH=src
from src.hematol.data import build_datasets
from src.hematol.transforms import val_tfms
from src.hematol.models import build_baseline_model


def main():
    results_root = Path("results")
    if not results_root.exists():
        print("❌ results/ directory not found. Run the experiments first.")
        return

    # 1. Find all experiment subfolders under results/
    exp_dirs = sorted([d for d in results_root.iterdir() if d.is_dir()])
    if not exp_dirs:
        print("❌ No experiment folders found under results/.")
        return

    # 2. Collect best validation F1 for each experiment
    summary = []
    for d in exp_dirs:
        exp_name = d.name.split("_")[-1]
        csv_path = d / "training_log.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            val_rows = df[df["phase"] == "val"]
            if not val_rows.empty:
                max_f1 = val_rows["f1_macro"].max()
                summary.append({"experiment": exp_name, "best_val_f1": max_f1})
            else:
                summary.append({"experiment": exp_name, "best_val_f1": np.nan})
        else:
            summary.append({"experiment": exp_name, "best_val_f1": np.nan})

    summary_df = pd.DataFrame(summary)

    # 3. Save summary CSV
    summary_csv = results_root / "results_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"✔ Wrote summary CSV → {summary_csv}")

    # 4. Bar chart of best F1 scores
    plt.figure(figsize=(8, 5))
    plt.bar(summary_df["experiment"], summary_df["best_val_f1"], color="skyblue")
    plt.title("Best Validation Macro-F1 by Experiment")
    plt.ylabel("Macro-F1")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    f1_plot = results_root / "summary_f1.png"
    plt.savefig(f1_plot)
    plt.close()
    print(f"✔ Saved F1 bar chart → {f1_plot}")

    # 5. Confusion matrix for baseline experiment (if present)
    baseline_dir = next((d for d in exp_dirs if "baseline" in d.name), None)
    if baseline_dir:
        ckpt_path = baseline_dir / "best_model.pth"
        if ckpt_path.exists():
            print("ℹ️  Generating confusion matrix for baseline validation set…")
            # Build full dataset and stratified split
            full_dataset, class_to_idx, targets = build_datasets(Path("Source"))
            num_classes = len(class_to_idx)
            indices = np.arange(len(full_dataset))
            from sklearn.model_selection import train_test_split

            train_idx, val_idx = train_test_split(
                indices, test_size=0.2, stratify=targets, random_state=42
            )
            val_subset = Subset(full_dataset, val_idx)

            # Apply validation transforms
            for ds in full_dataset.datasets:
                ds.transform = val_tfms

            val_loader = DataLoader(
                val_subset,
                batch_size=128,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )

            # Load model
            model = build_baseline_model(num_classes)
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            all_preds, all_labels = [], []
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs = imgs.to(device)
                    outputs = model(imgs)
                    preds = outputs.argmax(dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels.numpy())

            cm = confusion_matrix(
                all_labels, all_preds, labels=list(range(num_classes))
            )
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            plt.title("Baseline Validation Confusion Matrix")
            plt.colorbar()
            tick_marks = np.arange(num_classes)
            reverse_map = {v: k for k, v in class_to_idx.items()}
            class_names = [reverse_map[i] for i in range(num_classes)]
            plt.xticks(tick_marks, class_names, rotation=90)
            plt.yticks(tick_marks, class_names)
            thresh = cm.max() / 2.0
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                )
            plt.ylabel("True label")
            plt.xlabel("Predicted label")
            plt.tight_layout()
            cm_plot = results_root / "conf_matrix_baseline.png"
            plt.savefig(cm_plot)
            plt.close()
            print(f"✔ Saved confusion matrix → {cm_plot}")
        else:
            print("⚠️  Baseline checkpoint not found; skipping confusion matrix.")
    else:
        print("⚠️  No baseline folder found; skipping confusion matrix.")


if __name__ == "__main__":
    main()
