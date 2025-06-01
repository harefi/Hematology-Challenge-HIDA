from pathlib import Path
import itertools
import zipfile
import numpy as np

import torch
from torch.utils.data import ConcatDataset
from torchvision import datasets


def unpack_if_needed(zip_path: Path, dest_dir: Path):
    """Extract *zip_path* into *dest_dir* if *dest_dir* does not yet exist."""
    if dest_dir.exists():
        print(f"✓ {dest_dir} already exists — skipping extraction.")
        return
    print(f"→ Extracting {zip_path.name} …")
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        #        zf.extractall(dest_dir.parent)
        zf.extractall(dest_dir)  # extract *inside* the dataset folder

    print(f"✓ Extracted to {dest_dir}")


def build_datasets(root: Path):
    """
    Return
        merged_dataset : torch.utils.data.ConcatDataset
        class_to_idx   : global mapping  {class_name: new_idx}
        targets        : np.ndarray of remapped labels (same length as merged_dataset)
    """
    # 1. Decide which directories are “datasets”
    dataset_dirs = [root / d for d in ("Acevedo_20", "Matek_19") if (root / d).is_dir()]
    if not dataset_dirs:
        dataset_dirs = [p for p in root.glob("*") if p.is_dir()]
    if not dataset_dirs:
        raise RuntimeError(f"No dataset folders found in {root}")

    # 2. First pass – load each ImageFolder, record its classes
    raw_datasets, global_classes = [], set()
    for d in dataset_dirs:
        ds = datasets.ImageFolder(d)  # uses folder names as labels
        raw_datasets.append(ds)
        global_classes.update(ds.classes)  # union of all label names

    global_classes = sorted(global_classes)
    class_to_idx = {cls: i for i, cls in enumerate(global_classes)}

    # 3. Wrap every ImageFolder so its labels map into the global index space
    class RemapDataset(torch.utils.data.Dataset):
        """
        Wrap an ImageFolder so its labels map into the global index space
        *and* expose a `.transform` attribute like the original dataset.
        """

        def __init__(self, base):
            self.base = base
            self.targets = [class_to_idx[base.classes[t]] for t in base.targets]
            self.transform = None

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            path, _ = self.base.samples[idx]
            img = self.base.loader(path)
            if self.transform:
                img = self.transform(img)
            return img, self.targets[idx]

    # ------------------------------------------------ merge & return
    wrapped = [RemapDataset(ds) for ds in raw_datasets]
    merged = ConcatDataset(wrapped)
    targets = np.array(list(itertools.chain.from_iterable(w.targets for w in wrapped)))

    return merged, class_to_idx, targets
