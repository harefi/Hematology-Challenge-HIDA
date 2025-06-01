from pathlib import Path
import pytest
from hematol.data import build_datasets

DATA_ROOT = Path("Source")
# paths that signal the data are present
ACE_FOLDER = DATA_ROOT / "Acevedo_20"
MATEK_FOLDER = DATA_ROOT / "Matek_19"
ACE_ZIP = DATA_ROOT / "Acevedo_20.zip"
MATEK_ZIP = DATA_ROOT / "Matek_19.zip"

HAS_DATA = any(p.exists() for p in (ACE_FOLDER, MATEK_FOLDER, ACE_ZIP, MATEK_ZIP))


@pytest.mark.skipif(
    not HAS_DATA,
    reason="Dataset archives/folders not available on CI runner",
)
def test_dataset_shapes():
    ds, _, _ = build_datasets(DATA_ROOT)
    img, label = ds[0]
    assert img.shape == (3, 224, 224)
    assert img.dtype.kind == "f"
    assert isinstance(label, int)
