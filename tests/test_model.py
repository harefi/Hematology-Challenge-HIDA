import torch
from hematol.models import build_baseline_model


def test_forward_pass():
    model = build_baseline_model(num_classes=11)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 11)
