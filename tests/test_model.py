import torch

from dprf import DPRFNet


def test_forward_and_internal_ranges():
    model = DPRFNet()
    vibration = torch.randn(8, 41)
    electrical = torch.randn(8, 55)
    logits = model(vibration, electrical)
    fused, preference, reliability, calibrated, alpha = model.encode(
        vibration, electrical
    )
    assert logits.shape == (8, 6)
    assert fused.shape == (8, 64)
    assert preference.shape == reliability.shape == calibrated.shape == (8, 2)
    assert alpha.shape == (8, 1)
    assert torch.allclose(preference.sum(1), torch.ones(8), atol=1e-6)
    assert torch.allclose(calibrated.sum(1), torch.ones(8), atol=1e-6)
    assert torch.all((reliability >= 0) & (reliability <= 1))
    assert torch.all((alpha >= 0) & (alpha <= 1))
