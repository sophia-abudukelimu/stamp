"""
Two probes that measure what the CLS wiring actually costs and buys.

They exist because `CLS_MODE = "bidirectional"` is a deliberate relaxation of
strict causality, and a relaxation that is not measured is just a bug with a
comment next to it.

    measure_causal_leak                how much of the future reaches the past
    measure_cls_reconstruction_gradient  whether CLS learns from the MSE at all

Under CLS_MODE = "sink" both return exactly zero for their respective
quantities, and `models.transformer.verify_model` asserts that.
"""

import torch
import torch.nn.functional as F

import config


def measure_causal_leak(model, device=None):
    """
    Shift every frame from CAUSAL_LEAK_FIRST_FUTURE_PATCH onwards by a constant
    and see how far the reconstruction of the EARLIER patches moves.

    A strictly causal model cannot move at all, so past_shift is exactly 0.0.
    Under "bidirectional" it is not, and

        leak_ratio = past_shift / future_shift

    is the number to quote when someone asks how much the future leaks.

    The probe windows are drawn from a fixed generator, so the number is
    comparable across checkpoints.
    """
    device = config.get_device() if device is None else device

    was_training = model.training
    model.eval()

    split = config.CAUSAL_LEAK_FIRST_FUTURE_PATCH * config.N_REGIONS
    frame_split = config.CAUSAL_LEAK_FIRST_FUTURE_PATCH * config.PATCH_LENGTH

    generator = torch.Generator(device=device).manual_seed(config.SEED + 300_000)

    with torch.no_grad():
        probe = torch.randn(
            config.CAUSAL_LEAK_PROBE_WINDOWS,
            config.WINDOW_SIZE,
            config.N_REGIONS,
            device=device,
            generator=generator,
        )
        perturbed = probe.clone()
        perturbed[:, frame_split:, :] += config.CAUSAL_LEAK_PERTURBATION

        before, _, _, _ = model(probe, None)
        after, _, _, _ = model(perturbed, None)

        past_shift = (after[:, :split] - before[:, :split]).abs().max().item()
        future_shift = (after[:, split:] - before[:, split:]).abs().max().item()

    if was_training:
        model.train()

    return {
        "past_shift": past_shift,
        "future_shift": future_shift,
        "leak_ratio": past_shift / max(future_shift, 1e-12),
    }


def measure_cls_reconstruction_gradient(device=None):
    """
    Total |gradient| reaching cls_token from the reconstruction MSE alone.

    Under "sink" this is exactly 0.0: the decoder reads token positions only,
    and under "sink" those never depend on CLS, so the CLS embedding would be
    shaped by the VICReg term alone (weight 0.05). Under "bidirectional" it is
    positive, and that is the entire reason for the switch.
    """
    from models.transformer import STAMPEncoder, patchify

    device = config.get_device() if device is None else device

    probe_model = STAMPEncoder().to(device)
    probe_model.eval()

    generator = torch.Generator(device=device).manual_seed(config.SEED + 400_000)
    probe = torch.randn(
        8, config.WINDOW_SIZE, config.N_REGIONS,
        device=device, generator=generator,
    )
    mask = torch.zeros(8, config.N_TOKENS, dtype=torch.bool, device=device)
    mask[:, ::3] = True

    reconstruction, _, _, _ = probe_model(probe, mask)
    F.mse_loss(reconstruction[mask], patchify(probe)[mask]).backward()

    gradient = probe_model.cls_token.grad
    magnitude = 0.0 if gradient is None else float(gradient.abs().sum().item())

    del probe_model
    return magnitude
