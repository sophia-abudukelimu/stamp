"""
One pass over a loader, used for both training and evaluation.

Two independent mask draws are made per batch. Both are reconstructed, the MSE
is averaged over the two, and the two projected CLS embeddings form the VICReg
pair. Using the same window under two different masks is what makes the CLS
representation invariant to which cells happened to be hidden.

    total_loss = 0.5 * [ MSE(view 1) + MSE(view 2) ] + 0.05 * VICReg
"""

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

import config
from evaluation.metrics import (
    accumulate_region_sums, create_region_sums, finalize_region_metrics,
)
from models.transformer import create_spatiotemporal_mask, patchify


def run_pass(
    model,
    loader,
    training,
    mask_seed,
    n_masked_tokens,
    region_labels,
    device,
    optimizer=None,
    gradient_scaler=None,
):
    """Returns (metrics dict, per-region DataFrame)."""

    if training:
        if optimizer is None or gradient_scaler is None:
            raise ValueError("Training requires an optimizer and a scaler.")
        model.train()
    else:
        model.eval()

    region_sums = create_region_sums(device)
    token_region_index = model.token_region_index.to(device)

    loss_sums = defaultdict(float)
    total_windows = 0

    # The mask sequence is driven by a fixed seed, so validation uses the same
    # masks at every epoch and the curve is not contaminated by mask noise.
    mask_generator = torch.Generator(device=device).manual_seed(mask_seed)
    gradient_context = torch.enable_grad() if training else torch.no_grad()

    with gradient_context:

        for windows, _ in loader:

            windows = windows.to(device, dtype=torch.float32, non_blocking=True)
            batch_size = windows.shape[0]

            token_mask_1 = create_spatiotemporal_mask(
                batch_size, mask_generator, n_masked_tokens, device
            )
            token_mask_2 = create_spatiotemporal_mask(
                batch_size, mask_generator, n_masked_tokens, device
            )

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type, enabled=device.type == "cuda"
            ):
                reconstruction_1, _, projected_1, _ = model(windows, token_mask_1)
                reconstruction_2, _, projected_2, _ = model(windows, token_mask_2)

                target = patchify(windows)

                reconstruction_mse = 0.5 * (
                    F.mse_loss(
                        reconstruction_1[token_mask_1], target[token_mask_1]
                    )
                    + F.mse_loss(
                        reconstruction_2[token_mask_2], target[token_mask_2]
                    )
                )

                from models.transformer import compute_vicreg_loss
                vicreg = compute_vicreg_loss(
                    projected_1.float(), projected_2.float()
                )

                total_loss = (
                    reconstruction_mse + config.VICREG_WEIGHT * vicreg["total"]
                )

            if training:
                gradient_scaler.scale(total_loss).backward()
                gradient_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.GRADIENT_CLIP_NORM
                )
                gradient_scaler.step(optimizer)
                gradient_scaler.update()

            with torch.no_grad():
                for reconstruction, token_mask in (
                    (reconstruction_1, token_mask_1),
                    (reconstruction_2, token_mask_2),
                ):
                    accumulate_region_sums(
                        region_sums,
                        reconstruction.detach().float(),
                        target.float(),
                        token_mask,
                        token_region_index,
                    )

            loss_sums["total_loss"] += float(total_loss.item()) * batch_size
            for name in ["total", "invariance", "variance", "covariance"]:
                loss_sums[f"vicreg_{name}"] += (
                    float(vicreg[name].item()) * batch_size
                )
            total_windows += batch_size

    if total_windows == 0:
        raise RuntimeError("Loader produced no windows.")

    global_metrics, region_metrics_df = finalize_region_metrics(
        region_sums, region_labels
    )

    if global_metrics["value_count"] <= 0:
        raise RuntimeError("No masked values were evaluated.")

    metrics = {
        "total_loss": loss_sums["total_loss"] / total_windows,
        "masked_mse": global_metrics["masked_mse"],
        "masked_mae": global_metrics["masked_mae"],
        "masked_r2": global_metrics["masked_r2"],
        "masked_correlation": global_metrics["masked_correlation"],
        "zero_baseline_mse": global_metrics["zero_baseline_mse"],
        "vicreg_total": loss_sums["vicreg_total"] / total_windows,
        "vicreg_invariance": loss_sums["vicreg_invariance"] / total_windows,
        "vicreg_variance": loss_sums["vicreg_variance"] / total_windows,
        "vicreg_covariance": loss_sums["vicreg_covariance"] / total_windows,
    }
    return metrics, region_metrics_df
