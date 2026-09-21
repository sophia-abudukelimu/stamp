"""
Per-region reconstruction metrics, accumulated in streaming fashion.

Nothing is ever held for the whole epoch: each batch contributes eight running
sums per region, and every metric is derived from those sums at the end. This
keeps memory flat and makes the per-region and the global numbers come from
exactly the same code path (the global scalars are just the per-region arrays
summed).

Because the windows are z-scored per region, `zero_baseline_mse` should come
out at approximately 1.0. That is the single best sanity check in the whole
pipeline: if it is not near 1.0, the normalization or the masking is wrong and
every R2 below it is meaningless.
"""

import numpy as np
import pandas as pd
import torch

import config

REGION_SUM_KEYS = [
    "value_count",
    "squared_error",
    "absolute_error",
    "target_sum",
    "target_squared_sum",
    "prediction_sum",
    "prediction_squared_sum",
    "prediction_target_sum",
]


def create_region_sums(device):
    return {
        key: torch.zeros(config.N_REGIONS, dtype=torch.float64, device=device)
        for key in REGION_SUM_KEYS
    }


def accumulate_region_sums(
    region_sums, prediction, target, token_mask, token_region_index
):
    """
    Everything outside the mask is zeroed, so only masked cells contribute.
    Per-token sums are then scattered into their region bin.

    The elementwise work stays in float32 and only the reduced per-token sums
    are promoted to float64. Each per-token sum covers at most
    BATCH_SIZE * PATCH_LENGTH values of order 1, so the float32 reduction error
    is around 1e-6 relative, far below the precision the metrics are reported
    at, while the running totals across batches stay in float64 where it
    matters. Doing the elementwise work in float64 instead is roughly 2x slower
    on a T4 for no measurable gain.
    """
    # Expanded to the frame axis so that value_count counts FRAME VALUES, not
    # tokens. Getting this wrong inflates the MSE by a factor of PATCH_LENGTH.
    mask = token_mask.unsqueeze(-1).to(torch.float32)

    prediction = prediction.to(torch.float32) * mask
    target = target.to(torch.float32) * mask
    error = prediction - target

    per_token = {
        "value_count": mask.expand(-1, -1, config.PATCH_LENGTH).sum(-1).sum(0),
        "squared_error": error.pow(2).sum(-1).sum(0),
        "absolute_error": error.abs().sum(-1).sum(0),
        "target_sum": target.sum(-1).sum(0),
        "target_squared_sum": target.pow(2).sum(-1).sum(0),
        "prediction_sum": prediction.sum(-1).sum(0),
        "prediction_squared_sum": prediction.pow(2).sum(-1).sum(0),
        "prediction_target_sum": (prediction * target).sum(-1).sum(0),
    }

    for key, values in per_token.items():
        region_sums[key].index_add_(
            0, token_region_index, values.to(torch.float64)
        )


def metrics_from_sums(
    value_count,
    squared_error,
    absolute_error,
    target_sum,
    target_squared_sum,
    prediction_sum,
    prediction_squared_sum,
    prediction_target_sum,
):
    """Elementwise, so it serves per-region arrays and global scalars alike."""

    safe_count = np.maximum(value_count, 1.0)

    target_mean = target_sum / safe_count
    prediction_mean = prediction_sum / safe_count

    target_sst = target_squared_sum - value_count * target_mean ** 2
    prediction_sst = prediction_squared_sum - value_count * prediction_mean ** 2
    covariance = (
        prediction_target_sum - value_count * prediction_mean * target_mean
    )

    masked_mse = squared_error / safe_count
    masked_mae = absolute_error / safe_count

    # A sum of squares can reach zero through cancellation when a prediction is
    # nearly constant. Guard the ratios rather than clamping the denominator:
    # clamping produces spectacular nonsense (a correlation of 3.5e9 was the
    # symptom that led to this).
    target_sst = np.maximum(target_sst, 0.0)
    prediction_sst = np.maximum(prediction_sst, 0.0)

    masked_r2 = np.where(
        target_sst > 1e-12,
        1.0 - squared_error / np.maximum(target_sst, 1e-12),
        np.nan,
    )

    denominator = np.sqrt(prediction_sst * target_sst)
    masked_correlation = np.where(
        denominator > 1e-12,
        covariance / np.maximum(denominator, 1e-12),
        np.nan,
    )

    zero_baseline_mse = target_squared_sum / safe_count

    empty = value_count <= 0
    return {
        "value_count": value_count,
        "masked_mse": np.where(empty, np.nan, masked_mse),
        "masked_mae": np.where(empty, np.nan, masked_mae),
        "masked_r2": np.where(empty, np.nan, masked_r2),
        "masked_correlation": np.where(empty, np.nan, masked_correlation),
        "zero_baseline_mse": np.where(empty, np.nan, zero_baseline_mse),
    }


def finalize_region_metrics(region_sums, region_labels):
    """Returns (global_metrics dict, per-region DataFrame)."""

    arrays = {
        key: value.detach().cpu().numpy().astype(np.float64)
        for key, value in region_sums.items()
    }

    region_metrics = metrics_from_sums(**arrays)
    global_metrics = metrics_from_sums(
        **{key: value.sum() for key, value in arrays.items()}
    )

    region_metrics_df = pd.DataFrame({
        "region_index": np.arange(config.N_REGIONS, dtype=np.int64),
        "region_label": region_labels,
        "masked_value_count": region_metrics["value_count"],
        "masked_mse": region_metrics["masked_mse"],
        "masked_mae": region_metrics["masked_mae"],
        "masked_r2": region_metrics["masked_r2"],
        "masked_correlation": region_metrics["masked_correlation"],
        "zero_baseline_mse": region_metrics["zero_baseline_mse"],
    })

    return {k: float(v) for k, v in global_metrics.items()}, region_metrics_df
