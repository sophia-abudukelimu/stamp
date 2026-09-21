"""
Attention rollout (Abnar & Zuidema, 2020), adapted to a block-causal model.

    A_hat   = w * I + (1 - w) * A        then row-normalized
    rollout = A_hat_L @ ... @ A_hat_1

The residual term `w * I` accounts for the skip connection: a token carries its
own content forward whether or not attention selects it. With w = 0.5 the
identity and the attention contribute equally at each layer.

Heads are collapsed with a MAX inside each window, then averaged across
windows. Max keeps whatever the strongest of the four heads found for a given
(query, key) pair; a mean would dilute a pattern that only one head carries.
Reducing per window before averaging is the right order for the same reason:
averaging windows first washes out peaks that only some windows contain.

A note on CLS_MODE = "bidirectional": with the CLS column open, the product
across layers contains the path token -> CLS -> token, so entries above the
block diagonal are no longer exactly zero. `future_mass_fraction` records how
much rollout mass ends up there, so the rollout figures can still be read
honestly. Under "sink" it is exactly 0.
"""

import numpy as np
import pandas as pd
import torch

import config
from models.transformer import TOKEN_PATCH_INDEX, create_spatiotemporal_mask


def compute_attention_rollout(
    model, loader, region_labels, device, mask_seed=0, n_masked_tokens=None
):
    model.eval()

    layer_sums = [
        torch.zeros(
            config.SEQUENCE_LENGTH, config.SEQUENCE_LENGTH,
            dtype=torch.float64, device=device,
        )
        for _ in range(config.NUM_LAYERS)
    ]

    mask_generator = torch.Generator(device=device).manual_seed(mask_seed)
    total_windows = 0

    with torch.no_grad():
        for windows, _ in loader:

            windows = windows.to(device, dtype=torch.float32, non_blocking=True)
            batch_size = windows.shape[0]

            token_mask = None
            if config.ROLLOUT_USE_MASKING and n_masked_tokens is not None:
                token_mask = create_spatiotemporal_mask(
                    batch_size, mask_generator, n_masked_tokens, device
                )

            _, _, _, attention_maps = model(
                windows, token_mask, return_attention=True
            )

            for layer_index, attention in enumerate(attention_maps):
                # (B, NHEAD, S, S) -> max over heads -> (B, S, S)
                reduced = attention.to(torch.float64).max(dim=1).values
                layer_sums[layer_index] += reduced.sum(dim=0)

            total_windows += batch_size
            del attention_maps

    if total_windows == 0:
        raise RuntimeError("Rollout loader produced no windows.")

    layer_mean_attention = [s / float(total_windows) for s in layer_sums]

    identity = torch.eye(
        config.SEQUENCE_LENGTH, dtype=torch.float64, device=device
    )
    rollout = None
    for mean_attention in layer_mean_attention:
        augmented = (
            config.ROLLOUT_RESIDUAL_WEIGHT * identity
            + (1.0 - config.ROLLOUT_RESIDUAL_WEIGHT) * mean_attention
        )
        augmented = augmented / augmented.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
        rollout = augmented if rollout is None else augmented @ rollout

    rollout_numpy = rollout.detach().cpu().numpy()

    # ---- token block, CLS row and column excluded ---------------------------
    token_block = rollout_numpy[1:, 1:]
    query_patch = TOKEN_PATCH_INDEX.numpy()
    future_key = query_patch[None, :] > query_patch[:, None]
    future_mass_fraction = float(
        token_block[future_key].sum() / max(token_block.sum(), 1e-12)
    )

    # ---- CLS row: how much each token contributes to the CLS embedding ------
    cls_token_map = (
        rollout_numpy[0, 1:]
        .reshape(config.N_TIME_PATCHES, config.N_REGIONS)
        .T.copy()
    )
    region_rollout = cls_token_map.sum(axis=1)

    # ---- region x region, split by temporal lag -----------------------------
    # This is the part that separates "who talks to whom at the same instant"
    # from "whose past predicts whose present".
    blocks = token_block.reshape(
        config.N_TIME_PATCHES, config.N_REGIONS,
        config.N_TIME_PATCHES, config.N_REGIONS,
    )

    region_to_region = blocks.sum(axis=2).mean(axis=0)

    lag0 = np.zeros((config.N_REGIONS, config.N_REGIONS))
    lag_positive = np.zeros((config.N_REGIONS, config.N_REGIONS))
    for query_patch_index in range(config.N_TIME_PATCHES):
        for key_patch_index in range(config.N_TIME_PATCHES):
            block = blocks[query_patch_index, :, key_patch_index, :]
            if key_patch_index == query_patch_index:
                lag0 += block
            elif key_patch_index < query_patch_index:
                lag_positive += block
    lag0 /= config.N_TIME_PATCHES
    lag_positive /= config.N_TIME_PATCHES

    return {
        "layer_mean_attention": [
            layer.detach().cpu().numpy() for layer in layer_mean_attention
        ],
        "rollout": rollout_numpy,
        "cls_token_map": cls_token_map,
        "region_rollout": region_rollout,
        "region_to_region": region_to_region,
        "region_to_region_lag0": lag0,
        "region_to_region_lag_positive": lag_positive,
        "future_mass_fraction": future_mass_fraction,
        "windows_used": total_windows,
        "region_labels": list(region_labels),
    }


def build_region_rollout_frame(rollout_output, region_labels):
    values = rollout_output["region_rollout"]
    return pd.DataFrame({
        "region_index": np.arange(config.N_REGIONS, dtype=np.int64),
        "region_label": region_labels,
        "attention_rollout": values,
        "attention_rollout_normalized": values / max(float(values.sum()), 1e-12),
    })


def save_rollout(rollout_output, region_labels, path_npz, path_csv):
    np.savez_compressed(
        path_npz,
        region_rollout=rollout_output["region_rollout"],
        cls_token_map=rollout_output["cls_token_map"],
        region_to_region=rollout_output["region_to_region"],
        region_to_region_lag0=rollout_output["region_to_region_lag0"],
        region_to_region_lag_positive=(
            rollout_output["region_to_region_lag_positive"]
        ),
        rollout_full=rollout_output["rollout"],
        layer_mean_attention=np.stack(rollout_output["layer_mean_attention"]),
        region_labels=np.array(region_labels, dtype=object),
        head_reduction=np.array(config.ROLLOUT_HEAD_REDUCTION),
        cls_mode=np.array(config.CLS_MODE),
        future_mass_fraction=np.array(rollout_output["future_mass_fraction"]),
        windows_used=np.array(rollout_output["windows_used"]),
    )
    build_region_rollout_frame(rollout_output, region_labels).to_csv(
        path_csv, index=False
    )
    return path_npz
