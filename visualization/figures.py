"""
Every figure `analyze.py` produces.

Each function takes what it needs, saves a PNG, and returns the path. None of
them read files or reach for globals, so they can be reused from a notebook.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from visualization.palettes import (
    DIFF_CMAP, R2_CMAP, R2_NORM, ROLLOUT_CMAP, symmetric_norm,
)


def _save(figure, output_dir, name, dpi=200):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


# -----------------------------------------------------------------------------
# 1. Training curves
# -----------------------------------------------------------------------------

def plot_training_curves(history, output_dir):
    """MSE, MAE, R2, correlation, and the train-minus-validation gap."""

    panels = [
        ("masked_mse", "Masked MSE", None),
        ("masked_mae", "Masked MAE", None),
        ("masked_r2", "Masked R²", None),
        ("masked_correlation", "Masked correlation r", None),
    ]

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, (column, title, _) in zip(axes.ravel(), panels):
        axis.plot(history["epoch"], history[f"train_{column}"],
                  label="train", color="#4E7BA8", linewidth=1.8)
        axis.plot(history["epoch"], history[f"validation_{column}"],
                  label="validation", color="#B93B29", linewidth=1.8)
        axis.set_xlabel("Epoch")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)

    figure.suptitle(
        f"Training curves  ·  mask {config.MASK_RATIO:.0%}  ·  "
        f"CLS {config.CLS_MODE}", fontsize=13,
    )
    figure.tight_layout()
    curves_path = _save(figure, output_dir, "training_curves.png")

    # The overfitting check gets its own panel because it is the one people
    # ask about. A negative and flat gap means no overfitting; dropout is on
    # during training and off during validation, which is why it sits below
    # zero rather than at it.
    figure, axis = plt.subplots(figsize=(8, 4.5))
    gap = history["train_masked_r2"] - history["validation_masked_r2"]
    axis.axhline(0, color="#333333", linewidth=0.9)
    axis.plot(history["epoch"], gap, color="#2E5580", linewidth=1.9)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("train R² − validation R²")
    axis.set_title(
        "Overfitting check\npositive and growing would mean memorization"
    )
    axis.grid(alpha=0.25)
    figure.tight_layout()
    _save(figure, output_dir, "overfitting_check.png")

    return curves_path


# -----------------------------------------------------------------------------
# 2. Per-region R2
# -----------------------------------------------------------------------------

def plot_region_r2_bars(region_metrics, output_dir):
    values = region_metrics["masked_r2"].to_numpy()
    labels = region_metrics["region_label"].tolist()
    order = np.argsort(values)

    figure, axis = plt.subplots(figsize=(8, max(7, config.N_REGIONS * 0.22)))
    axis.barh(
        np.arange(config.N_REGIONS), values[order],
        color=R2_CMAP(R2_NORM(values[order])),
        edgecolor="#555555", linewidth=0.4,
    )
    axis.axvline(0, color="#333333", linewidth=0.8)
    axis.set_yticks(np.arange(config.N_REGIONS))
    axis.set_yticklabels([labels[i] for i in order], fontsize=7)
    axis.set_xlabel("Masked R²")
    axis.set_title(f"Per-region reconstruction, mask {config.MASK_RATIO:.0%}")
    axis.grid(alpha=0.25, axis="x")
    figure.tight_layout()
    return _save(figure, output_dir, "region_r2_bars.png")


def plot_brain_r2_all(parcellation, values, output_dir):
    figure, axis = plt.subplots(figsize=(6.5, 6))
    handle = parcellation.draw(
        axis, values,
        f"All mice, all ages  ·  mask {config.MASK_RATIO:.0%}",
        R2_CMAP, R2_NORM,
    )
    figure.colorbar(handle, ax=axis, fraction=0.045, label="Masked R²")
    figure.tight_layout()
    return _save(figure, output_dir, "brain_r2_all.png")


def plot_brain_r2_by_age_genotype(parcellation, group_r2, group_counts,
                                  output_dir):
    ages = [a for a in config.AGE_ORDER if ("wt", a) in
            {(g, a) for (a, g) in group_r2 if isinstance(group_r2, dict)}] \
        or config.AGE_ORDER

    figure, axes = plt.subplots(
        2, len(ages), figsize=(3.1 * len(ages), 7.0), squeeze=False
    )
    handle = None
    for row, genotype in enumerate(config.GENOTYPES):
        for column, age in enumerate(ages):
            axis = axes[row][column]
            values = group_r2.get((age, genotype))
            if values is None:
                axis.text(0.5, 0.5, "no data", ha="center", va="center",
                          fontsize=9, color="#888888")
                axis.axis("off")
                continue
            _, n_mice = group_counts[(age, genotype)]
            handle = parcellation.draw(
                axis, values,
                f"{config.GENOTYPE_LABELS[genotype]}  {age}\n{n_mice} mice",
                R2_CMAP, R2_NORM, fontsize=10,
            )

    if handle is not None:
        figure.colorbar(handle, ax=axes, fraction=0.018, pad=0.02,
                        label="Masked R²")
    figure.suptitle(
        f"Per-region masked R² by age and genotype  ·  "
        f"mask {config.MASK_RATIO:.0%}", fontsize=14,
    )
    return _save(figure, output_dir, "brain_r2_by_age_genotype.png")


def plot_brain_r2_age_differences(parcellation, group_r2, output_dir):
    """Consecutive ages, within genotype. Not a derivative, just b minus a."""
    pairs = list(zip(config.AGE_ORDER[:-1], config.AGE_ORDER[1:]))

    differences = {}
    for genotype in config.GENOTYPES:
        for earlier, later in pairs:
            a, b = group_r2.get((earlier, genotype)), group_r2.get(
                (later, genotype)
            )
            if a is not None and b is not None:
                differences[(genotype, earlier, later)] = b - a

    if not differences:
        return None

    norm = symmetric_norm(np.concatenate(list(differences.values())))

    figure, axes = plt.subplots(
        2, len(pairs), figsize=(3.1 * len(pairs), 7.0), squeeze=False
    )
    handle = None
    for row, genotype in enumerate(config.GENOTYPES):
        for column, (earlier, later) in enumerate(pairs):
            axis = axes[row][column]
            values = differences.get((genotype, earlier, later))
            if values is None:
                axis.text(0.5, 0.5, "no data", ha="center", va="center",
                          fontsize=9, color="#888888")
                axis.axis("off")
                continue
            handle = parcellation.draw(
                axis, values,
                f"{config.GENOTYPE_LABELS[genotype]}\n{later} − {earlier}",
                DIFF_CMAP, norm, fontsize=10,
            )

    if handle is not None:
        figure.colorbar(handle, ax=axes, fraction=0.018, pad=0.02,
                        label="Δ Masked R²")
    figure.suptitle("Change in reconstruction between consecutive ages",
                    fontsize=14)
    return _save(figure, output_dir, "brain_r2_age_differences.png")


def plot_brain_r2_genotype_difference(parcellation, group_r2, output_dir):
    """AD minus WT, pooled and per age. The headline comparison."""
    entries = [("all ages", group_r2.get(("all", "mt")),
                group_r2.get(("all", "wt")))]
    for age in config.AGE_ORDER:
        entries.append((age, group_r2.get((age, "mt")),
                        group_r2.get((age, "wt"))))

    usable = [(name, mt - wt) for name, mt, wt in entries
              if mt is not None and wt is not None]
    if not usable:
        return None

    norm = symmetric_norm(np.concatenate([v for _, v in usable]))

    figure, axes = plt.subplots(
        1, len(usable), figsize=(3.1 * len(usable), 4.2), squeeze=False
    )
    handle = None
    for column, (name, values) in enumerate(usable):
        handle = parcellation.draw(
            axes[0][column], values, f"AD − WT\n{name}",
            DIFF_CMAP, norm, fontsize=10,
        )
    figure.colorbar(handle, ax=axes, fraction=0.02, pad=0.02,
                    label="Δ Masked R²  (AD − WT)")
    figure.suptitle(
        "Genotype difference in reconstruction  ·  "
        "blue = AD less predictable than WT", fontsize=13,
    )
    return _save(figure, output_dir, "brain_r2_genotype_difference.png")


def plot_r2_trajectories(group_r2, output_dir):
    """Mean per-region R2 against age, one line per genotype."""
    figure, axis = plt.subplots(figsize=(7, 4.6))
    colors = {"wt": "#2E5580", "mt": "#B93B29"}

    for genotype in config.GENOTYPES:
        xs, ys = [], []
        for index, age in enumerate(config.AGE_ORDER):
            values = group_r2.get((age, genotype))
            if values is None:
                continue
            xs.append(index)
            ys.append(float(np.nanmean(values)))
        if xs:
            axis.plot(xs, ys, "o-", color=colors[genotype], linewidth=2.0,
                      markersize=6, label=config.GENOTYPE_LABELS[genotype])

    axis.set_xticks(range(len(config.AGE_ORDER)))
    axis.set_xticklabels(config.AGE_ORDER)
    axis.set_xlabel("Age")
    axis.set_ylabel("Mean masked R² across regions")
    axis.set_title(f"Reconstruction across the lifespan  ·  "
                   f"mask {config.MASK_RATIO:.0%}")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    return _save(figure, output_dir, "r2_trajectories.png")


# -----------------------------------------------------------------------------
# 3. Attention
# -----------------------------------------------------------------------------

def plot_attention_mask_and_rollout(allow_mask, rollout, output_dir):
    figure, axes = plt.subplots(1, 2, figsize=(13, 6))

    axes[0].imshow(allow_mask.astype(float), cmap="gray",
                   interpolation="nearest")
    axes[0].set_title("Allowed attention\nwhite = allowed, black = blocked")
    axes[0].set_xlabel("Key index")
    axes[0].set_ylabel("Query index")

    with np.errstate(divide="ignore"):
        log_rollout = np.log10(np.maximum(rollout, 1e-12))
    axes[1].imshow(log_rollout, cmap="Oranges", interpolation="nearest")
    axes[1].set_title(
        f"Attention rollout, log10\nmask {config.MASK_RATIO:.0%}, "
        f"heads collapsed with {config.ROLLOUT_HEAD_REDUCTION}"
    )
    axes[1].set_xlabel("Key index")
    axes[1].set_ylabel("Query index")

    figure.tight_layout()
    return _save(figure, output_dir, "attention_mask_and_rollout.png")


def plot_region_to_region(rollout_output, region_labels, output_dir):
    """
    The three-panel figure that separates same-instant coupling from delayed
    coupling. The middle and right panels are the point: if the off-diagonal
    structure lives only in the right panel, regions talk to each other across
    time rather than within a moment.
    """
    panels = [
        ("region_to_region", "All lags pooled"),
        ("region_to_region_lag0", "Lag 0  ·  same time patch"),
        ("region_to_region_lag_positive", "Lag ≥ 1  ·  past patches only"),
    ]

    figure, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    for axis, (key, title) in zip(axes, panels):
        matrix = rollout_output[key]
        handle = axis.imshow(matrix, cmap=ROLLOUT_CMAP,
                             interpolation="nearest")
        axis.set_title(f"{title}\nrow = query region, column = key region",
                       fontsize=10)
        axis.set_xlabel("Key region")
        axis.set_ylabel("Query region")
        figure.colorbar(handle, ax=axis, fraction=0.046)

    figure.suptitle(
        f"Region-to-region attention rollout  ·  "
        f"mask {config.MASK_RATIO:.0%}  ·  heads "
        f"{config.ROLLOUT_HEAD_REDUCTION}", fontsize=13,
    )
    figure.tight_layout()
    return _save(figure, output_dir, "region_to_region_rollout.png")


def plot_rollout_region_time(rollout_output, region_labels, output_dir):
    cls_map = rollout_output["cls_token_map"]

    figure, axes = plt.subplots(
        1, 2, figsize=(15, 9), gridspec_kw={"width_ratios": [1, 1.25]}
    )

    handle = axes[0].imshow(cls_map, aspect="auto", cmap=ROLLOUT_CMAP,
                            interpolation="nearest")
    axes[0].set_yticks(np.arange(config.N_REGIONS))
    axes[0].set_yticklabels(region_labels, fontsize=7)
    axes[0].set_xticks(np.arange(config.N_TIME_PATCHES))
    axes[0].set_xticklabels(
        [f"t{i}" for i in range(config.N_TIME_PATCHES)]
    )
    axes[0].set_xlabel(
        f"Time patch ({config.PATCH_LENGTH / config.FRAME_RATE:.1f} s each)"
    )
    axes[0].set_title("Rollout mass, region × time")
    figure.colorbar(handle, ax=axes[0], fraction=0.035, label="Rollout weight")

    for region in range(config.N_REGIONS):
        axes[1].plot(cls_map[region], color="#7FA6C6", linewidth=0.8, alpha=0.6)
    axes[1].plot(cls_map.mean(axis=0), color="#17304F", linewidth=2.6,
                 label="mean over regions")
    axes[1].set_xticks(np.arange(config.N_TIME_PATCHES))
    axes[1].set_xticklabels([f"t{i}" for i in range(config.N_TIME_PATCHES)])
    axes[1].set_xlabel("Time patch")
    axes[1].set_ylabel("Rollout weight")
    # Stated carefully: an EARLY patch is visible to every later query, so it
    # accumulates mass from more queries. The high value at t0 is structural.
    axes[1].set_title(
        "Earlier patches are readable by more queries,\n"
        "so the peak at t0 is structural, not a finding"
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    figure.tight_layout()
    return _save(figure, output_dir, "rollout_region_time.png")


def plot_brain_rollout(parcellation, region_rollout, output_dir):
    normalized = region_rollout / max(float(region_rollout.sum()), 1e-12)
    norm = matplotlib.colors.Normalize(
        vmin=0.0, vmax=float(np.nanmax(normalized))
    )

    figure, axis = plt.subplots(figsize=(6.5, 6))
    handle = parcellation.draw(
        axis, normalized,
        f"Attention rollout to CLS  ·  mask {config.MASK_RATIO:.0%}",
        ROLLOUT_CMAP, norm,
    )
    figure.colorbar(handle, ax=axis, fraction=0.045,
                    label="Normalized rollout weight")
    figure.tight_layout()
    return _save(figure, output_dir, "brain_attention_rollout.png")
