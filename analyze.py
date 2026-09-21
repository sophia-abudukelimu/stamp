"""
STAMP - stage 2 of 2: read a finished training run and produce the analysis.

    python analyze.py --run runs/demo --grid synthetic_grid.npz

Reads only the files `train.py` wrote, so it can be run days later, on another
machine, without a GPU, and without the original .h5.

Two questions are answered here.

  R2  -  WHICH regions can the model reconstruct, and does that change with
         age and genotype? Reported per region, per age x genotype cell, as
         consecutive-age differences and as AD minus WT, painted on the cortex.

  ROLLOUT  -  WHERE does the model look? Reported as region x region, split by
              temporal lag, and as the contribution of each (region, time) cell
              to the CLS embedding.

Outputs land in <run>/figures/ plus a few summary tables in <run>/.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

import config
from visualization import figures
from visualization.brain_maps import Parcellation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze a finished STAMP training run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run", required=True,
                        help="directory produced by train.py")
    parser.add_argument("--grid", default=None,
                        help="grid.mat or synthetic_grid.npz. Without it the "
                             "brain maps are skipped and everything else "
                             "still runs.")
    parser.add_argument("--reshape-order", default=None, choices=["F", "C"],
                        help="override if the brain comes out transposed")
    parser.add_argument("--flip-vertical", action="store_true")
    parser.add_argument("--flip-horizontal", action="store_true")
    parser.add_argument("--figures-dir", default=None)
    return parser.parse_args()


def load_run(run_dir):
    """Every artifact train.py wrote, as one dict. Missing files are None."""

    def read_csv(name):
        path = os.path.join(run_dir, name)
        return pd.read_csv(path) if os.path.exists(path) else None

    bundle = {
        "history": read_csv("history.csv"),
        "region_metrics": read_csv("region_metrics.csv"),
        "group_r2_frame": read_csv("region_r2_by_age_genotype.csv"),
        "causal_leak": read_csv("causal_leak.csv"),
        "split_summary": read_csv("split_summary.csv"),
        "window_counts": read_csv("window_counts_by_age_genotype.csv"),
    }

    config_path = os.path.join(run_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as handle:
            saved = json.load(handle)
        # Make the analysis inherit the settings the run was trained with,
        # so figure titles never claim the wrong mask ratio.
        for name in ["MASK_RATIO", "CLS_MODE", "N_REGIONS", "N_TIME_PATCHES",
                     "PATCH_LENGTH", "FRAME_RATE", "ROLLOUT_HEAD_REDUCTION"]:
            if name in saved:
                setattr(config, name, saved[name])
        bundle["config"] = saved

    labels_path = os.path.join(run_dir, "region_labels.json")
    if os.path.exists(labels_path):
        with open(labels_path) as handle:
            bundle["region_labels"] = json.load(handle)
    elif bundle["region_metrics"] is not None:
        bundle["region_labels"] = bundle["region_metrics"][
            "region_label"
        ].tolist()
    else:
        bundle["region_labels"] = [
            f"ROI_{i:02d}" for i in range(config.N_REGIONS)
        ]

    rollout_path = os.path.join(run_dir, "attention_rollout.npz")
    if os.path.exists(rollout_path):
        with np.load(rollout_path, allow_pickle=True) as handle:
            bundle["rollout"] = {key: handle[key] for key in handle.files}
    else:
        bundle["rollout"] = None

    return bundle


def build_group_lookup(frame):
    """(age, genotype) -> per-region R2 array, plus the window and mouse counts."""
    group_r2, group_counts = {}, {}
    if frame is None:
        return group_r2, group_counts

    for (age, genotype), subset in frame.groupby(["age", "genotype"]):
        subset = subset.sort_values("region_index")
        group_r2[(age, genotype)] = subset["masked_r2"].to_numpy()
        group_counts[(age, genotype)] = (
            int(subset["n_windows"].iloc[0]),
            int(subset["n_mice"].iloc[0]),
        )
    return group_r2, group_counts


def main():
    args = parse_args()
    run_dir = args.run
    figures_dir = args.figures_dir or os.path.join(run_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    bundle = load_run(run_dir)
    region_labels = bundle["region_labels"]
    written = []

    print("=" * 72)
    print(f"STAMP analysis  ·  {run_dir}")
    print(f"mask {config.MASK_RATIO:.0%}  ·  CLS {config.CLS_MODE}")
    print("=" * 72)

    # ---- 1. training curves -------------------------------------------------

    if bundle["history"] is not None:
        history = bundle["history"]
        written.append(figures.plot_training_curves(history, figures_dir))
        written.append(os.path.join(figures_dir, "overfitting_check.png"))

        final = history.iloc[-1]
        best = history.loc[history["validation_total_loss"].idxmin()]
        print()
        print("TRAINING")
        print(f"  epochs run            {int(final['epoch'])}")
        print(f"  best epoch            {int(best['epoch'])}")
        print(f"  final validation R2   {final['validation_masked_r2']:.4f}")
        print(f"  final validation MSE  {final['validation_masked_mse']:.4f}")
        print(f"  zero baseline MSE     "
              f"{final['validation_zero_baseline_mse']:.4f}  "
              f"(must be near 1.0)")
        gap = float(
            final["train_masked_r2"] - final["validation_masked_r2"]
        )
        print(f"  train − validation R2 {gap:+.4f}  "
              f"({'no overfitting' if gap < 0.02 else 'CHECK THIS'})")

    # ---- 2. per-region R2 ---------------------------------------------------

    group_r2, group_counts = build_group_lookup(bundle["group_r2_frame"])

    if bundle["region_metrics"] is not None:
        region_metrics = bundle["region_metrics"].sort_values("region_index")
        written.append(figures.plot_region_r2_bars(region_metrics, figures_dir))

        values = region_metrics["masked_r2"].to_numpy()
        order = np.argsort(values)[::-1]
        print()
        print("PER-REGION RECONSTRUCTION (validation)")
        print(f"  mean R2   {np.nanmean(values):.4f}")
        print(f"  range     {np.nanmin(values):.4f} to {np.nanmax(values):.4f}")
        print("  easiest   " + ", ".join(
            f"{region_labels[i]} ({values[i]:.3f})" for i in order[:5]
        ))
        print("  hardest   " + ", ".join(
            f"{region_labels[i]} ({values[i]:.3f})" for i in order[-5:]
        ))

    # ---- 3. age and genotype ------------------------------------------------

    if group_r2:
        written.append(figures.plot_r2_trajectories(group_r2, figures_dir))

        rows = []
        for age in config.AGE_ORDER:
            row = {"age": age}
            for genotype in config.GENOTYPES:
                values = group_r2.get((age, genotype))
                row[config.GENOTYPE_LABELS[genotype]] = (
                    float(np.nanmean(values)) if values is not None else np.nan
                )
            wt, mt = group_r2.get((age, "wt")), group_r2.get((age, "mt"))
            row["AD − WT"] = (
                float(np.nanmean(mt) - np.nanmean(wt))
                if wt is not None and mt is not None else np.nan
            )
            rows.append(row)

        summary = pd.DataFrame(rows)
        summary.to_csv(
            os.path.join(run_dir, "r2_by_age_genotype_summary.csv"), index=False
        )
        print()
        print("MEAN R2 BY AGE AND GENOTYPE")
        print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- 4. brain maps ------------------------------------------------------

    if args.grid is not None and os.path.exists(args.grid):
        kwargs = {
            "flip_vertical": args.flip_vertical,
            "flip_horizontal": args.flip_horizontal,
        }
        if args.reshape_order is not None:
            kwargs["reshape_order"] = args.reshape_order

        parcellation = Parcellation.from_file(args.grid, **kwargs)
        counts = parcellation.region_pixel_counts()
        print()
        print("PARCELLATION")
        print(f"  grid          {parcellation.grid_side} x "
              f"{parcellation.grid_side}")
        print(f"  brain pixels  {int(parcellation.brain_mask.sum())}")
        print(f"  per region    min {int(counts.min())}, "
              f"max {int(counts.max())}")
        empty = np.flatnonzero(counts == 0)
        if len(empty):
            print(f"  WARNING regions with no pixels: {empty.tolist()}")

        if ("all", "all") in group_r2:
            written.append(figures.plot_brain_r2_all(
                parcellation, group_r2[("all", "all")], figures_dir
            ))
        if group_r2:
            written.append(figures.plot_brain_r2_by_age_genotype(
                parcellation, group_r2, group_counts, figures_dir
            ))
            path = figures.plot_brain_r2_age_differences(
                parcellation, group_r2, figures_dir
            )
            if path:
                written.append(path)
            path = figures.plot_brain_r2_genotype_difference(
                parcellation, group_r2, figures_dir
            )
            if path:
                written.append(path)
        if bundle["rollout"] is not None:
            written.append(figures.plot_brain_rollout(
                parcellation, bundle["rollout"]["region_rollout"], figures_dir
            ))
    else:
        print()
        print("No parcellation given (--grid), brain maps skipped.")

    # ---- 5. attention -------------------------------------------------------

    if bundle["rollout"] is not None:
        rollout = bundle["rollout"]

        from models.transformer import build_block_causal_mask
        written.append(figures.plot_attention_mask_and_rollout(
            build_block_causal_mask().numpy(), rollout["rollout_full"], figures_dir
        ))
        written.append(figures.plot_region_to_region(
            rollout, region_labels, figures_dir
        ))
        written.append(figures.plot_rollout_region_time(
            rollout, region_labels, figures_dir
        ))

        lag0 = rollout["region_to_region_lag0"]
        lag_positive = rollout["region_to_region_lag_positive"]
        off_diagonal = ~np.eye(config.N_REGIONS, dtype=bool)

        print()
        print("ATTENTION ROLLOUT")
        print(f"  windows used                {int(rollout['windows_used'])}")
        print(f"  head reduction              "
              f"{rollout['head_reduction']}")
        print(f"  mass on future keys         "
              f"{float(rollout['future_mass_fraction']):.3%}  "
              f"(0 when CLS_MODE = sink)")
        print(f"  off-diagonal mass, lag 0    "
              f"{lag0[off_diagonal].sum() / max(lag0.sum(), 1e-12):.3%}")
        print(f"  off-diagonal mass, lag >= 1 "
              f"{lag_positive[off_diagonal].sum() / max(lag_positive.sum(), 1e-12):.3%}")
        print("  If the second number is much larger than the first, regions")
        print("  relate to each other ACROSS time rather than within a moment.")

        top = np.argsort(rollout["region_rollout"])[::-1][:5]
        print("  top regions to CLS          " + ", ".join(
            region_labels[i] for i in top
        ))

    # ---- 6. causality -------------------------------------------------------

    if bundle["causal_leak"] is not None:
        leak = bundle["causal_leak"].iloc[0]
        print()
        print("CAUSALITY")
        print(f"  CLS mode        {leak['cls_mode']}")
        print(f"  future moves    {leak['future_shift']:.4e}")
        print(f"  past moves      {leak['past_shift']:.4e}")
        print(f"  leak ratio      {leak['leak_ratio']:.3%}")
        if leak["cls_mode"] == "sink":
            print("  Strictly causal: the past is untouched by the future.")
        else:
            print("  Relaxed by design so that CLS receives gradient from the")
            print("  reconstruction loss. Quote the leak ratio when reporting.")

    print()
    print("=" * 72)
    print(f"{len(written)} figures written to {figures_dir}")
    for path in written:
        print(f"  {os.path.basename(path)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
