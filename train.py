"""
STAMP - stage 1 of 2: train the masked spatio-temporal transformer.

    python train.py --data synthetic_ach.h5 --out runs/demo --epochs 60

Everything is written to `--out` as it is produced, one file per artifact, so a
session that dies halfway still leaves usable results behind:

    config.json                     the exact configuration this run used
    session_metadata.csv            one row per recording session
    region_labels.json              the 41 bilateral region names
    window_counts_by_age_genotype.csv
    split_summary.csv               mice and windows per side of the split
    split_mice.json                 which mouse went where
    model_checks.json               parameter count, fused-vs-explicit, leak
    history.csv                     rewritten EVERY epoch
    checkpoint.pt                   best validation loss
    region_metrics.csv              per-region R2 on the validation set
    region_r2_by_age_genotype.csv   per-region R2 for every age x genotype cell
    causal_leak.csv                 leak measured on the trained model

Stage 2 is `analyze.py`, which reads this directory and produces the figures.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

import config
from data.datasets import SplitBundle, close_loader_workers
from data.loading import load_dataset
from models.transformer import STAMPEncoder, create_gradient_scaler, verify_model
from evaluation.leak import measure_causal_leak
from evaluation.rollout import compute_attention_rollout, save_rollout
from training.loop import run_pass


def make_logger(output_dir):
    """Prints and appends to run_log.txt, so a disconnected session keeps it."""
    path = os.path.join(output_dir, "run_log.txt")

    def log(message=""):
        text = str(message)
        print(text, flush=True)
        try:
            with open(path, "a") as handle:
                handle.write(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {text}\n"
                )
        except OSError:
            pass

    return log


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the STAMP encoder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="path to the .h5 file")
    parser.add_argument("--out", default="runs/demo", help="output directory")
    parser.add_argument("--epochs", type=int, default=config.SSL_EPOCHS)
    parser.add_argument("--mask-ratio", type=float, default=config.MASK_RATIO)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument(
        "--cls-mode", choices=["bidirectional", "sink"], default=config.CLS_MODE,
        help="'sink' is strictly causal but starves CLS of gradient; "
             "see config.py",
    )
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument(
        "--cpu", action="store_true", help="force CPU, for a smoke test"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Command-line overrides are written back into config so that every module
    # importing it sees the same values.
    config.SSL_EPOCHS = args.epochs
    config.MASK_RATIO = args.mask_ratio
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.lr
    config.NUM_WORKERS = args.num_workers
    config.CLS_MODE = args.cls_mode
    config.SEED = args.seed

    os.makedirs(args.out, exist_ok=True)
    log = make_logger(args.out)

    device = torch.device("cpu") if args.cpu else config.get_device()
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    config.reset_all_seeds(args.seed)

    log("=" * 72)
    log(f"STAMP training run  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 72)
    log(config.describe())
    log(f"Device                {device}")
    log(f"Output directory      {args.out}")
    log("")

    with open(os.path.join(args.out, "config.json"), "w") as handle:
        json.dump(
            {
                name: value
                for name, value in vars(config).items()
                if name.isupper()
                and isinstance(value, (bool, int, float, str, list))
            },
            handle, indent=2, sort_keys=True,
        )

    # ---- data ---------------------------------------------------------------

    log("-" * 72)
    log("DATA")
    log("-" * 72)
    all_windows, window_metadata, region_labels = load_dataset(
        args.data, output_dir=args.out, log=log
    )

    split = SplitBundle(
        all_windows, window_metadata,
        validation_fraction=config.VALIDATION_FRACTION, seed=args.seed,
    )
    split.save(args.out)
    log(split.summary().to_string(index=False))
    log("")

    # ---- model --------------------------------------------------------------

    log("-" * 72)
    log("MODEL CHECKS")
    log("-" * 72)
    checks = verify_model(device=device, log=log)
    with open(os.path.join(args.out, "model_checks.json"), "w") as handle:
        json.dump(checks, handle, indent=2)
    log("")

    n_masked = config.n_masked_tokens(config.MASK_RATIO)

    config.reset_all_seeds(args.seed)
    train_loader, validation_loader = split.build_loaders(
        sampler_seed=args.seed, num_workers=config.NUM_WORKERS
    )

    model = STAMPEncoder().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    gradient_scaler = create_gradient_scaler(device)

    # ---- training -----------------------------------------------------------

    log("-" * 72)
    log(f"TRAINING  {config.SSL_EPOCHS} epochs, mask {config.MASK_RATIO:.0%} "
        f"({n_masked}/{config.N_TOKENS} cells)")
    log("-" * 72)

    history_path = os.path.join(args.out, "history.csv")
    history_records = []
    best_state, best_epoch, best_loss = None, None, np.inf
    started = time.time()

    for epoch in range(1, config.SSL_EPOCHS + 1):

        train_metrics, _ = run_pass(
            model, train_loader, training=True,
            mask_seed=args.seed + epoch, n_masked_tokens=n_masked,
            region_labels=region_labels, device=device,
            optimizer=optimizer, gradient_scaler=gradient_scaler,
        )
        validation_metrics, _ = run_pass(
            model, validation_loader, training=False,
            mask_seed=args.seed + 100_000, n_masked_tokens=n_masked,
            region_labels=region_labels, device=device,
        )

        record = {"epoch": epoch}
        record.update({f"train_{k}": v for k, v in train_metrics.items()})
        record.update(
            {f"validation_{k}": v for k, v in validation_metrics.items()}
        )
        history_records.append(record)

        # Rewritten every epoch on purpose: an interrupted run keeps its curve.
        pd.DataFrame(history_records).to_csv(history_path, index=False)

        if validation_metrics["total_loss"] < best_loss:
            best_loss = validation_metrics["total_loss"]
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

        if epoch == 1 or epoch % 5 == 0 or epoch == config.SSL_EPOCHS:
            log(f"Epoch {epoch:03d}/{config.SSL_EPOCHS} | "
                f"train MSE={train_metrics['masked_mse']:.4f} "
                f"R2={train_metrics['masked_r2']:.3f} | "
                f"val MSE={validation_metrics['masked_mse']:.4f} "
                f"MAE={validation_metrics['masked_mae']:.4f} "
                f"R2={validation_metrics['masked_r2']:.3f} "
                f"r={validation_metrics['masked_correlation']:.3f}")

    if best_state is None:
        raise RuntimeError("No checkpoint was recorded.")

    model.load_state_dict(best_state)
    log(f"Best epoch {best_epoch}, validation loss {best_loss:.6f}")
    log("")

    # ---- what the CLS wiring cost, measured on the trained model ------------

    leak = measure_causal_leak(model, device=device)
    log(f"Causal leak (trained, CLS_MODE={config.CLS_MODE}):")
    log(f"  future tokens move  {leak['future_shift']:.4e}")
    log(f"  past tokens move    {leak['past_shift']:.4e}  "
        f"(leak ratio {leak['leak_ratio']:.3%})")
    if config.CLS_MODE == "sink":
        assert leak["past_shift"] < 1e-5, "A strictly causal model leaked."

    pd.DataFrame([{
        "cls_mode": config.CLS_MODE,
        "mask_ratio": config.MASK_RATIO,
        "best_epoch": best_epoch,
        **leak,
    }]).to_csv(os.path.join(args.out, "causal_leak.csv"), index=False)
    log("")

    # ---- final evaluation ---------------------------------------------------

    log("-" * 72)
    log("EVALUATION")
    log("-" * 72)

    final_metrics, region_metrics_df = run_pass(
        model, validation_loader, training=False,
        mask_seed=args.seed + 100_000, n_masked_tokens=n_masked,
        region_labels=region_labels, device=device,
    )
    region_metrics_df["mask_ratio"] = config.MASK_RATIO
    region_metrics_df["cls_mode"] = config.CLS_MODE
    region_metrics_df.to_csv(
        os.path.join(args.out, "region_metrics.csv"), index=False
    )

    log(f"Validation  MSE={final_metrics['masked_mse']:.6f}  "
        f"MAE={final_metrics['masked_mae']:.6f}  "
        f"R2={final_metrics['masked_r2']:.6f}  "
        f"r={final_metrics['masked_correlation']:.6f}")
    log(f"Zero-prediction baseline MSE = "
        f"{final_metrics['zero_baseline_mse']:.4f}  "
        f"(must be near 1.0 on z-scored windows)")
    log("")

    # ---- per age and genotype, so the brain maps never need a rerun ---------
    # Evaluated on ALL windows: the absolute R2 is optimistic on training mice,
    # but that bias is shared by every group, so comparisons between groups
    # stay meaningful.

    ages = window_metadata["age"].to_numpy()
    genotypes = window_metadata["genotype"].to_numpy()

    groups = [("all", "all", np.ones(len(window_metadata), dtype=bool))]
    for genotype in config.GENOTYPES:
        groups.append(("all", genotype, genotypes == genotype))
    for age in config.AGE_ORDER:
        for genotype in config.GENOTYPES:
            groups.append(
                (age, genotype, (ages == age) & (genotypes == genotype))
            )

    group_records = []
    log("Per-group evaluation:")
    for age, genotype, selector in groups:
        indices = np.flatnonzero(selector)
        if len(indices) == 0:
            continue

        group_metrics, group_region_df = run_pass(
            model, split.build_group_loader(indices), training=False,
            mask_seed=args.seed + 100_000, n_masked_tokens=n_masked,
            region_labels=region_labels, device=device,
        )
        n_mice = int(
            pd.Series(split.mouse_uids[indices]).nunique()
        )
        for region in range(config.N_REGIONS):
            group_records.append({
                "mask_ratio": config.MASK_RATIO,
                "cls_mode": config.CLS_MODE,
                "age": age,
                "genotype": genotype,
                "n_windows": len(indices),
                "n_mice": n_mice,
                "region_index": region,
                "region_label": region_labels[region],
                "masked_r2": group_region_df["masked_r2"].iloc[region],
                "masked_mse": group_region_df["masked_mse"].iloc[region],
                "masked_mae": group_region_df["masked_mae"].iloc[region],
                "masked_correlation":
                    group_region_df["masked_correlation"].iloc[region],
            })
        log(f"  {age:>5s} {genotype:>3s}  windows={len(indices):6d}  "
            f"mice={n_mice:3d}  R2={group_metrics['masked_r2']:.4f}")

    pd.DataFrame(group_records).to_csv(
        os.path.join(args.out, "region_r2_by_age_genotype.csv"), index=False
    )
    log("")

    # ---- attention rollout --------------------------------------------------

    log("-" * 72)
    log("ATTENTION ROLLOUT")
    log("-" * 72)

    rollout_loader = split.build_rollout_loader()
    rollout = compute_attention_rollout(
        model, rollout_loader, region_labels, device,
        mask_seed=args.seed + 200_000, n_masked_tokens=n_masked,
    )
    close_loader_workers(rollout_loader)

    save_rollout(
        rollout, region_labels,
        os.path.join(args.out, "attention_rollout.npz"),
        os.path.join(args.out, "region_attention_rollout.csv"),
    )

    top = np.argsort(rollout["region_rollout"])[::-1][:5]
    log(f"Windows used            {rollout['windows_used']}")
    log(f"Top regions by rollout  "
        f"{', '.join(region_labels[i] for i in top)}")
    log(f"Rollout mass on future keys  "
        f"{rollout['future_mass_fraction']:.3%}  "
        f"(0 when CLS_MODE = sink)")
    log("")

    # ---- checkpoint ---------------------------------------------------------

    checkpoint_path = os.path.join(args.out, "checkpoint.pt")
    torch.save(
        {
            "model_state_dict": best_state,
            "best_epoch": best_epoch,
            "best_validation_total_loss": best_loss,
            "final_validation_metrics": final_metrics,
            "region_labels": list(region_labels),
            "causal_leak": leak,
            "model_checks": checks,
            "train_mice": sorted(split.train_mice),
            "validation_mice": sorted(split.validation_mice),
            "configuration": {
                name: value
                for name, value in vars(config).items()
                if name.isupper()
                and isinstance(value, (bool, int, float, str, list))
            },
        },
        checkpoint_path,
    )

    close_loader_workers(train_loader)
    close_loader_workers(validation_loader)

    log(f"Wrote {checkpoint_path}")
    log(f"Finished in {(time.time() - started) / 60:.1f} min")
    log("")
    log(f"Next:  python analyze.py --run {args.out}")
    log("=" * 72)


if __name__ == "__main__":
    main()
