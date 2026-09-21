"""
Generate a synthetic ACh dataset with the same layout as the real recordings.

    python scripts/make_synthetic_data.py --out synthetic_ach.h5

WHY THIS EXISTS
---------------
The recordings this project was built on are unpublished and cannot be shared.
This script writes a file with an identical structure - same group layout, same
field names, same dtypes, same 82 channels at 10 Hz, same age and genotype
labels - filled with simulated signals. Every entry point in the repository
runs on it unchanged, so the whole pipeline can be executed end to end by
anyone.

    *** The signals are simulated. Any biological conclusion drawn from     ***
    *** this file is meaningless. It exists to exercise the code, not to    ***
    *** support a result.                                                   ***

It doubles as the executable specification of the input format: whatever this
script writes is exactly what `data/loading.py` expects to read.

HOW THE SIGNAL IS BUILT
-----------------------
1. K latent factors, each an AR(1) process with its own timescale. These stand
   in for slow brain-wide modes (arousal, locomotion, global cholinergic tone).

2. The 41 bilateral regions are assigned to a handful of modules. Regions in
   the same module load on the same factors, which is what produces block
   structure in the region x region rollout.

3. Every region reads its factors at a REGION-SPECIFIC DELAY of 0-3 seconds.
   This is the important one: it means a region's present is predictable from
   other regions' past but not from their present, so the lag >= 1 panel of the
   rollout analysis has something real to find.

4. Each region adds private AR(1) noise. The ratio of shared to private
   variance is the single knob that controls how predictable a region is, and
   therefore what R2 the model can reach.

5. Genotype and age set that ratio:
       WT       stays near `--wt-shared` at every age
       AD (DKI) starts at the same place at 3 months and decays towards
                `--ad-shared-final` by 15 months
   so the intended read-out of the demo is "AD regions become progressively
   less predictable from the rest of the cortex with age", which is the shape
   of hypothesis the real model is meant to test.

6. Each raw channel is one hemisphere of a region: the region signal plus a
   little independent hemisphere noise, interleaved as
   (left_0, right_0, left_1, right_1, ...) so that averaging adjacent pairs
   recovers the region.

A fraction of frames is marked invalid in `ml_valid_frame`, and a few short
stretches are set to NaN, so the quality-control path in `data/loading.py` is
actually exercised rather than skipped.
"""

import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402


# -----------------------------------------------------------------------------
# Signal construction
# -----------------------------------------------------------------------------

def ar1_process(n_frames, n_series, timescale_frames, rng):
    """AR(1) with unit stationary variance."""
    phi = np.exp(-1.0 / np.maximum(timescale_frames, 1e-6))
    phi = np.broadcast_to(np.atleast_1d(phi), (n_series,)).astype(np.float64)
    innovation_scale = np.sqrt(np.maximum(1.0 - phi ** 2, 1e-9))

    series = np.zeros((n_frames, n_series), dtype=np.float64)
    series[0] = rng.standard_normal(n_series)
    noise = rng.standard_normal((n_frames, n_series))
    for t in range(1, n_frames):
        series[t] = phi * series[t - 1] + innovation_scale * noise[t]
    return series


def build_region_structure(n_regions, n_factors, n_modules, frame_rate,
                           global_weight, rng):
    """
    Fixed across the whole dataset: which regions belong to which module, how
    they load on the latent factors, and how delayed each one is.

    Keeping this fixed is what makes the region x region structure consistent
    between mice, so that averaging the rollout over windows converges to
    something rather than to noise.
    """
    module_of_region = np.sort(rng.integers(0, n_modules, size=n_regions))

    module_loadings = rng.standard_normal((n_modules, n_factors))
    module_loadings /= np.linalg.norm(module_loadings, axis=1, keepdims=True)

    loadings = (
        module_loadings[module_of_region]
        + 0.35 * rng.standard_normal((n_regions, n_factors))
    )
    loadings /= np.linalg.norm(loadings, axis=1, keepdims=True)

    # Factor 0 is a GLOBAL mode that every region loads on, with the same sign.
    # Real widefield cholinergic signal is dominated by one brain-wide
    # component; without it the latent state is not identifiable from the
    # handful of cells left visible at 70% masking, and the model never gets
    # off the ground.
    loadings[:, 0] = (
        global_weight + (1.0 - global_weight) * np.abs(loadings[:, 0])
    )
    loadings /= np.linalg.norm(loadings, axis=1, keepdims=True)

    # 0 to 3 s of delay, quantized to frames. Regions in the same module get
    # similar but not identical delays, which produces the diagonal bands in
    # the lag >= 1 panel.
    module_delay = rng.uniform(0.0, 2.0, size=n_modules)
    delay_seconds = np.clip(
        module_delay[module_of_region] + rng.uniform(0.0, 1.0, size=n_regions),
        0.0, 3.0,
    )
    delay_frames = np.round(delay_seconds * frame_rate).astype(int)

    return {
        "module_of_region": module_of_region,
        "loadings": loadings,
        "delay_frames": delay_frames,
    }


def shared_fraction_for(genotype, age_months, wt_shared, ad_shared_final):
    """
    Fraction of a region's variance that comes from the shared factors.

    WT is flat. AD matches WT at 3 months and decays towards ad_shared_final at
    15 months, so the demo reproduces an age-by-genotype interaction rather
    than a flat group difference.
    """
    if genotype == "wt":
        return wt_shared
    progress = (age_months - 3.0) / 12.0
    return wt_shared + (ad_shared_final - wt_shared) * np.clip(progress, 0.0, 1.0)


def simulate_session(
    n_frames, structure, genotype, age_months, args, rng
):
    """Returns (raw_trace (n_frames, 82) float32, valid_mask (n_frames,) bool)."""

    n_regions = config.N_REGIONS
    max_delay = int(structure["delay_frames"].max())

    factors = ar1_process(
        n_frames + max_delay,
        args.factors,
        timescale_frames=rng.uniform(
            1.5 * args.frame_rate, 6.0 * args.frame_rate, size=args.factors
        ),
        rng=rng,
    )

    shared = np.zeros((n_frames, n_regions), dtype=np.float64)
    for region in range(n_regions):
        offset = max_delay - int(structure["delay_frames"][region])
        shared[:, region] = (
            factors[offset:offset + n_frames] @ structure["loadings"][region]
        )
    shared /= np.maximum(shared.std(axis=0, keepdims=True), 1e-9)

    private = ar1_process(
        n_frames, n_regions,
        timescale_frames=rng.uniform(0.4 * args.frame_rate,
                                     1.2 * args.frame_rate, size=n_regions),
        rng=rng,
    )

    fraction = shared_fraction_for(
        genotype, age_months, args.wt_shared, args.ad_shared_final
    )
    # Region-to-region spread in how "networked" each area is. This is what
    # makes some regions reliably easier to reconstruct than others, which is
    # the per-region R2 result the analysis half of the pipeline reports.
    region_fraction = np.clip(
        fraction + 0.06 * np.sin(np.arange(n_regions) * 0.7), 0.05, 0.97
    )

    region_signal = (
        np.sqrt(region_fraction) * shared
        + np.sqrt(1.0 - region_fraction) * private
    )

    # Split each region into two hemispheres with a little private noise, then
    # interleave into the 82-channel raw layout.
    hemisphere_noise = args.hemisphere_noise * rng.standard_normal(
        (n_frames, n_regions, 2)
    )
    raw = region_signal[:, :, None] + hemisphere_noise
    raw = raw.reshape(n_frames, n_regions * 2)

    # Put it on a dF/F-like scale: small positive baseline, percent units.
    raw = args.baseline + args.scale * raw

    # ---- quality control artifacts ------------------------------------------
    valid = np.ones(n_frames, dtype=bool)

    n_dropouts = rng.integers(1, 4)
    for _ in range(n_dropouts):
        start = rng.integers(0, max(n_frames - 200, 1))
        valid[start:start + rng.integers(20, 200)] = False

    n_nan_runs = rng.integers(0, 3)
    for _ in range(n_nan_runs):
        start = rng.integers(0, max(n_frames - 60, 1))
        raw[start:start + rng.integers(5, 60), :] = np.nan

    return raw.astype(np.float32), valid


# -----------------------------------------------------------------------------
# Parcellation grid, so the brain-map figures work without the real grid.mat
# -----------------------------------------------------------------------------

def make_parcellation_grid(grid_side, rng):
    """
    A cartoon dorsal cortex: an ellipse split into 82 parcels, 41 per
    hemisphere, mirrored left to right and interleaved the same way the raw
    channels are.

    Returns (n_pixels, 82) float32 indicator matrix, matching what
    `visualization/brain_maps.py` reads out of grid.mat.
    """
    y, x = np.mgrid[0:grid_side, 0:grid_side]
    cx = cy = (grid_side - 1) / 2.0
    radius_x, radius_y = grid_side * 0.42, grid_side * 0.46

    inside = ((x - cx) / radius_x) ** 2 + ((y - cy) / radius_y) ** 2 <= 1.0
    right_half = x > cx

    n_regions = config.N_REGIONS
    indicators = np.zeros((grid_side * grid_side, config.RAW_N_REGIONS),
                          dtype=np.float32)

    # Parcels are k-means clusters of the left-half pixels, which guarantees
    # every one of the 41 regions gets a contiguous, non-empty territory. The
    # right hemisphere is the mirror image, so averaging adjacent raw channels
    # recovers the region exactly as it does for the real parcellation.
    from sklearn.cluster import KMeans

    left_pixels = inside & ~right_half
    ly, lx = np.nonzero(left_pixels)
    coordinates = np.column_stack([lx, ly]).astype(np.float64)

    kmeans = KMeans(n_clusters=n_regions, n_init=10, random_state=0)
    assignment = kmeans.fit_predict(coordinates)

    for region in range(n_regions):
        selected = assignment == region
        if not selected.any():
            raise RuntimeError(f"Parcel {region} came out empty.")
        rows, cols = ly[selected], lx[selected]
        indicators[rows * grid_side + cols, 2 * region] = 1.0       # left
        mirrored = (grid_side - 1) - cols
        indicators[rows * grid_side + mirrored, 2 * region + 1] = 1.0  # right

    return indicators


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Write a synthetic ACh dataset in the STAMP input format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", default="synthetic_ach.h5")
    parser.add_argument("--grid-out", default="synthetic_grid.npz",
                        help="parcellation used by the brain-map figures")
    parser.add_argument("--mice-per-genotype", type=int, default=6)
    parser.add_argument("--sessions-per-mouse-age", type=int, default=1)
    parser.add_argument("--frames-per-session", type=int, default=3000,
                        help="3000 frames at 10 Hz = 5 minutes = 20 windows")
    parser.add_argument("--ages", nargs="+", default=config.AGE_ORDER)
    parser.add_argument("--factors", type=int, default=6)
    parser.add_argument("--modules", type=int, default=5)
    parser.add_argument("--wt-shared", type=float, default=0.92,
                        help="shared-variance fraction for WT at every age")
    parser.add_argument("--ad-shared-final", type=float, default=0.62,
                        help="shared-variance fraction for AD at 15 months")
    parser.add_argument("--hemisphere-noise", type=float, default=0.12)
    parser.add_argument("--global-weight", type=float, default=0.75,
                        help="how dominant the brain-wide mode is")
    parser.add_argument("--baseline", type=float, default=0.0)
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--grid-side", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--small", action="store_true",
                        help="tiny file for a smoke test (2 mice, 1200 frames)")
    args = parser.parse_args()

    args.frame_rate = config.FRAME_RATE

    if args.small:
        args.mice_per_genotype = 2
        args.frames_per_session = 1200
        args.ages = ["3mo", "9mo", "15mo"]

    rng = np.random.default_rng(args.seed)
    structure = build_region_structure(
        config.N_REGIONS, args.factors, args.modules, args.frame_rate,
        args.global_weight, rng,
    )

    age_to_months = {age: int(age.replace("mo", "")) for age in args.ages}

    session_keys, ages, genotypes, mice = [], [], [], []
    total_bytes = 0

    with h5py.File(args.out, "w") as h5_file:

        for genotype in ["wt", "mt"]:
            for mouse_number in range(1, args.mice_per_genotype + 1):

                mouse = f"{genotype.upper()}{mouse_number:02d}"

                for age in args.ages:
                    for repeat in range(args.sessions_per_mouse_age):

                        session_key = f"{mouse}_{age}_s{repeat + 1}"

                        raw, valid = simulate_session(
                            args.frames_per_session,
                            structure,
                            genotype,
                            age_to_months[age],
                            args,
                            rng,
                        )

                        group = h5_file.create_group(session_key)
                        group.create_dataset(
                            "ACh", data=raw, compression="gzip",
                            compression_opts=4,
                        )
                        group.create_dataset("ml_valid_frame", data=valid)

                        session_keys.append(session_key)
                        ages.append(age)
                        # Written as 'dki' on purpose: loading.py maps it to
                        # 'mt', and the real files use that spelling too.
                        genotypes.append("dki" if genotype == "mt" else "wt")
                        mice.append(mouse)
                        total_bytes += raw.nbytes

        index = h5_file.create_group("session_index")
        string_type = h5py.string_dtype(encoding="utf-8")
        index.create_dataset("session_key", data=session_keys, dtype=string_type)
        index.create_dataset("age", data=ages, dtype=string_type)
        index.create_dataset("genotype", data=genotypes, dtype=string_type)
        index.create_dataset("mouse", data=mice, dtype=string_type)

        # 82 raw channel names, interleaved left/right, so build_region_labels
        # can recover a shared name per bilateral pair.
        module_of_region = structure["module_of_region"]
        raw_names = []
        for region in range(config.N_REGIONS):
            stem = f"M{module_of_region[region]}_A{region:02d}"
            raw_names.append(f"{stem}_L")
            raw_names.append(f"{stem}_R")
        index.create_dataset("region_name", data=raw_names, dtype=string_type)

        h5_file.attrs["synthetic"] = True
        h5_file.attrs["warning"] = (
            "Simulated data. Not real recordings. Results are meaningless."
        )
        h5_file.attrs["generator"] = "scripts/make_synthetic_data.py"

    indicators = make_parcellation_grid(args.grid_side, np.random.default_rng(1))
    np.savez_compressed(
        args.grid_out,
        indicators=indicators,
        grid_side=np.array(args.grid_side),
        synthetic=np.array(True),
    )

    n_windows_per_session = args.frames_per_session // config.WINDOW_SIZE
    print()
    print(f"Wrote {args.out}")
    print(f"  sessions            {len(session_keys)}")
    print(f"  mice                {2 * args.mice_per_genotype} "
          f"({args.mice_per_genotype} per genotype)")
    print(f"  ages                {', '.join(args.ages)}")
    print(f"  frames per session  {args.frames_per_session} "
          f"({args.frames_per_session / args.frame_rate:.0f} s)")
    print(f"  windows (before QC) ~{len(session_keys) * n_windows_per_session}")
    print(f"  on disk             {os.path.getsize(args.out) / 1e6:.1f} MB")
    print(f"Wrote {args.grid_out}  ({args.grid_side} x {args.grid_side} "
          f"parcellation for the brain maps)")
    print()
    print("These signals are SIMULATED. Use them to run the pipeline, never to "
          "draw a conclusion.")


if __name__ == "__main__":
    main()
