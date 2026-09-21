"""
Reading the HDF5 file, hemisphere averaging, windowing, normalization.

The expected file layout is documented in the README under "Data Format" and
produced exactly by `scripts/make_synthetic_data.py`.

    <file>.h5
    ├── session_index/
    │   ├── session_key : (S,) string   name of the group holding each session
    │   ├── age         : (S,) string   '3mo' | '6mo' | '9mo' | '12mo' | '15mo'
    │   ├── genotype    : (S,) string   'wt' | 'mt' (or 'dki', mapped to 'mt')
    │   ├── mouse       : (S,) string   physical mouse ID
    │   └── region_name : (82,) string  optional, raw channel names
    └── <session_key>/                  one group per session
        ├── ACh            : (T, 82) float   raw traces, either orientation
        └── ml_valid_frame : (T,) bool       optional QC mask
"""

import json
import os
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd

import config


# -----------------------------------------------------------------------------
# Small HDF5 helpers
# -----------------------------------------------------------------------------

def decode_value(value):
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value)


def decode_array(array):
    return [decode_value(value) for value in array]


def find_dataset_case_insensitive(group, candidates):
    candidate_set = {c.lower() for c in candidates}
    for name, node in group.items():
        if isinstance(node, h5py.Dataset) and name.lower() in candidate_set:
            return name
    return None


def find_ach_dataset(session_group):
    """Locate the (T, 82) cholinergic trace inside one session group."""
    direct = find_dataset_case_insensitive(
        session_group, ["ach", "green", "ach_trace", "green_trace"]
    )
    if direct is not None:
        node = session_group[direct]
        if node.ndim == 2 and config.RAW_N_REGIONS in node.shape:
            return direct

    possible = [
        name
        for name, node in session_group.items()
        if isinstance(node, h5py.Dataset)
        and node.ndim == 2
        and config.RAW_N_REGIONS in node.shape
        and ("ach" in name.lower() or "green" in name.lower())
    ]
    if len(possible) == 1:
        return possible[0]

    raise KeyError(
        f"Cannot uniquely locate the ACh trace in {session_group.name}. "
        f"Candidates={possible}. Expected a 2-D dataset with "
        f"{config.RAW_N_REGIONS} on one axis, named 'ACh' or containing "
        f"'ach'/'green'."
    )


def find_valid_dataset(session_group):
    return find_dataset_case_insensitive(
        session_group,
        ["ml_valid_frame", "valid_frame", "valid_frames", "frame_valid"],
    )


def find_raw_region_labels(h5_file):
    """Best-effort lookup of the 82 raw channel names. None if absent."""
    candidates = [
        "region_name", "region_names", "regionname",
        "roi_name", "roi_names",
        "region_label", "region_labels",
        "area_name", "area_names", "labels",
    ]
    groups = [h5_file]
    if "session_index" in h5_file:
        groups.insert(0, h5_file["session_index"])

    for group in groups:
        name = find_dataset_case_insensitive(group, candidates)
        if name is None:
            continue
        values = decode_array(np.asarray(group[name][:]).ravel())
        if len(values) == config.RAW_N_REGIONS:
            return values
    return None


def build_region_labels(raw_labels):
    """One label per bilateral pair, from the shared prefix of left and right."""
    if not config.HEMISPHERE_AVERAGE:
        return (
            list(raw_labels) if raw_labels is not None
            else [f"ROI_{i:02d}" for i in range(config.RAW_N_REGIONS)]
        )
    if raw_labels is None:
        return [f"ROI_{i:02d}" for i in range(config.N_REGIONS)]

    labels = []
    for index in range(config.N_REGIONS):
        left = str(raw_labels[2 * index]).strip()
        right = str(raw_labels[2 * index + 1]).strip()
        shared = os.path.commonprefix([left, right]).rstrip(" _-.|")
        labels.append(shared if len(shared) >= 2 else f"{left}|{right}")
    return labels


# -----------------------------------------------------------------------------
# Trace handling
# -----------------------------------------------------------------------------

def orient_ach_trace(trace):
    trace = np.asarray(trace)
    if trace.ndim == 2 and trace.shape[1] == config.RAW_N_REGIONS:
        return trace
    if trace.ndim == 2 and trace.shape[0] == config.RAW_N_REGIONS:
        return trace.T
    raise ValueError(f"Unexpected ACh shape: {trace.shape}")


def apply_hemisphere_average(trace):
    """(T, 82) -> (T, 41) by averaging pairs (1,2)(3,4)...(81,82)."""
    trace = np.asarray(trace, dtype=np.float32)
    if not config.HEMISPHERE_AVERAGE:
        return trace
    n_frames = trace.shape[0]
    return np.asarray(
        trace.reshape(n_frames, config.N_REGIONS, 2).mean(axis=2),
        dtype=np.float32,
    )


def load_session_trace(session_group, ach_dataset_name):
    return apply_hemisphere_average(
        orient_ach_trace(session_group[ach_dataset_name][:])
    )


def read_valid_frame_mask(session_group, valid_dataset_name, n_frames):
    if valid_dataset_name is None:
        return np.ones(n_frames, dtype=bool)

    mask = np.asarray(session_group[valid_dataset_name][:]).squeeze()

    if mask.ndim == 2:
        if mask.shape[0] == n_frames:
            mask = np.all(mask > 0, axis=1)
        elif mask.shape[1] == n_frames:
            mask = np.all(mask > 0, axis=0)
        else:
            raise ValueError(
                f"Valid-mask shape {mask.shape} does not match {n_frames} frames."
            )
    if mask.ndim != 1 or len(mask) != n_frames:
        raise ValueError(f"Invalid valid-mask shape: {mask.shape}")
    return mask.astype(bool)


def normalize_window_per_roi(window):
    """Z-score each region independently within the window."""
    window = np.asarray(window, dtype=np.float32)
    mean = window.mean(axis=0, keepdims=True)
    std = np.maximum(window.std(axis=0, keepdims=True), config.STD_FLOOR)
    return np.asarray((window - mean) / std, dtype=np.float32)


# -----------------------------------------------------------------------------
# Session index
# -----------------------------------------------------------------------------

def read_session_metadata(h5_path, log=print):
    """Returns (session_metadata DataFrame, REGION_LABELS list)."""

    with h5py.File(h5_path, "r") as h5_file:

        if "session_index" not in h5_file:
            raise KeyError(
                "session_index group is missing. See the Data Format section "
                "of the README, or run scripts/make_synthetic_data.py for a "
                "correctly shaped example file."
            )

        index_group = h5_file["session_index"]
        missing = [
            field for field in ["session_key", "age", "genotype", "mouse"]
            if field not in index_group
        ]
        if missing:
            raise KeyError(f"session_index is missing fields: {missing}")

        raw_region_labels = find_raw_region_labels(h5_file)

        raw_metadata = pd.DataFrame({
            "session_key": decode_array(index_group["session_key"][:]),
            "age": decode_array(index_group["age"][:]),
            "genotype": decode_array(index_group["genotype"][:]),
            "mouse": decode_array(index_group["mouse"][:]),
        })

        records = []
        for _, row in raw_metadata.iterrows():

            session_key = str(row["session_key"])
            age = str(row["age"]).lower().strip()
            genotype = str(row["genotype"]).lower().strip()
            if genotype == "dki":
                genotype = "mt"
            mouse = str(row["mouse"]).strip()

            if age not in config.AGE_ORDER or genotype not in config.GENOTYPES:
                continue
            if session_key not in h5_file:
                continue

            session_group = h5_file[session_key]
            try:
                ach_dataset = find_ach_dataset(session_group)
            except KeyError:
                continue

            records.append({
                "session_key": session_key,
                "age": age,
                "genotype": genotype,
                "mouse": mouse,
                "ach_dataset": ach_dataset,
                "valid_dataset": find_valid_dataset(session_group),
            })

    session_metadata = pd.DataFrame(records).reset_index(drop=True)
    if len(session_metadata) == 0:
        raise RuntimeError("No usable ACh sessions found in the file.")

    session_metadata["session_id"] = np.arange(
        len(session_metadata), dtype=np.int64
    )
    # A mouse is identified by genotype + name. One physical mouse can appear at
    # several ages; every one of its sessions must stay on the same side of the
    # train/validation split.
    session_metadata["physical_mouse_uid"] = (
        session_metadata["genotype"] + "/" + session_metadata["mouse"]
    )
    session_metadata["mouse_age_uid"] = (
        session_metadata["physical_mouse_uid"] + "/" + session_metadata["age"]
    )

    region_labels = build_region_labels(raw_region_labels)
    if len(region_labels) != config.N_REGIONS:
        raise RuntimeError(
            f"Got {len(region_labels)} region labels, expected "
            f"{config.N_REGIONS}."
        )

    log(f"Usable sessions : {len(session_metadata)}")
    log(f"Physical mice   : {session_metadata['physical_mouse_uid'].nunique()}")
    log(f"Named regions   : {raw_region_labels is not None}")

    return session_metadata, region_labels


# -----------------------------------------------------------------------------
# Windows
# -----------------------------------------------------------------------------

def build_windows(h5_path, session_metadata, log=print):
    """
    Two passes over the file.

    Pass 1 indexes every window whose frames are all valid and finite.
    Pass 2 normalizes those windows and stacks them into one float32 array.

    Returns (all_windows, window_metadata).
      all_windows     (n_windows, WINDOW_SIZE, N_REGIONS) float32
      window_metadata DataFrame, one row per window
    """

    window_records = []
    rejected = 0

    with h5py.File(h5_path, "r") as h5_file:

        for row_number, row in session_metadata.iterrows():

            session_group = h5_file[row["session_key"]]
            trace = load_session_trace(session_group, row["ach_dataset"])
            n_frames = int(trace.shape[0])

            valid = read_valid_frame_mask(
                session_group, row["valid_dataset"], n_frames
            )
            # A non-finite value in either hemisphere propagates through the
            # average, so finiteness is tested after averaging.
            finite = np.isfinite(trace).all(axis=1)
            usable = valid & finite

            invalid_prefix = np.concatenate([
                np.array([0], dtype=np.int64),
                np.cumsum((~usable).astype(np.int64)),
            ])

            for start in range(
                0, n_frames - config.WINDOW_SIZE + 1, config.STRIDE
            ):
                bad = (
                    invalid_prefix[start + config.WINDOW_SIZE]
                    - invalid_prefix[start]
                )
                if bad == 0:
                    window_records.append(
                        (int(row["session_id"]), int(start))
                    )
                else:
                    rejected += 1

    if len(window_records) == 0:
        raise RuntimeError("No complete valid windows found.")

    window_index = np.asarray(window_records, dtype=np.int64)
    del window_records

    log(f"Valid windows   : {len(window_index)}  (rejected {rejected})")

    estimated_gb = (
        len(window_index) * config.WINDOW_SIZE * config.N_REGIONS * 4 / 1e9
    )
    log(f"Preloading      : about {estimated_gb:.2f} GB of float32")

    all_windows = np.empty(
        (len(window_index), config.WINDOW_SIZE, config.N_REGIONS),
        dtype=np.float32,
    )

    positions_by_session = defaultdict(list)
    for window_id, (session_id, start) in enumerate(window_index):
        positions_by_session[int(session_id)].append((int(window_id), int(start)))

    with h5py.File(h5_path, "r") as h5_file:
        for session_id in sorted(positions_by_session.keys()):
            row = session_metadata.iloc[session_id]
            trace = load_session_trace(
                h5_file[row["session_key"]], row["ach_dataset"]
            )
            for window_id, start in positions_by_session[session_id]:
                all_windows[window_id] = normalize_window_per_roi(
                    trace[start:start + config.WINDOW_SIZE, :]
                )

    window_metadata = pd.DataFrame({
        "window_id": np.arange(len(window_index), dtype=np.int64),
        "session_id": window_index[:, 0],
        "start_frame": window_index[:, 1],
    }).merge(
        session_metadata[[
            "session_id", "session_key", "physical_mouse_uid",
            "mouse_age_uid", "mouse", "age", "genotype",
        ]],
        on="session_id", how="left", validate="many_to_one",
    )

    if window_metadata.isna().any().any():
        raise RuntimeError("window_metadata contains missing values.")

    return all_windows, window_metadata


def load_dataset(h5_path, output_dir=None, log=print):
    """
    Convenience wrapper: metadata + windows in one call, with the summary
    tables written to `output_dir` if given.
    """
    session_metadata, region_labels = read_session_metadata(h5_path, log=log)
    all_windows, window_metadata = build_windows(
        h5_path, session_metadata, log=log
    )

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

        session_metadata.to_csv(
            os.path.join(output_dir, "session_metadata.csv"), index=False
        )
        with open(os.path.join(output_dir, "region_labels.json"), "w") as handle:
            json.dump(list(region_labels), handle, indent=2)

        window_metadata.groupby(
            ["age", "genotype"], observed=True
        ).agg(
            window_count=("window_id", "count"),
            mouse_count=("physical_mouse_uid", "nunique"),
        ).reset_index().to_csv(
            os.path.join(output_dir, "window_counts_by_age_genotype.csv"),
            index=False,
        )

    log(f"all_windows     : {all_windows.shape} {all_windows.dtype}")
    return all_windows, window_metadata, region_labels
