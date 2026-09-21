"""
Train / validation split by physical mouse, Dataset and DataLoader construction.

The split is the single most important correctness detail in this repository.
One physical mouse is recorded at several ages and produces many windows. If
the split were made over windows, or even over sessions, the same animal would
appear on both sides and the validation R2 would be meaningless. Everything
here is grouped on `physical_mouse_uid`.
"""

import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import config


class PreloadedWindowDataset(Dataset):
    """Indexes into one big in-RAM float32 array. No disk access per item."""

    def __init__(self, all_windows, global_window_indices):
        self.all_windows = all_windows
        self.global_window_indices = np.asarray(
            global_window_indices, dtype=np.int64
        )

    def __len__(self):
        return len(self.global_window_indices)

    def __getitem__(self, item):
        global_window_id = int(self.global_window_indices[item])
        return (
            torch.from_numpy(self.all_windows[global_window_id]),
            global_window_id,
        )


class SplitBundle:
    """Everything the training loop needs about the split, in one object."""

    def __init__(
        self,
        all_windows,
        window_metadata,
        validation_fraction=config.VALIDATION_FRACTION,
        seed=config.SEED,
    ):
        self.all_windows = all_windows
        self.window_metadata = window_metadata
        self.mouse_uids = window_metadata["physical_mouse_uid"].to_numpy()

        unique_mice = pd.unique(self.mouse_uids)

        splitter = GroupShuffleSplit(
            n_splits=1, test_size=validation_fraction, random_state=seed
        )
        train_positions, validation_positions = next(
            splitter.split(unique_mice, groups=unique_mice)
        )

        self.train_mice = set(unique_mice[train_positions])
        self.validation_mice = set(unique_mice[validation_positions])

        overlap = self.train_mice & self.validation_mice
        if overlap:
            raise RuntimeError(
                f"Physical-mouse leakage between train and validation: {overlap}"
            )

        self.train_indices = np.flatnonzero(
            np.isin(self.mouse_uids, list(self.train_mice))
        )
        self.validation_indices = np.flatnonzero(
            np.isin(self.mouse_uids, list(self.validation_mice))
        )

        self.train_dataset = PreloadedWindowDataset(
            all_windows, self.train_indices
        )
        self.validation_dataset = PreloadedWindowDataset(
            all_windows, self.validation_indices
        )

        # Windows per mouse is very uneven. Sampling with weight 1/count makes
        # every mouse contribute equally in expectation, so the model is not
        # dominated by whichever animal happened to be recorded the longest.
        counts = pd.Series(self.mouse_uids[self.train_indices]).value_counts()
        self.train_sample_weights = np.asarray(
            [1.0 / counts[uid] for uid in self.mouse_uids[self.train_indices]],
            dtype=np.float64,
        )

    # -- loaders ---------------------------------------------------------------

    def build_loaders(self, sampler_seed=config.SEED, num_workers=None):
        """A fresh sampler per run, so every model sees the same sequence."""
        num_workers = (
            config.NUM_WORKERS if num_workers is None else num_workers
        )
        pin_memory = config.get_device().type == "cuda"

        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(
                self.train_sample_weights, dtype=torch.double
            ),
            num_samples=len(self.train_dataset),
            replacement=True,
            generator=torch.Generator().manual_seed(sampler_seed),
        )

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.BATCH_SIZE,
            sampler=train_sampler,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )

        validation_loader = DataLoader(
            self.validation_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )
        return train_loader, validation_loader

    def build_rollout_loader(self):
        """Deterministic loader over the first validation windows."""
        selected = self.validation_indices[:config.ROLLOUT_MAX_WINDOWS]
        return DataLoader(
            PreloadedWindowDataset(self.all_windows, selected),
            batch_size=config.ROLLOUT_BATCH_SIZE,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=config.get_device().type == "cuda",
        )

    def build_group_loader(self, indices):
        """Loader over an arbitrary subset, used for per age x genotype R2."""
        return DataLoader(
            PreloadedWindowDataset(self.all_windows, indices),
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=config.get_device().type == "cuda",
        )

    # -- reporting -------------------------------------------------------------

    def summary(self):
        return pd.DataFrame({
            "split": ["train", "validation"],
            "physical_mouse_count": [
                len(self.train_mice), len(self.validation_mice)
            ],
            "window_count": [
                len(self.train_dataset), len(self.validation_dataset)
            ],
        })

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.summary().to_csv(
            os.path.join(output_dir, "split_summary.csv"), index=False
        )
        with open(os.path.join(output_dir, "split_mice.json"), "w") as handle:
            json.dump(
                {
                    "train_mice": sorted(self.train_mice),
                    "validation_mice": sorted(self.validation_mice),
                },
                handle,
                indent=2,
            )


def close_loader_workers(loader):
    """Colab and SLURM both leak worker processes without this."""
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown):
            shutdown()
