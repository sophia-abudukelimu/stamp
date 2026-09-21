"""
Painting a per-region value onto the cortical surface.

The parcellation comes from an (n_pixels x 82) indicator matrix: column c is 1
at every pixel belonging to raw channel c. Model region r covers raw channels
2r and 2r+1, the bilateral pair that was averaged together, so the two
hemispheres of a region are painted with the same value.

Two sources are supported:

    grid.mat          the real MATLAB parcellation (scipy.io or h5py)
    synthetic_grid.npz  written by scripts/make_synthetic_data.py

Both end up in the same `Parcellation` object, so every figure below works
identically on real and synthetic data.
"""

import numpy as np

import config


def _collect(obj, found, depth=0):
    """Walk an arbitrarily nested MATLAB struct and collect numeric arrays."""
    if depth > 8:
        return
    if isinstance(obj, np.ndarray):
        if obj.dtype.names:
            for name in obj.dtype.names:
                for item in obj.ravel():
                    _collect(item[name], found, depth + 1)
        elif obj.dtype == object:
            for item in obj.ravel():
                _collect(item, found, depth + 1)
        else:
            found.append(np.asarray(obj))
        return
    if hasattr(obj, "_fieldnames"):
        for name in obj._fieldnames:
            _collect(getattr(obj, name), found, depth + 1)


def _load_grid_arrays(path):
    found = []
    try:
        import scipy.io as sio
        contents = sio.loadmat(path, squeeze_me=False, struct_as_record=False)
        for key, value in contents.items():
            if not key.startswith("__"):
                _collect(value, found)
    except NotImplementedError:
        import h5py
        with h5py.File(path, "r") as handle:
            handle.visititems(
                lambda name, node: found.append(np.asarray(node[()]).T)
                if isinstance(node, h5py.Dataset) and node.dtype.kind in "fiu"
                else None
            )

    indicators = None
    for array in found:
        squeezed = np.squeeze(array)
        if squeezed.ndim == 2 and config.RAW_N_REGIONS in squeezed.shape:
            if squeezed.shape[0] == config.RAW_N_REGIONS:
                squeezed = squeezed.T
            if indicators is None or squeezed.size > indicators.size:
                indicators = squeezed.astype(np.float32)

    if indicators is None:
        raise RuntimeError(
            f"No (n_pixels x {config.RAW_N_REGIONS}) indicator array found in "
            f"{path}."
        )
    return indicators


class Parcellation:
    """Region pixel masks, brain outline, and the painting helpers."""

    def __init__(self, indicators, reshape_order="F",
                 flip_vertical=False, flip_horizontal=False):

        self.indicators = indicators
        self.reshape_order = reshape_order
        self.flip_vertical = flip_vertical
        self.flip_horizontal = flip_horizontal

        n_pixels = indicators.shape[0]
        side = int(round(np.sqrt(n_pixels)))
        if side * side != n_pixels:
            raise RuntimeError(f"{n_pixels} pixels is not a square image.")
        self.grid_side = side

        self.region_pixel_masks = np.stack([
            self.to_image(
                (
                    (indicators[:, 2 * r] > 0.5)
                    | (indicators[:, 2 * r + 1] > 0.5)
                ).astype(np.float32)
            ) > 0.5
            for r in range(config.N_REGIONS)
        ])

        self.brain_mask = self.region_pixel_masks.any(axis=0)

        label_image = np.zeros(self.brain_mask.shape, dtype=np.float32)
        for r in range(config.N_REGIONS):
            label_image[self.region_pixel_masks[r]] = r + 1

        boundary = np.zeros_like(label_image, dtype=bool)
        boundary[:-1, :] |= label_image[:-1, :] != label_image[1:, :]
        boundary[:, :-1] |= label_image[:, :-1] != label_image[:, 1:]
        self.boundary = boundary & self.brain_mask

    # -- constructors ----------------------------------------------------------

    @classmethod
    def from_file(cls, path, **kwargs):
        if str(path).endswith(".npz"):
            with np.load(path, allow_pickle=True) as handle:
                indicators = handle["indicators"]
            # The synthetic grid is written row-major already.
            kwargs.setdefault("reshape_order", "C")
        else:
            indicators = _load_grid_arrays(path)
        return cls(indicators, **kwargs)

    # -- painting --------------------------------------------------------------

    def to_image(self, flat):
        image = np.asarray(flat, dtype=np.float32).reshape(
            self.grid_side, self.grid_side, order=self.reshape_order
        )
        if self.flip_vertical:
            image = image[::-1, :]
        if self.flip_horizontal:
            image = image[:, ::-1]
        return image

    def paint(self, values):
        """(41,) -> a 2-D image, NaN outside the brain."""
        image = np.full(self.brain_mask.shape, np.nan, dtype=np.float64)
        for r in range(config.N_REGIONS):
            image[self.region_pixel_masks[r]] = values[r]
        return image

    def draw(self, axis, values, title, cmap, norm, fontsize=11):
        handle = axis.imshow(
            self.paint(values), cmap=cmap, norm=norm, interpolation="nearest"
        )
        overlay = np.zeros(self.boundary.shape + (4,), dtype=np.float32)
        overlay[self.boundary] = [0.15, 0.15, 0.15, 0.55]
        axis.imshow(overlay, interpolation="nearest")
        axis.set_title(title, fontsize=fontsize)
        axis.axis("off")
        return handle

    def region_pixel_counts(self):
        return self.region_pixel_masks.sum(axis=(1, 2))
