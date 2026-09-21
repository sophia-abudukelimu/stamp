"""
Colormaps shared by every figure.

The R2 map is the one that matters. Two constraints drove its design:

  1. White sits exactly at R2 = 0, which is the "predict the window mean"
     baseline. Anything blue is a region the model does WORSE than that
     baseline on, and the eye picks those out instantly.
  2. The colour stops are packed between 0.70 and 1.00, because that is where
     nearly all the regions land and a linear colormap would render the whole
     result as one flat shade.
"""

import matplotlib.colors as mcolors
import numpy as np

R2_VMIN, R2_VMAX = -0.30, 1.00

R2_COLOR_STOPS = [
    (-0.300, "#2B5E8C"), (-0.150, "#7FA8C9"), (-0.050, "#D7E4EE"),
    ( 0.000, "#FFFFFF"),
    ( 0.200, "#FDF4E4"), ( 0.450, "#FAE2B4"), ( 0.700, "#F4BE6A"),
    ( 0.775, "#EC9748"), ( 0.850, "#DB6A32"), ( 0.925, "#B93B29"),
    ( 1.000, "#72102E"),
]

R2_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "r2_white_zero",
    [((v - R2_VMIN) / (R2_VMAX - R2_VMIN), c) for v, c in R2_COLOR_STOPS],
    N=512,
)
R2_CMAP.set_bad("#EFEFEF")
R2_NORM = mcolors.Normalize(vmin=R2_VMIN, vmax=R2_VMAX)

# Symmetric around white for differences (age to age, AD minus WT).
DIFF_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "diff_white_zero",
    ["#16436B", "#5C93BE", "#BBD5E6", "#FFFFFF",
     "#F3CBB4", "#DC7A4E", "#94301A"],
    N=512,
)
DIFF_CMAP.set_bad("#EFEFEF")

ROLLOUT_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "rollout_white_zero",
    ["#FFFFFF", "#E7EEF4", "#B9CFE0", "#7FA6C6",
     "#4E7BA8", "#2E5580", "#17304F"],
    N=512,
)
ROLLOUT_CMAP.set_bad("#EFEFEF")


def symmetric_norm(values, cap=None):
    """A Normalize centred on zero, scaled to the data."""
    finite = np.asarray(values)[np.isfinite(values)]
    limit = float(np.abs(finite).max()) if finite.size else 1.0
    if cap is not None:
        limit = min(limit, cap)
    limit = max(limit, 1e-6)
    return mcolors.Normalize(vmin=-limit, vmax=limit)
