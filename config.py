"""
Every configurable constant for STAMP lives here.

Nothing in this file depends on the data, so it can be imported from any
script. Values that a user is likely to change from the command line
(mask ratio, epochs, data path, output directory) are exposed as argparse
flags in `train.py` and `analyze.py` and override what is set here.
"""

import numpy as np
import torch

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

SEED = 42

# -----------------------------------------------------------------------------
# Recording geometry
# -----------------------------------------------------------------------------
# The widefield parcellation gives 82 raw channels: 41 anatomical areas, each
# present in both hemispheres, interleaved as
#     (left_0, right_0, left_1, right_1, ...)
# Adjacent channels (1,2), (3,4) ... (81,82) in one-based indexing are averaged
# into a single bilateral region, leaving 41 regions.

RAW_N_REGIONS = 82
HEMISPHERE_AVERAGE = True
N_REGIONS = RAW_N_REGIONS // 2 if HEMISPHERE_AVERAGE else RAW_N_REGIONS

FRAME_RATE = 10.0                      # Hz

AGE_ORDER = ["3mo", "6mo", "9mo", "12mo", "15mo"]
GENOTYPES = ["wt", "mt"]               # mt = AD model (DKI)
GENOTYPE_LABELS = {"wt": "WT", "mt": "AD (DKI)"}

# -----------------------------------------------------------------------------
# Windowing
# -----------------------------------------------------------------------------
# Continuous recordings are cut into non-overlapping 15 s windows. Each window
# is z-scored per region, so a window has mean 0 and std 1 along time for every
# region independently. This makes an R2 of 0 exactly equal to "predict the
# window mean" and puts every region on the same scale.

WINDOW_SIZE = 150                      # frames  = 15.0 s
STRIDE = 150                           # frames  -> non-overlapping
STD_FLOOR = 1e-6

# -----------------------------------------------------------------------------
# Spatio-temporal tokenization
# -----------------------------------------------------------------------------
# A window becomes an (N_REGIONS x N_TIME_PATCHES) grid. Every cell of that grid
# is one token holding PATCH_LENGTH frames of one region.
#
# Token order is TIME-MAJOR:
#     token_index = patch_index * N_REGIONS + region_index
#
# so tokens 0..40 are all 41 regions at t0, tokens 41..81 are all 41 regions at
# t1, and so on. This ordering is what makes the block-causal mask a clean
# staircase of 41x41 blocks.
#
# The CLS token sits at sequence index 0 and is given time index -1.

PATCH_LENGTH = 15                      # frames = 1.5 s

assert WINDOW_SIZE % PATCH_LENGTH == 0, "WINDOW_SIZE must divide by PATCH_LENGTH"

N_TIME_PATCHES = WINDOW_SIZE // PATCH_LENGTH        # 10
N_TOKENS = N_REGIONS * N_TIME_PATCHES               # 410
SEQUENCE_LENGTH = N_TOKENS + 1                      # 411 with CLS

# -----------------------------------------------------------------------------
# Attention
# -----------------------------------------------------------------------------
#   "pre_softmax"          fill disallowed scores with -inf, then softmax.
#                          Standard, no future leakage, numerically stable,
#                          and the only mode that can use the fused kernel.
#   "post_softmax_renorm"  softmax, multiply by the mask, divide by the row sum.
#                          Mathematically identical to "pre_softmax".
#   "post_softmax_raw"     softmax, multiply by the mask, no renormalization.
#                          Future content leaks into the output scale. Kept only
#                          so the failure mode can be demonstrated.

ATTENTION_MASK_MODE = "pre_softmax"

# -----------------------------------------------------------------------------
# CLS connectivity
# -----------------------------------------------------------------------------
# "sink"           CLS reads every token, no token reads CLS. Strictly causal.
#                  Cost: the reconstruction decoder reads token positions only,
#                  and under "sink" those never depend on CLS, so cls_token
#                  receives EXACTLY ZERO gradient from the reconstruction MSE.
#                  It would learn from the VICReg term alone, at weight 0.05.
#
# "bidirectional"  CLS reads every token AND every token reads CLS. cls_token
#                  now also receives gradient from the reconstruction MSE, which
#                  is the reason this is the default: the CLS embedding is what
#                  gets extracted for downstream analysis and it needs a real
#                  training signal.
#                  Cost: with NUM_LAYERS >= 2 this opens the two-hop path
#                      past token -> CLS -> future token
#                  so the model is no longer strictly causal. Direct attention
#                  from a past query to a future key is still blocked; the leak
#                  is indirect and has to squeeze through one shared 128-d
#                  vector. `evaluation.leak.measure_causal_leak` quantifies it
#                  on every trained checkpoint so the compromise is measured
#                  rather than assumed.

CLS_MODE = "bidirectional"

# Probe settings for the causal-leak measurement.
CAUSAL_LEAK_FIRST_FUTURE_PATCH = 6
CAUSAL_LEAK_PERTURBATION = 5.0
CAUSAL_LEAK_PROBE_WINDOWS = 8

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 128
DROPOUT = 0.20

assert D_MODEL % NHEAD == 0, "D_MODEL must divide by NHEAD"
HEAD_DIM = D_MODEL // NHEAD

# VICReg on the projected CLS embedding, computed between two independent mask
# draws of the same window. Keeps the CLS representation from collapsing.
VICREG_WEIGHT = 0.05
VICREG_INVARIANCE_WEIGHT = 25.0
VICREG_VARIANCE_WEIGHT = 25.0
VICREG_COVARIANCE_WEIGHT = 1.0

# -----------------------------------------------------------------------------
# Masking
# -----------------------------------------------------------------------------
# Masking is applied independently to every (region, time patch) cell, free
# across both axes: a region can be hidden at some moments and visible at
# others. 0.70 is the headline setting and everything downstream is reported
# at that ratio.

MASK_RATIO = 0.70

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

SSL_EPOCHS = 60
BATCH_SIZE = 128
NUM_WORKERS = 4

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-3
GRADIENT_CLIP_NORM = 1.0

# The split is by PHYSICAL MOUSE, never by window or by session. A mouse
# recorded at 3, 6 and 9 months is one mouse: putting two of its sessions on
# opposite sides of the split would leak.
VALIDATION_FRACTION = 0.20

# -----------------------------------------------------------------------------
# Attention rollout
# -----------------------------------------------------------------------------
# Heads are collapsed with a MAX inside each window, then averaged across
# windows. Max keeps whatever the strongest of the four heads found for each
# (query, key) pair; averaging the heads first would dilute a pattern that only
# one head carries. Reducing per window before averaging is the right order for
# the same reason.

ROLLOUT_HEAD_REDUCTION = "max"
ROLLOUT_BATCH_SIZE = 32
ROLLOUT_MAX_WINDOWS = 2048
ROLLOUT_RESIDUAL_WEIGHT = 0.5

# Rollout is measured on UNMASKED windows, so it reflects learned connectivity
# rather than the accident of one mask draw.
ROLLOUT_USE_MASKING = False


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reset_all_seeds(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def n_masked_tokens(mask_ratio=MASK_RATIO):
    return max(1, int(round(mask_ratio * N_TOKENS)))


def describe():
    """One-screen summary of the configuration, printed by both entry points."""
    lines = [
        f"Signal                ACh, {FRAME_RATE:.0f} Hz",
        f"Regions               {RAW_N_REGIONS} raw -> {N_REGIONS} bilateral",
        f"Window                {WINDOW_SIZE} frames = "
        f"{WINDOW_SIZE / FRAME_RATE:.1f} s, stride {STRIDE}",
        f"Time patch            {PATCH_LENGTH} frames = "
        f"{PATCH_LENGTH / FRAME_RATE:.1f} s",
        f"Tokens                {N_REGIONS} x {N_TIME_PATCHES} = {N_TOKENS} "
        f"(+CLS = {SEQUENCE_LENGTH})",
        f"Mask ratio            {MASK_RATIO:.0%} "
        f"({n_masked_tokens()}/{N_TOKENS} cells)",
        f"Model                 d_model {D_MODEL}, {NHEAD} heads, "
        f"{NUM_LAYERS} layers, dropout {DROPOUT}",
        f"Attention             block-causal, mask applied {ATTENTION_MASK_MODE}",
        f"CLS connectivity      {CLS_MODE}",
        f"Rollout               heads collapsed with {ROLLOUT_HEAD_REDUCTION}, "
        f"residual {ROLLOUT_RESIDUAL_WEIGHT}",
    ]
    return "\n".join(lines)
