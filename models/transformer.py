"""
The STAMP encoder: a spatio-temporal masked transformer with block-causal
attention.

Shapes, for one batch of B windows:

    input                       (B, 150, 41)     z-scored ACh
    patchify                    (B, 410, 15)     410 = 41 regions x 10 patches
    token_projection            (B, 410, 128)    Linear(15 -> 128), shared
    mask substitution           (B, 410, 128)    287 of 410 replaced at 70%
    + region + time embedding   (B, 410, 128)
    prepend CLS                 (B, 411, 128)
    2 x causal encoder layer    (B, 411, 128)
    split                       (B, 128) CLS  +  (B, 410, 128) tokens
    reconstruction_decoder      (B, 410, 15)     scored on masked cells only
    projection_head (on CLS)    (B, 128)         VICReg between two mask draws

The attention mask is block-causal over the 410 tokens: every region may attend
to every other region within its own 1.5 s patch, plus every earlier patch, and
never to a later one.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from config import (
    ATTENTION_MASK_MODE, CLS_MODE, D_MODEL, DIM_FEEDFORWARD, DROPOUT, HEAD_DIM,
    NHEAD, NUM_LAYERS, N_REGIONS, N_TIME_PATCHES, N_TOKENS, PATCH_LENGTH,
    SEQUENCE_LENGTH, VICREG_COVARIANCE_WEIGHT, VICREG_INVARIANCE_WEIGHT,
    VICREG_VARIANCE_WEIGHT, WINDOW_SIZE,
)

# Token t belongs to time patch t // 41 and region t % 41 (time-major order).
TOKEN_PATCH_INDEX = (torch.arange(N_TOKENS) // N_REGIONS).long()
TOKEN_REGION_INDEX = (torch.arange(N_TOKENS) % N_REGIONS).long()


def build_block_causal_mask():
    """
    (SEQUENCE_LENGTH, SEQUENCE_LENGTH) boolean.
    allow[query, key] is True when the query may attend to the key.

    Giving CLS time index -1 makes the whole thing one comparison: a query may
    read a key when the key is not in the future.
    """
    time_index = torch.cat([
        torch.full((1,), -1, dtype=torch.long),
        TOKEN_PATCH_INDEX,
    ])

    allow = time_index.unsqueeze(1) >= time_index.unsqueeze(0)

    allow[0, :] = True                      # CLS reads every token

    if CLS_MODE == "sink":
        # No token reads CLS. Blocks the two-hop path
        # past token -> CLS -> future token.
        allow[1:, 0] = False

    # Under "bidirectional" the CLS column is left open on purpose. The
    # block-causal structure over the 410 real tokens is identical either way:
    # a past query still cannot attend directly to a future key.
    return allow


def patchify(windows):
    """(B, WINDOW_SIZE, N_REGIONS) -> (B, N_TOKENS, PATCH_LENGTH), time-major."""
    batch_size = windows.shape[0]
    grid = windows.reshape(batch_size, N_TIME_PATCHES, PATCH_LENGTH, N_REGIONS)
    grid = grid.permute(0, 1, 3, 2)
    return grid.reshape(batch_size, N_TOKENS, PATCH_LENGTH)


def tokens_to_region_patch_grid(token_values):
    """(B, N_TOKENS, L) -> (B, N_REGIONS, N_TIME_PATCHES, L)."""
    batch_size = token_values.shape[0]
    grid = token_values.reshape(batch_size, N_TIME_PATCHES, N_REGIONS, -1)
    return grid.permute(0, 2, 1, 3)


class CausalSelfAttention(nn.Module):

    def __init__(self):
        super().__init__()
        self.qkv_projection = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.output_projection = nn.Linear(D_MODEL, D_MODEL)
        self.attention_dropout = nn.Dropout(DROPOUT)
        self.scale = HEAD_DIM ** -0.5

    def forward(self, x, allow_mask, return_attention=False):

        batch_size, seq_len, _ = x.shape

        qkv = (
            self.qkv_projection(x)
            .reshape(batch_size, seq_len, 3, NHEAD, HEAD_DIM)
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv[0], qkv[1], qkv[2]
        allow = allow_mask.reshape(1, 1, seq_len, seq_len)

        # ---- fast path -------------------------------------------------------
        # The fused kernel never materializes the (B, NHEAD, 411, 411) matrix.
        # With a boolean attn_mask, True means "attend" and False is filled with
        # -inf before the softmax, which is exactly what the slow path does.
        # Used whenever the explicit weights are not needed, so during all
        # training. This is worth roughly a 3x speedup.

        if not return_attention and ATTENTION_MASK_MODE == "pre_softmax":
            context = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=allow,
                dropout_p=DROPOUT if self.training else 0.0,
            )
            context = context.transpose(1, 2).reshape(
                batch_size, seq_len, D_MODEL
            )
            return self.output_projection(context), None

        # ---- slow path -------------------------------------------------------
        # Attention rollout needs the explicit weights, and the two
        # post-softmax modes cannot be handed to the fused kernel as a mask.

        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        # Softmax runs in at least float32: float16 and bfloat16 are promoted
        # up, float64 is left alone. Writing .float() here instead silently
        # demotes float64 and costs about 5 orders of magnitude of agreement
        # with the fused path.
        softmax_dtype = torch.promote_types(scores.dtype, torch.float32)

        if ATTENTION_MASK_MODE == "pre_softmax":
            scores = scores.masked_fill(~allow, float("-inf"))
            attention = torch.softmax(scores.to(softmax_dtype), dim=-1)
        else:
            attention = torch.softmax(scores.to(softmax_dtype), dim=-1)
            attention = attention * allow.to(softmax_dtype)
            if ATTENTION_MASK_MODE == "post_softmax_renorm":
                attention = attention / attention.sum(
                    -1, keepdim=True
                ).clamp_min(1e-12)

        attention = attention.to(value.dtype)
        returned = attention if return_attention else None

        context = torch.matmul(self.attention_dropout(attention), value)
        context = context.transpose(1, 2).reshape(batch_size, seq_len, D_MODEL)
        return self.output_projection(context), returned


class CausalEncoderLayer(nn.Module):
    """Pre-norm, same structure as nn.TransformerEncoderLayer(norm_first=True)."""

    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.attention = CausalSelfAttention()
        self.dropout1 = nn.Dropout(DROPOUT)
        self.norm2 = nn.LayerNorm(D_MODEL)
        self.feed_forward = nn.Sequential(
            nn.Linear(D_MODEL, DIM_FEEDFORWARD),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(DIM_FEEDFORWARD, D_MODEL),
        )
        self.dropout2 = nn.Dropout(DROPOUT)

    def forward(self, x, allow_mask, return_attention=False):
        attention_output, weights = self.attention(
            self.norm1(x), allow_mask, return_attention
        )
        x = x + self.dropout1(attention_output)
        x = x + self.dropout2(self.feed_forward(self.norm2(x)))
        return x, weights


class STAMPEncoder(nn.Module):
    """243,343 parameters at the default configuration."""

    def __init__(self):
        super().__init__()

        self.token_projection = nn.Linear(PATCH_LENGTH, D_MODEL)

        self.cls_token = nn.Parameter(torch.randn(1, 1, D_MODEL) * 0.02)
        self.cls_position_embedding = nn.Parameter(
            torch.randn(1, 1, D_MODEL) * 0.02
        )
        self.mask_token = nn.Parameter(torch.randn(1, 1, D_MODEL) * 0.02)

        self.region_embedding = nn.Parameter(
            torch.randn(N_REGIONS, D_MODEL) * 0.02
        )
        self.time_embedding = nn.Parameter(
            torch.randn(N_TIME_PATCHES, D_MODEL) * 0.02
        )

        self.layers = nn.ModuleList(
            [CausalEncoderLayer() for _ in range(NUM_LAYERS)]
        )

        self.output_norm = nn.LayerNorm(D_MODEL)
        self.reconstruction_decoder = nn.Linear(D_MODEL, PATCH_LENGTH)

        self.projection_head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.GELU(),
            nn.Linear(D_MODEL, D_MODEL),
        )

        self.register_buffer(
            "allow_mask", build_block_causal_mask(), persistent=False
        )
        self.register_buffer(
            "token_region_index", TOKEN_REGION_INDEX.clone(), persistent=False
        )
        self.register_buffer(
            "token_patch_index", TOKEN_PATCH_INDEX.clone(), persistent=False
        )

    def forward(self, x, token_mask=None, return_attention=False):

        batch_size = x.shape[0]

        tokens = self.token_projection(patchify(x))

        if token_mask is not None:
            tokens = torch.where(
                token_mask.unsqueeze(-1),
                self.mask_token.expand(batch_size, N_TOKENS, D_MODEL),
                tokens,
            )

        # Position is added AFTER the mask swap, so a masked token still knows
        # which region and which moment it stands for. Adding it before would
        # make every masked token identical.
        tokens = (
            tokens
            + self.region_embedding[self.token_region_index].unsqueeze(0)
            + self.time_embedding[self.token_patch_index].unsqueeze(0)
        )

        cls_token = (
            self.cls_token.expand(batch_size, -1, -1)
            + self.cls_position_embedding
        )
        hidden = torch.cat([cls_token, tokens], dim=1)

        attention_maps = []
        for layer in self.layers:
            hidden, weights = layer(hidden, self.allow_mask, return_attention)
            if return_attention:
                attention_maps.append(weights)

        hidden = self.output_norm(hidden)

        cls_embedding = hidden[:, 0, :]
        token_output = hidden[:, 1:, :]

        return (
            self.reconstruction_decoder(token_output),
            cls_embedding,
            self.projection_head(cls_embedding),
            attention_maps,
        )


# -----------------------------------------------------------------------------
# Masking
# -----------------------------------------------------------------------------

def create_spatiotemporal_mask(batch_size, generator, n_masked_tokens, device):
    """
    Masks exactly n_masked_tokens of the 410 (region, time patch) cells, drawn
    freely across both axes. A region can be hidden at some time patches and
    visible at others, which is what forces the model to use both its spatial
    and its temporal neighbours rather than interpolating one region in time.
    """
    scores = torch.rand((batch_size, N_TOKENS), device=device, generator=generator)
    indices = torch.topk(scores, k=n_masked_tokens, dim=1, largest=False).indices

    token_mask = torch.zeros(
        (batch_size, N_TOKENS), dtype=torch.bool, device=device
    )
    token_mask.scatter_(1, indices, True)
    return token_mask


# -----------------------------------------------------------------------------
# VICReg
# -----------------------------------------------------------------------------

def off_diagonal(matrix):
    n = matrix.shape[0]
    return matrix.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def compute_vicreg_loss(projected_view_1, projected_view_2):
    """
    Invariance pulls the two mask views of the same window together, variance
    keeps each embedding dimension from collapsing to a constant, covariance
    decorrelates the dimensions.
    """
    invariance_loss = F.mse_loss(projected_view_1, projected_view_2)

    std_1 = torch.sqrt(projected_view_1.var(dim=0, unbiased=False) + 1e-4)
    std_2 = torch.sqrt(projected_view_2.var(dim=0, unbiased=False) + 1e-4)
    variance_loss = 0.5 * (
        F.relu(1.0 - std_1).mean() + F.relu(1.0 - std_2).mean()
    )

    centered_1 = projected_view_1 - projected_view_1.mean(dim=0, keepdim=True)
    centered_2 = projected_view_2 - projected_view_2.mean(dim=0, keepdim=True)
    denominator = max(projected_view_1.shape[0] - 1, 1)

    covariance_1 = (centered_1.T @ centered_1) / denominator
    covariance_2 = (centered_2.T @ centered_2) / denominator
    covariance_loss = (
        off_diagonal(covariance_1).pow(2).sum()
        + off_diagonal(covariance_2).pow(2).sum()
    ) / D_MODEL

    total = (
        VICREG_INVARIANCE_WEIGHT * invariance_loss
        + VICREG_VARIANCE_WEIGHT * variance_loss
        + VICREG_COVARIANCE_WEIGHT * covariance_loss
    )
    return {
        "total": total,
        "invariance": invariance_loss,
        "variance": variance_loss,
        "covariance": covariance_loss,
    }


def create_gradient_scaler(device):
    try:
        return torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=device.type == "cuda")


# -----------------------------------------------------------------------------
# Self-checks
# -----------------------------------------------------------------------------

def verify_model(device=None, log=print):
    """
    Structural checks on the mask, agreement between the fused and explicit
    attention paths, and the two measurements that justify CLS_MODE.

    Called by `train.py` before training starts. Raises on anything that would
    silently corrupt the results.
    """
    from evaluation.leak import (
        measure_causal_leak, measure_cls_reconstruction_gradient,
    )

    device = config.get_device() if device is None else device
    allow = build_block_causal_mask()

    q3 = 1 + 3 * N_REGIONS + 7          # a query in patch 3
    k5 = 1 + 5 * N_REGIONS + 2          # a key   in patch 5
    s3 = 1 + 3 * N_REGIONS + 11         # another token in patch 3

    assert allow[0].all(), "CLS must read every token."
    assert not allow[q3, k5], "A past query must not read a future key."
    assert allow[k5, q3], "A future query must read a past key."
    assert allow[q3, s3], "The same time patch must be fully connected."
    assert allow.any(dim=1).all(), "Every query must have at least one key."

    if CLS_MODE == "sink":
        assert not allow[1:, 0].any(), "sink requires a closed CLS column."
    else:
        assert allow[1:, 0].all(), "bidirectional requires an open CLS column."

    model = STAMPEncoder().to(device).eval()

    with torch.no_grad():
        x = torch.randn(4, WINDOW_SIZE, N_REGIONS, device=device)
        fast, _, _, _ = model(x, None, return_attention=False)
        slow, _, _, _ = model(x, None, return_attention=True)
        scale = max(slow.abs().max().item(), 1e-12)
        relative_gap = (fast - slow).abs().max().item() / scale

    leak = measure_causal_leak(model, device=device)
    cls_gradient = measure_cls_reconstruction_gradient(device=device)

    parameter_count = sum(p.numel() for p in model.parameters())

    log("")
    log(f"Model parameters                  {parameter_count:,}")
    log(f"Allowed attention fraction        "
        f"{allow.float().mean().item():.4f}")
    log(f"Fused vs explicit attention       {relative_gap:.3e} relative")
    log(f"CLS gradient from reconstruction  {cls_gradient:.3e}")
    log(f"Causal leak probe                 "
        f"future {leak['future_shift']:.3e}, past {leak['past_shift']:.3e} "
        f"({leak['leak_ratio']:.3%})")

    assert leak["future_shift"] > 1e-3, (
        "The probe perturbation never reached the future tokens, so its "
        "'no leak' answer would mean nothing."
    )

    if CLS_MODE == "sink":
        assert leak["past_shift"] < 1e-5, "The fused path leaks the future."
        assert cls_gradient == 0.0, (
            "Under sink, cls_token must receive no gradient from the MSE."
        )
        log("Strict causality verified: the past is untouched by the future.")
    else:
        assert cls_gradient > 0.0, (
            "Under bidirectional, cls_token must receive gradient from the "
            "MSE. If this fails the CLS column was not actually opened."
        )
        log("CLS is bidirectional: cls_token trains on the reconstruction loss.")
        log("The non-zero past shift is the accepted cost and is re-measured "
            "on the trained checkpoint.")

    del model
    return {
        "parameter_count": parameter_count,
        "allowed_fraction": float(allow.float().mean().item()),
        "fused_vs_explicit": relative_gap,
        "cls_gradient_from_mse": cls_gradient,
        **{f"init_{k}": v for k, v in leak.items()},
    }
