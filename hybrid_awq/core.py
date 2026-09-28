"""INT4 group quantization and the paper's activation-only alpha search.

W is [output, input]; X is [token, input]. No gradients are used.
This is a transparent research implementation, not a fused AWQ inference kernel.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class SearchConfig:
    group_size: int = 128
    grid_size: int = 20
    candidate_pool: int = 8
    max_outlier_fraction: float = 0.001
    min_relative_improvement: float = 1e-4
    loss_chunk: int = 128

    def __post_init__(self):
        if self.group_size < 1 or self.grid_size < 1 or self.candidate_pool < 1 or self.loss_chunk < 1:
            raise ValueError("Group/grid/candidate/chunk sizes must be positive")
        if not 0 <= self.max_outlier_fraction < 1:
            raise ValueError("max_outlier_fraction must be in [0, 1)")
        if not 0 <= self.min_relative_improvement < 1:
            raise ValueError("min_relative_improvement must be in [0, 1)")


def pack_int4(q):
    if q.ndim != 2 or q.shape[1] % 2:
        raise ValueError("Packing requires an even, padded input dimension")
    q = q.to(torch.uint8)
    return q[:, 0::2] | (q[:, 1::2] << 4)


def unpack_int4(packed):
    return torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)


@torch.no_grad()
def quantize_groups(weight, group_size, excluded=None):
    """Zero-inclusive affine INT4; excluded columns do not set group extrema.

    Padding and excluded positions encode *real zero* via the group's zero point.
    Keeps original column positions, avoiding group reordering/alignment changes.
    """
    if weight.ndim != 2 or not torch.isfinite(weight).all():
        raise ValueError("Weight must be a finite 2-D tensor")
    rows, cols = weight.shape
    # Pad also to an even width for nibble packing, including odd group sizes.
    width = math.ceil(cols / math.lcm(group_size, 2)) * math.lcm(group_size, 2)
    w = F.pad(weight.float(), (0, width - cols)).reshape(rows, -1, group_size)
    valid = torch.arange(width, device=weight.device) < cols
    if excluded is not None:
        valid[excluded] = False
    valid = valid.reshape(1, -1, group_size)
    lo = w.masked_fill(~valid, float('inf')).amin(-1).clamp(max=0)
    hi = w.masked_fill(~valid, -float('inf')).amax(-1).clamp(min=0)
    # Empty groups get lo=hi=0 from clamps above.
    delta = ((hi - lo) / 15).clamp_min(1e-8)
    zero = (-lo / delta).round().clamp(0, 15)
    q = (w / delta.unsqueeze(-1) + zero.unsqueeze(-1)).round().clamp(0, 15)
    q = torch.where(valid, q, zero.unsqueeze(-1)).to(torch.uint8)
    recovered = ((q.float() - zero.unsqueeze(-1)) * delta.unsqueeze(-1)).reshape(rows, width)[:, :cols]
    return recovered, q.reshape(rows, width), delta, zero.to(torch.uint8)


def channel_scales(act_mean, alpha, excluded):
    """s=a**alpha; geometric normalization as in the upstream AWQ search.

    Excluded columns neither influence scale normalization nor receive scaling.
    """
    keep = torch.ones_like(act_mean, dtype=torch.bool)
    keep[excluded] = False
    if not keep.any():
        return torch.ones_like(act_mean, dtype=torch.float32)
    raw = act_mean.float().clamp_min(0).pow(alpha).clamp_min(1e-4)
    normalizer = (raw[keep].max() * raw[keep].min()).sqrt()
    scales = raw / normalizer
    scales[excluded] = 1
    return scales


@torch.no_grad()
def effective_weight(weight, act_mean, alpha, excluded, group_size):
    s = channel_scales(act_mean, alpha, excluded)
    low, _, _, _ = quantize_groups(weight.float() * s, group_size, excluded)
    low = low / s
    # Preserve FP16 semantics even if the search itself uses FP32.
    low[:, excluded] = weight[:, excluded].half().float()
    return low, s


@torch.no_grad()
def reconstruction_mse(weight, approx, x, chunk=128):
    """Linear output error, accumulated in FP32 (bias cancels)."""
    if x.shape[0] == 0:
        raise ValueError("No tokens for reconstruction loss")
    error = approx.float() - weight.float()
    total = 0.0
    for part in x.split(chunk):
        residual = F.linear(part.float(), error)
        total += residual.square().sum().item()
    return total / (x.shape[0] * weight.shape[0])


@torch.no_grad()
def search_alpha(weight, x, act_mean, config, excluded=None):
    excluded = torch.as_tensor([] if excluded is None else excluded, device=weight.device, dtype=torch.long)
    best = None
    history = []
    for step in range(config.grid_size):
        alpha = step / config.grid_size  # default 0, .05, ..., .95; upstream AWQ
        approx, scales = effective_weight(weight, act_mean, alpha, excluded, config.group_size)
        loss = reconstruction_mse(weight, approx, x, config.loss_chunk)
        if not math.isfinite(loss):
            raise ValueError("Non-finite calibration loss; inspect inputs and checkpoint dtype")
        history.append({"alpha": alpha, "mse": loss})
        if best is None or loss < best['mse']:
            best = {"alpha": alpha, "mse": loss, "scales": scales.detach().clone()}
    best.update(excluded=excluded.detach().clone(), history=history)
    return best


@torch.no_grad()
def output_hit_counts(y, multiplier):
    """Relative output magnitude, not Dettmers' absolute activation threshold."""
    y = y.float()
    rms = y.square().mean(-1, keepdim=True).sqrt()
    return (y.abs() > multiplier * rms.clamp_min(1e-12)).sum(0)


def persistent_outputs(stats, multipliers, token_fraction=0.06, layer_fraction=0.25):
    """Persistence of OUTPUT indices within the same projection family across blocks.

    Only locally frequent AND globally recurrent output indices are eligible.
    No dimensions from unlike projection families are compared.
    """
    result = {name: {} for name in stats}
    families = {}
    for name, item in stats.items():
        families.setdefault(item['family'], []).append(name)
    for names in families.values():
        for k in multipliers:
            key = str(float(k))
            local = torch.stack([stats[n]['hits'][key].double() / stats[n]['tokens'] >= token_fraction for n in names])
            required = math.ceil(layer_fraction * len(names))
            shared = local.sum(0) >= required
            for row, name in enumerate(names):
                result[name][key] = local[row] & shared
    return result


@torch.no_grad()
def rank_contributing_columns(weight, x, eligible_outputs, multiplier, bias=None, chunk=128):
    """Sum |x_tj w_ij| over flagged tokens/eligible output neurons i.

    Aggregation uses a matrix multiply, never a tokens x outputs x inputs tensor.
    Full original outputs determine RMS even when only a subset is eligible.
    """
    scores = torch.zeros(weight.shape[1], device=weight.device, dtype=torch.float32)
    rows = torch.where(eligible_outputs.to(weight.device))[0]
    if rows.numel() == 0:
        return scores
    abs_w = weight[rows].float().abs()
    for part in x.split(chunk):
        part = part.float()
        y = F.linear(part, weight.float(), None if bias is None else bias.float())
        rms = y.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-12)
        flags = (y[:, rows].abs() > multiplier * rms).float()
        scores += (part.abs() * (flags @ abs_w)).sum(0)
    return scores


@torch.no_grad()
def select_hybrid(weight, x, act_mean, baseline, scores, config):
    """Greedy column exclusion, exact group requantization, then repeat alpha search.

    Candidate evaluations use the current best alpha. After each accepted column
    the complete alpha grid is searched again. No held-out data is used here.
    """
    budget = min(math.floor(config.max_outlier_fraction * weight.shape[1]), config.candidate_pool)
    pool = torch.argsort(scores, descending=True, stable=True)[:config.candidate_pool]
    pool = [int(j) for j in pool if scores[j] > 0]
    current = baseline
    selected = []
    trace = []
    for _ in range(min(budget, len(pool))):
        best_loss, best_j = current['mse'], None
        for j in pool:
            indices = torch.tensor(selected + [j], device=weight.device, dtype=torch.long)
            approx, _ = effective_weight(weight, act_mean, current['alpha'], indices, config.group_size)
            loss = reconstruction_mse(weight, approx, x, config.loss_chunk)
            if loss < best_loss:
                best_loss, best_j = loss, j
        required_gain = config.min_relative_improvement * max(current['mse'], 1e-30)
        if best_j is None or current['mse'] - best_loss <= required_gain:
            break
        selected.append(best_j)
        pool.remove(best_j)
        current = search_alpha(weight, x, act_mean, config, selected)
        trace.append({"column": best_j, "alpha": current['alpha'], "mse": current['mse']})
    result = dict(current)
    result['selection_trace'] = trace
    result['budget'] = budget
    return result


@torch.no_grad()
def make_state(weight, bias, result, config):
    indices = result['excluded'].to(weight.device)
    scales = result['scales'].to(weight.device)
    _, q, delta, zero = quantize_groups(weight.float() * scales, config.group_size, indices)
    high = weight[:, indices].half()
    if not torch.isfinite(high).all():
        raise ValueError("Selected weights are not representable in FP16")
    return {
        'format_version': 1, 'in_features': weight.shape[1], 'out_features': weight.shape[0],
        'group_size': config.group_size, 'packed': pack_int4(q).cpu(),
        'delta': delta.cpu(), 'zero': zero.cpu(), 'scales': scales.cpu(),
        'indices': indices.cpu(), 'high': high.cpu(),
        'bias': None if bias is None else bias.detach().cpu(),
        'alpha': result['alpha'], 'calibration_mse': result['mse'],
    }


class PackedLinear(nn.Module):
    """Actual nibble storage; reference (unfused) dequantize + matmul inference.

    This saves resident weight bytes, but does NOT imply a speedup. It temporarily
    expands one full weight matrix each forward, and launches an FP16 side matmul.
    """
    def __init__(self, state):
        super().__init__()
        self.in_features, self.out_features = state['in_features'], state['out_features']
        self.group_size = state['group_size']
        for name in ['packed', 'delta', 'zero', 'scales', 'indices', 'high', 'bias']:
            self.register_buffer(name, state[name])

    def dequantize_low(self, dtype):
        q = unpack_int4(self.packed).reshape(self.out_features, -1, self.group_size)
        w = (q.float() - self.zero.unsqueeze(-1).float()) * self.delta.unsqueeze(-1)
        w = w.reshape(self.out_features, -1)[:, :self.in_features] / self.scales
        return w.to(dtype)

    def forward(self, x):
        y = F.linear(x, self.dequantize_low(x.dtype), None if self.bias is None else self.bias.to(x.dtype))
        if self.indices.numel():
            # FP16 storage and arithmetic for the preserved branch.
            side = F.linear(x.index_select(-1, self.indices).half(), self.high)
            y = y + side.to(y.dtype)
        return y

    def fake_linear(self, dtype):
        """Float effective weights for accuracy-only simulation; not compressed."""
        w = self.dequantize_low(dtype)
        w[:, self.indices] = self.high.to(dtype)
        result = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None,
                           device=w.device, dtype=dtype)
        result.weight = nn.Parameter(w, requires_grad=False)
        if self.bias is not None:
            result.bias = nn.Parameter(self.bias.to(dtype), requires_grad=False)
        return result
