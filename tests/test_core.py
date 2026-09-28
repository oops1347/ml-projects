import math

import pytest
import torch
from torch.nn import functional as F

from hybrid_awq.core import (PackedLinear, SearchConfig, channel_scales,
    effective_weight, make_state, output_hit_counts, pack_int4, persistent_outputs,
    quantize_groups, rank_contributing_columns, reconstruction_mse,
    search_alpha, select_hybrid, unpack_int4)


def test_nibbles_all_values_and_odd_group_padding():
    q = torch.arange(16, dtype=torch.uint8).reshape(2, 8)
    assert torch.equal(unpack_int4(pack_int4(q)), q)
    w = torch.tensor([[1., 2., -3., 4., 5.]])
    recovered, q, _, _ = quantize_groups(w, 3)
    assert q.shape == (1, 6)
    assert recovered.shape == w.shape
    assert torch.equal(unpack_int4(pack_int4(q)), q)


def test_exclusion_shrinks_group_step_and_encodes_real_zero():
    w = torch.tensor([[-1., 100., 1., .4], [2., 50., 3., 2.5]])
    _, _, before, _ = quantize_groups(w, 4)
    recovered, q, after, z = quantize_groups(w, 4, torch.tensor([1]))
    assert (after < before).all()
    assert torch.equal(q[:, 1], z[:, 0])
    assert torch.equal(recovered[:, 1], torch.zeros(2))
    assert q[0, 1].item() != 0  # integer zero != real zero for this group


def test_empty_and_constant_groups_are_finite():
    for value in [0., 2., -2.]:
        w = torch.full((3, 4), value)
        recovered, _, _, _ = quantize_groups(w, 2, torch.arange(4))
        assert torch.equal(recovered, torch.zeros_like(w))
        recovered, _, _, _ = quantize_groups(w, 2)
        torch.testing.assert_close(recovered, w)


def test_scaling_identity_and_excluded_channel_ignored_in_normalizer():
    a = torch.tensor([1., 4., 1e20, 9.])
    excluded = torch.tensor([2])
    s = channel_scales(a, .5, excluded)
    assert s[2] == 1
    torch.testing.assert_close(s[[0, 1, 3]], torch.tensor([1., 2., 3.]) / math.sqrt(3))
    x, w = torch.randn(7, 4), torch.randn(5, 4)
    torch.testing.assert_close(F.linear(x / s, w * s), F.linear(x, w))


def test_side_path_restores_column_without_double_counting():
    w = torch.tensor([[-1., 80., 1., .4], [2., 40., 3., 2.5]])
    config = SearchConfig(group_size=4)
    x = torch.eye(4).half()
    result = search_alpha(w, x.float(), torch.ones(4), config, [1])
    state = make_state(w, torch.tensor([.5, -.5]), result, config)
    packed = PackedLinear(state)
    y = packed(x)
    # Isolates only the preserved channel, independently of the low-bit quantizer.
    torch.testing.assert_close(y[1], w[:, 1].half() + torch.tensor([.5, -.5]).half())
    torch.testing.assert_close(y, packed.fake_linear(torch.float16)(x), atol=.1, rtol=.001)
    assert packed.packed.dtype == torch.uint8
    assert packed.packed.numel() == w.numel() // 2
    assert packed.high.dtype == torch.float16


def test_reconstruction_loss_uses_signed_difference():
    w = torch.tensor([[2., -3.]])
    approx = -w
    x = torch.eye(2)
    assert reconstruction_mse(w, approx, x) == pytest.approx((16 + 36) / 2)


def test_output_event_threshold_is_relative_and_sign_invariant():
    y = torch.zeros(10, 64)
    y[:, 2] = -100
    counts = output_hit_counts(y, 6)
    assert counts[2] == 10 and counts.sum() == 10
    assert torch.equal(counts, output_hit_counts(y * .01, 6))


def test_both_frequency_conditions_and_family_separation():
    stats = {}
    for family in ['q_proj', 'up_proj']:
        for layer in range(8):
            # Channel 0 occurs in two layers (25%); channel 1 in only one.
            hits = torch.tensor([6 if layer < 2 and family == 'q_proj' else 0, 7 if layer == 0 else 0, 5])
            stats[f'{family}.{layer}'] = {'family': family, 'tokens': 100, 'hits': {'6.0': hits}}
    masks = persistent_outputs(stats, [6.])
    assert masks['q_proj.0']['6.0'].tolist() == [True, False, False]
    assert not masks['q_proj.2']['6.0'].any()
    assert not masks['up_proj.0']['6.0'].any()


def test_attribution_only_counts_large_eligible_outputs():
    w = torch.zeros(64, 4)
    w[0, 2] = 50
    w[1, 3] = 1
    x = torch.ones(10, 4)
    eligible = torch.zeros(64, dtype=torch.bool)
    eligible[0] = True
    scores = rank_contributing_columns(w, x, eligible, 6)
    assert scores.argmax() == 2 and scores[2] == 500
    assert scores.sum() == 500
    eligible.zero_()
    assert not rank_contributing_columns(w, x, eligible, 6).any()


def test_grid_search_and_hybrid_monotonicity_with_actual_improvement():
    torch.manual_seed(12)
    w = torch.randn(12, 16) * .1
    w[:, 3] *= 100
    x = torch.randn(64, 16)
    a = x.abs().mean(0)
    cfg = SearchConfig(group_size=8, candidate_pool=2, max_outlier_fraction=.125)
    baseline = search_alpha(w, x, a, cfg)
    assert [row['alpha'] for row in baseline['history']] == [i / 20 for i in range(20)]
    assert baseline['mse'] == min(row['mse'] for row in baseline['history'])
    scores = torch.zeros(16)
    scores[3] = 100
    hybrid = select_hybrid(w, x, a, baseline, scores, cfg)
    assert hybrid['excluded'].tolist() == [3]
    assert hybrid['mse'] < baseline['mse']
    high, _ = effective_weight(w, a, hybrid['alpha'], hybrid['excluded'], 8)
    torch.testing.assert_close(high[:, 3], w[:, 3].half().float(), atol=0, rtol=0)


def test_no_candidates_and_sub_one_column_budget_do_not_force_outliers():
    w, x = torch.randn(4, 8), torch.randn(12, 8)
    cfg = SearchConfig(group_size=4, max_outlier_fraction=.0001)
    a = x.abs().mean(0)
    base = search_alpha(w, x, a, cfg)
    result = select_hybrid(w, x, a, base, torch.ones(8), cfg)
    assert result['mse'] == base['mse'] and result['excluded'].numel() == 0
