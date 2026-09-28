import copy
import json

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from hybrid_awq.core import SearchConfig
from hybrid_awq.experiment import (apply_variant, benchmark, fit, model_fingerprint,
    save_tensor, storage_bytes, survey, write_json)


def test_qwen3_gqa_norms_swiglu_checkpoint_and_heldout_isolation(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(4)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, max_position_embeddings=64)).eval().requires_grad_(False)
    original_hash = model_fingerprint(model)
    blocks = {'calibration': torch.randint(0, 32, (2, 8)), 'evaluation': torch.randint(0, 32, (1, 8))}
    save_tensor(tmp_path / 'blocks.pt', blocks)
    stats = survey(model, blocks, tmp_path, [6.], search_tokens=8, validation_tokens=4)
    assert all(item['tokens'] == 16 for item in stats.values())  # excludes held-out tokens
    # sqrt(output dimension) < 6 for every projection here, so none can pass.
    result = fit(model, tmp_path, stats, SearchConfig(group_size=8, grid_size=20), [6.])
    assert result['summary']['hybrid-6']['fp16_weight_count'] == 0
    assert result['variants']['hybrid-6'] == result['variants']['awq']
    assert model_fingerprint(model) == original_hash  # fitting must not mutate original
    result['model_fingerprint'] = original_hash
    write_json(tmp_path / 'manifest.json', result)
    fake, packed = copy.deepcopy(model), copy.deepcopy(model)
    apply_variant(fake, tmp_path, 'awq', 'fake')
    apply_variant(packed, tmp_path, 'hybrid-6', 'packed')
    with torch.inference_mode():
        yf = fake(blocks['evaluation']).logits
        yp = packed(blocks['evaluation']).logits
    torch.testing.assert_close(yf, yp)
    assert storage_bytes(packed) < storage_bytes(fake)
    timing = benchmark(packed, tmp_path, 'hybrid-6', prompt_length=4, new_tokens=3, repeats=1, warmups=1)
    assert timing['decode_tokens_per_second'] > 0
    assert timing['peak_allocated_bytes'] is None
    with torch.no_grad():
        model.model.embed_tokens.weight[0, 0] += .1
    with pytest.raises(ValueError, match='Checkpoint weights/dtype changed'):
        apply_variant(model, tmp_path, 'awq')
