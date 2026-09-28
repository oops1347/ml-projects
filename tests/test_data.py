import json

import pytest
import torch

from hybrid_awq.data import build_blocks


class Tokenizer:
    eos_token_id = 99
    def encode(self, text, add_special_tokens=False):
        return [ord(char) % 90 for char in text]


def test_local_jsonl_disjointness_and_reproducibility(tmp_path):
    path = tmp_path / 'data.jsonl'
    lines = [json.dumps({'text': f'Document {i} with some sufficiently long text'}) for i in range(40)]
    path.write_text('\n'.join(lines + lines), encoding='utf-8')
    first = build_blocks(Tokenizer(), path, samples=4, eval_samples=2, length=16)
    second = build_blocks(Tokenizer(), path, samples=4, eval_samples=2, length=16)
    assert torch.equal(first['calibration'], second['calibration'])
    doc = first['metadata']['document_hashes']
    assert set(doc['calibration']).isdisjoint(doc['evaluation'])
    assert first['calibration'].shape == (4, 16)


def test_identical_train_eval_docs_are_rejected_as_insufficient(tmp_path):
    path = tmp_path / 'text.txt'
    path.write_text('same document with enough tokens to fill both pools\n')
    with pytest.raises(ValueError, match='Not enough disjoint evaluation'):
        build_blocks(Tokenizer(), path, path, samples=1, eval_samples=1, length=8)
