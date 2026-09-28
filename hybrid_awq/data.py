"""Reproducible, document-disjoint calibration/evaluation token blocks."""
import hashlib
import json
from pathlib import Path

import torch


def read_texts(source, field='text', seed=42, cache_dir=None, offline=False):
    if source is not None:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix.lower() == '.json':
            records = json.loads(path.read_text(encoding='utf-8-sig'))
            if not isinstance(records, list):
                raise ValueError("JSON calibration data must be a list of text strings or objects")
            for row in records:
                yield row if isinstance(row, str) else row[field]
        else:
            with path.open(encoding='utf-8-sig') as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    if path.suffix.lower() == '.jsonl':
                        row = json.loads(line)
                        yield row if isinstance(row, str) else row[field]
                    else:
                        yield line.strip()
        return
    from datasets import DownloadConfig, load_dataset
    dataset = load_dataset('mit-han-lab/pile-val-backup', split='validation',
                           cache_dir=cache_dir,
                           download_config=DownloadConfig(local_files_only=offline))
    for row in dataset.shuffle(seed=seed):
        yield row[field]


def build_blocks(tokenizer, calib_source=None, eval_source=None, *, samples=128,
                 eval_samples=32, length=512, field='text', seed=42,
                 cache_dir=None, offline=False):
    if min(samples, eval_samples) < 1 or length < 2:
        raise ValueError("Need positive sample counts and sequence length >= 2")
    limits = {'calibration': samples * length, 'evaluation': eval_samples * length}
    pools = {key: [] for key in limits}
    seen = set()
    document_counts = {key: 0 for key in limits}
    documents = {key: [] for key in limits}

    def add(text, split):
        if not isinstance(text, str) or not text.strip():
            return
        digest = hashlib.sha256(text.strip().encode()).hexdigest()
        if digest in seen:
            return
        seen.add(digest)
        if len(pools[split]) >= limits[split]:
            return
        ids = tokenizer.encode(text.strip(), add_special_tokens=False)
        if not ids:
            return
        if tokenizer.eos_token_id is not None:
            ids.append(tokenizer.eos_token_id)
        pools[split].extend(ids[:limits[split] - len(pools[split])])
        document_counts[split] += 1
        documents[split].append(digest)

    sources = [(calib_source, 'calibration'), (eval_source, 'evaluation')] if eval_source else [(calib_source, None)]
    for source, fixed_split in sources:
        for number, text in enumerate(read_texts(source, field, seed, cache_dir, offline)):
            split = fixed_split or ('evaluation' if number % 5 == 0 else 'calibration')
            add(text, split)
            if fixed_split:
                if len(pools[split]) == limits[split]:
                    break
            elif all(len(pools[key]) == limits[key] for key in limits):
                break
    for key, limit in limits.items():
        if len(pools[key]) < limit:
            raise ValueError(f"Not enough disjoint {key} text: {len(pools[key])}/{limit} tokens; provide more text or reduce sample counts")
    result = {key: torch.tensor(values, dtype=torch.long).reshape(-1, length) for key, values in pools.items()}
    result['metadata'] = {
        'source': str(calib_source or 'mit-han-lab/pile-val-backup:validation'),
        'eval_source': str(eval_source) if eval_source else 'document-disjoint portion of source',
        'seed': seed, 'length': length, 'documents': document_counts,
        'document_hashes': documents,
        'token_sha256': {key: hashlib.sha256(result[key].numpy().tobytes()).hexdigest() for key in pools},
    }
    return result
