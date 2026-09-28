"""Main entry point: calibrate, compare accuracy, then benchmark hybrid AWQ.

Run `python train.py --help`. Model downloads are disabled by default so your
existing Hugging Face cache is used. Calibration datasets can be replaced.
"""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from hybrid_awq.core import SearchConfig
from hybrid_awq.data import build_blocks
from hybrid_awq.experiment import (apply_variant, benchmark, evaluate, fit, load_model,
                                   load_tensor, model_fingerprint, save_tensor, survey,
                                   write_json)


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    def runtime(cmd):
        cmd.add_argument('--model', default='Qwen/Qwen3-14B-Base', help='HF ID or local fine-tuned checkpoint directory')
        cmd.add_argument('--cache-dir', default=None, help='Model cache directory; otherwise honors HF_HOME')
        cmd.add_argument('--device', default='cuda:0')
        cmd.add_argument('--dtype', choices=['float16', 'bfloat16', 'float32'], default='float16')
        cmd.add_argument('--allow-model-download', action='store_true')
        cmd.add_argument('--offline', action='store_true', help='Disable HF network access, including datasets')
        cmd.add_argument('--cpu-threads', type=int, default=4)
    cal = sub.add_parser('calibrate', help='Survey original outputs; fit AWQ and hybrid variants')
    runtime(cal)
    cal.add_argument('--output', required=True, help='New experiment directory (never overwrite a completed run)')
    cal.add_argument('--calib-data', help='TXT (one document per line), JSONL, or JSON list; default: Pile validation')
    cal.add_argument('--eval-data', help='Optional separate held-out text file')
    cal.add_argument('--text-field', default='text')
    cal.add_argument('--dataset-cache-dir')
    cal.add_argument('--calib-samples', type=int, default=128)
    cal.add_argument('--eval-samples', type=int, default=32)
    cal.add_argument('--seq-len', type=int, default=512)
    cal.add_argument('--search-tokens', type=int, default=512, help='Uniformly sampled calibration input rows per projection for fitting')
    cal.add_argument('--validation-tokens', type=int, default=256)
    cal.add_argument('--seed', type=int, default=42)
    cal.add_argument('--multipliers', nargs='+', type=float, default=[4., 6., 8.])
    cal.add_argument('--token-fraction', type=float, default=.06)
    cal.add_argument('--layer-fraction', type=float, default=.25)
    cal.add_argument('--group-size', type=int, default=128)
    cal.add_argument('--alpha-grid', type=int, default=20)
    cal.add_argument('--candidate-pool', type=int, default=8)
    cal.add_argument('--max-outlier-fraction', type=float, default=.001,
                     help='Per-projection upper bound, NOT a quota; floor(fraction * input_channels)')
    cal.add_argument('--min-relative-improvement', type=float, default=1e-4)
    cal.add_argument('--loss-chunk', type=int, default=128)
    resume = sub.add_parser('fit', help='Restart fitting from a completed survey after interruption')
    runtime(resume)
    resume.add_argument('--artifact', required=True)
    for name in ['evaluate', 'compare', 'benchmark', 'benchmark-all']:
        cmd = sub.add_parser(name)
        runtime(cmd)
        cmd.add_argument('--artifact', required=True)
        cmd.add_argument('--backend', choices=['fake', 'packed'], default='packed' if 'benchmark' in name else 'fake')
        if name in ['evaluate', 'benchmark']:
            cmd.add_argument('--variant', default='hybrid-6', help='original, awq, or hybrid-<multiplier> present in manifest')
        if name in ['evaluate', 'compare']:
            cmd.add_argument('--logit-tokens', type=int, default=32)
        else:
            cmd.add_argument('--prompt-length', type=int, default=128)
            cmd.add_argument('--new-tokens', type=int, default=32)
            cmd.add_argument('--repeats', type=int, default=5)
            cmd.add_argument('--warmups', type=int, default=2)
    smoke = sub.add_parser('smoke', help='Offline, small Qwen3 integration test; NOT a 14B result')
    smoke.add_argument('--output', required=True)
    return p


def versions():
    return {name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'datasets', 'accelerate']}


def runtime_model(args):
    return load_model(args.model, device=args.device, dtype=args.dtype, cache_dir=args.cache_dir,
                      allow_download=args.allow_model_download)


def complete_fit(model, output, metadata):
    stats = load_tensor(output / 'survey.pt')
    result = fit(model, output, stats, SearchConfig(**metadata['search_config']), metadata['multipliers'],
                 metadata['token_fraction'], metadata['layer_fraction'])
    result.update(metadata)
    write_json(output / 'manifest.json', result)
    print(json.dumps(result['summary'], indent=2), flush=True)


def calibrate(args):
    if min(args.search_tokens, args.validation_tokens) < 1:
        raise ValueError('Search and validation token counts must be positive')
    if not 0 < args.token_fraction <= 1 or not 0 < args.layer_fraction <= 1:
        raise ValueError('Token/layer fractions must be in (0, 1]')
    if not args.multipliers or any(k <= 0 or not torch.isfinite(torch.tensor(k)) for k in args.multipliers):
        raise ValueError('Output multipliers must be finite and positive')
    args.multipliers = sorted(set(args.multipliers))
    config = SearchConfig(args.group_size, args.alpha_grid, args.candidate_pool,
                          args.max_outlier_fraction, args.min_relative_improvement, args.loss_chunk)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    model, tokenizer = runtime_model(args)
    print('Hashing original checkpoint to prevent reuse after fine-tuning...', flush=True)
    fingerprint = model_fingerprint(model)
    blocks = build_blocks(tokenizer, args.calib_data, args.eval_data, samples=args.calib_samples,
                          eval_samples=args.eval_samples, length=args.seq_len, field=args.text_field,
                          seed=args.seed, cache_dir=args.dataset_cache_dir, offline=args.offline)
    save_tensor(output / 'blocks.pt', blocks)
    metadata = {
        'model': args.model, 'model_fingerprint': fingerprint, 'dtype': args.dtype,
        'search_config': asdict(config), 'multipliers': args.multipliers,
        'token_fraction': args.token_fraction, 'layer_fraction': args.layer_fraction,
        'seed': args.seed, 'search_tokens': args.search_tokens,
        'validation_tokens': args.validation_tokens, 'data': blocks['metadata'],
        'versions': versions(), 'model_config': model.config.to_dict(),
        'baseline': 'Paper-style AWQ per-linear output-MSE scale search, INT4 g128 by default; no clipping search or cross-module scale fusion',
    }
    write_json(output / 'run.json', metadata)
    survey(model, blocks, output, args.multipliers, args.search_tokens, args.validation_tokens, args.seed)
    write_json(output / 'survey-complete.json', {'model_fingerprint': fingerprint})
    complete_fit(model, output, metadata)


def run_suite(args):
    directory = Path(args.artifact)
    manifest = json.loads((directory / 'manifest.json').read_text())
    task = 'evaluate' if args.command == 'compare' else 'benchmark'
    results = []
    for variant in ['original', *manifest['variants']]:
        command = [sys.executable, str(Path(__file__).resolve()), task,
                   '--artifact', str(directory.resolve()), '--variant', variant,
                   '--backend', args.backend, '--model', args.model, '--device', args.device,
                   '--dtype', args.dtype, '--cpu-threads', str(args.cpu_threads)]
        if args.cache_dir:
            command += ['--cache-dir', args.cache_dir]
        for flag in ['offline', 'allow_model_download']:
            if getattr(args, flag):
                command += ['--' + flag.replace('_', '-')]
        fields = ['logit_tokens'] if task == 'evaluate' else ['prompt_length', 'new_tokens', 'repeats', 'warmups']
        for field in fields:
            command += ['--' + field.replace('_', '-'), str(getattr(args, field))]
        print('Starting isolated process:', variant, task, flush=True)
        subprocess.run(command, check=True)
        prefix = 'eval' if task == 'evaluate' else 'benchmark'
        results.append(json.loads((directory / f'{prefix}-{variant}-{args.backend}.json').read_text()))
    write_json(directory / f'{args.command}-{args.backend}.json', results)
    print(json.dumps(results, indent=2))


def smoke_test(output):
    """Exercise real Qwen3 GQA, Q/K norms, RoPE, SwiGLU, serialization, and logits."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    import copy
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                                       num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                       head_dim=8, max_position_embeddings=128,
                                       attention_dropout=0., tie_word_embeddings=False)).eval().requires_grad_(False)
    with torch.no_grad():
        for layer in model.model.layers:
            layer.self_attn.q_proj.weight[0] *= 30
            layer.mlp.up_proj.weight[0] *= 30
    blocks = {'calibration': torch.randint(0, 64, (4, 16)), 'evaluation': torch.randint(0, 64, (2, 16))}
    save_tensor(output / 'blocks.pt', blocks)
    stats = survey(model, blocks, output, [2.], 32, 16)
    result = fit(model, output, stats, SearchConfig(group_size=8, grid_size=20, candidate_pool=2,
                                                  max_outlier_fraction=.04), [2.], .06, .25)
    result['model_fingerprint'] = model_fingerprint(model)
    write_json(output / 'manifest.json', result)
    evaluations = [evaluate(model, output, 'original', logit_tokens=4)]
    for backend in ['fake', 'packed']:
        for variant in result['variants']:
            candidate = copy.deepcopy(model)
            apply_variant(candidate, output, variant, backend)
            evaluations.append(evaluate(candidate, output, variant, backend, logit_tokens=4))
            del candidate
    write_json(output / 'smoke-results.json', evaluations)
    print(json.dumps(evaluations, indent=2))


def main():
    args = parser().parse_args()
    if args.command == 'smoke':
        smoke_test(args.output)
        return
    if args.cpu_threads < 1:
        raise ValueError('cpu-threads must be positive')
    torch.set_num_threads(args.cpu_threads)
    # Stable error comparisons; no TF32 or dropout during the experiment.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.offline:
        if args.allow_model_download:
            raise ValueError('--offline and --allow-model-download conflict')
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'
    if args.command == 'calibrate':
        calibrate(args)
    elif args.command in ['compare', 'benchmark-all']:
        run_suite(args)
    elif args.command == 'fit':
        directory = Path(args.artifact)
        if (directory / 'manifest.json').exists():
            raise ValueError('This experiment is complete; recalibrate into a new directory')
        if not (directory / 'survey-complete.json').exists():
            raise ValueError('No complete survey to reuse; run calibrate into a new directory')
        metadata = json.loads((directory / 'run.json').read_text())
        model, _ = runtime_model(args)
        if model_fingerprint(model) != metadata['model_fingerprint']:
            raise ValueError('Checkpoint/dtype differs from survey; recalibrate')
        complete_fit(model, directory, metadata)
    else:
        if args.command == 'evaluate' and args.logit_tokens < 1:
            raise ValueError('logit-tokens must be positive')
        if args.command == 'benchmark' and (min(args.prompt_length, args.repeats) < 1 or args.new_tokens < 2 or args.warmups < 1):
            raise ValueError('Need positive prompt/repeats/warmups and at least 2 new tokens')
        model, _ = runtime_model(args)
        apply_variant(model, args.artifact, args.variant, args.backend)
        if args.command == 'evaluate':
            result = evaluate(model, args.artifact, args.variant, args.backend, args.logit_tokens)
        else:
            result = benchmark(model, args.artifact, args.variant, args.backend, args.prompt_length,
                               args.new_tokens, args.repeats, args.warmups)
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
