"""Qwen3 calibration, fitting and honest accuracy/performance measurement."""
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import time

import torch
from torch import nn
from torch.nn import functional as F

from .core import (PackedLinear, make_state, output_hit_counts, persistent_outputs,
                   rank_contributing_columns, reconstruction_mse, search_alpha,
                   select_hybrid, effective_weight)

FAMILIES = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
            'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def save_tensor(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, temporary)
    temporary.replace(path)


def load_tensor(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def projections(model):
    if model.config.model_type != 'qwen3':
        raise ValueError('This adapter targets dense Qwen3 (including Qwen3-14B-Base), not MoE or Qwen3.5')
    found = {}
    for block, layer in enumerate(model.model.layers):
        for family in FAMILIES:
            module = layer.get_submodule(family)
            if not isinstance(module, nn.Linear):
                raise ValueError(f'Expected original floating nn.Linear at layer {block}: {family}')
            found[f'model.layers.{block}.{family}'] = (module, family, block)
    return found


def model_fingerprint(model):
    """Reject stale masks/scales after ANY checkpoint weight changes."""
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(f'{name}:{tuple(tensor.shape)}:{tensor.dtype}'.encode())
        raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
        h.update(memoryview(raw).cast('B'))
    return h.hexdigest()


def storage_bytes(model):
    # Count tied storages once (not Python module counts).
    storages = {}
    for tensor in list(model.parameters()) + list(model.buffers()):
        storage = tensor.untyped_storage()
        storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


def load_model(model_name, *, device='cuda:0', dtype='float16', cache_dir=None,
               allow_download=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    target = torch.device(device)
    if target.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Use --device cpu for a small test model; run 14B on your GPU host.')
    common = dict(cache_dir=cache_dir, local_files_only=not allow_download)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **common)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=getattr(torch, dtype),
                                               device_map={'': str(target)}, attn_implementation='eager', **common)
    model.eval().requires_grad_(False)
    projections(model)  # Validate before any costly calibration.
    return model, tokenizer


def _sample_positions(blocks, count, seed):
    generator = torch.Generator().manual_seed(seed)
    total = blocks.numel()
    choices = torch.randperm(total, generator=generator)[:min(total, count)].sort().values
    return [choices[(choices // blocks.shape[1]) == i] % blocks.shape[1] for i in range(len(blocks))]


@torch.inference_mode()
def survey(model, blocks, directory, multipliers=(4., 6., 8.), search_tokens=512,
           validation_tokens=256, seed=42):
    """Count output events over ALL calibration tokens; cache bounded input samples.

    Sequences are fixed length with no padding. Held-out inputs are never included
    in activation means, event frequencies, attribution, or alpha selection.
    """
    directory = Path(directory)
    modules = projections(model)
    stats, inputs = {}, {name: {'calibration': [], 'evaluation': []} for name in modules}
    for name, (module, family, block) in modules.items():
        stats[name] = {'family': family, 'block': block, 'tokens': 0,
                       'act_sum': torch.zeros(module.in_features, dtype=torch.float64),
                       'hits': {str(float(k)): torch.zeros(module.out_features, dtype=torch.long) for k in multipliers}}
    context = {}
    def hook_for(name):
        def hook(module, args, output):
            x = args[0].detach().reshape(-1, module.in_features)
            y = output.detach().reshape(-1, module.out_features)
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                raise ValueError(f'Non-finite activations in {name}; change checkpoint dtype or inspect checkpoint')
            split = context['split']
            if split == 'calibration':
                stats[name]['tokens'] += x.shape[0]
                stats[name]['act_sum'] += x.float().abs().sum(0).double().cpu()
                for k in multipliers:
                    stats[name]['hits'][str(float(k))] += output_hit_counts(y, k).cpu()
            positions = context['positions']
            if positions.numel():
                inputs[name][split].append(x.index_select(0, positions.to(x.device)).cpu())
        return hook
    handles = [module.register_forward_hook(hook_for(name)) for name, (module, _, _) in modules.items()]
    device = next(model.parameters()).device
    try:
        for split, count in [('calibration', search_tokens), ('evaluation', validation_tokens)]:
            positions = _sample_positions(blocks[split], count, seed + (split == 'evaluation'))
            for i, ids in enumerate(blocks[split]):
                context.update(split=split, positions=positions[i])
                # Base model avoids materializing the enormous vocabulary logits.
                model.model(input_ids=ids.unsqueeze(0).to(device), use_cache=False)
                if i % 8 == 0:
                    print(f'survey {split}: {i + 1}/{len(positions)}', flush=True)
    finally:
        for handle in handles:
            handle.remove()
    for name, item in stats.items():
        item['act_mean'] = (item.pop('act_sum') / item['tokens']).float()
        save_tensor(directory / 'samples' / f'{name}.pt', {key: torch.cat(value) for key, value in inputs[name].items()})
        inputs[name].clear()
    save_tensor(directory / 'survey.pt', stats)
    return stats


def _result_summary(result):
    return {key: result[key] for key in ['alpha', 'mse', 'history', 'selection_trace', 'budget'] if key in result} | {
        'outlier_columns': result['excluded'].cpu().tolist()
    }


@torch.inference_mode()
def fit(model, directory, stats, config, multipliers=(4., 6., 8.),
        token_fraction=.06, layer_fraction=.25):
    directory = Path(directory)
    eligible = persistent_outputs(stats, multipliers, token_fraction, layer_fraction)
    manifest = {'format_version': 1, 'variants': {'awq': {}}, 'layers': {}}
    for k in multipliers:
        manifest['variants'][f'hybrid-{k:g}'] = {}
    for number, (name, (module, _, _)) in enumerate(projections(model).items()):
        print(f'fit {number + 1}/{len(stats)}: {name}', flush=True)
        device = module.weight.device
        weight = module.weight.detach().float()
        samples = load_tensor(directory / 'samples' / f'{name}.pt')
        x, validation = [samples[key].to(device).float() for key in ['calibration', 'evaluation']]
        act = stats[name]['act_mean'].to(device)
        baseline = search_alpha(weight, x, act, config)
        baseline_state = make_state(weight, module.bias, baseline, config)
        baseline_path = f'awq/{name}.pt'
        save_tensor(directory / baseline_path, baseline_state)
        manifest['variants']['awq'][name] = baseline_path
        approx, _ = effective_weight(weight, act, baseline['alpha'], baseline['excluded'], config.group_size)
        info = {'shape': list(weight.shape), 'validation_elements': validation.shape[0] * weight.shape[0],
                'awq': _result_summary(baseline) | {'validation_mse': reconstruction_mse(weight, approx, validation, config.loss_chunk)}}
        del approx
        for k in multipliers:
            key, variant = str(float(k)), f'hybrid-{k:g}'
            scores = rank_contributing_columns(weight, x, eligible[name][key], k, module.bias, config.loss_chunk)
            result = select_hybrid(weight, x, act, baseline, scores, config)
            if result['excluded'].numel() == 0:
                path = baseline_path  # exact baseline fallback; no redundant artifact copies
            else:
                path = f'{variant}/{name}.pt'
                save_tensor(directory / path, make_state(weight, module.bias, result, config))
            manifest['variants'][variant][name] = path
            approx, _ = effective_weight(weight, act, result['alpha'], result['excluded'], config.group_size)
            info[variant] = _result_summary(result) | {
                'eligible_output_count': int(eligible[name][key].sum()),
                'locally_frequent_output_count': int((stats[name]['hits'][key].double() / stats[name]['tokens'] >= token_fraction).sum()),
                'validation_mse': reconstruction_mse(weight, approx, validation, config.loss_chunk),
            }
            del approx
        manifest['layers'][name] = info
        # A partial manifest is deliberately not loadable as a finished experiment.
        write_json(directory / 'fit-progress.json', manifest)
        del weight, x, validation, samples, baseline_state
    totals = {}
    for variant in manifest['variants']:
        denominator = sum(item['validation_elements'] for item in manifest['layers'].values())
        totals[variant] = {
            'projection_validation_mse': sum(item[variant]['validation_mse'] * item['validation_elements'] for item in manifest['layers'].values()) / denominator,
            'fp16_weight_count': sum(len(item[variant]['outlier_columns']) * item['shape'][0] for item in manifest['layers'].values()),
        }
    manifest['summary'] = totals
    return manifest


def apply_variant(model, directory, variant, backend='fake'):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if model_fingerprint(model) != manifest['model_fingerprint']:
        raise ValueError('Checkpoint weights/dtype changed. Recalibrate into a new directory instead of reusing these masks/scales.')
    if variant == 'original':
        return
    entries = manifest['variants'][variant]
    expected = projections(model)
    if set(entries) != set(expected):
        raise ValueError('Artifact does not cover exactly all target projections')
    for name, relative in entries.items():
        original = expected[name][0]
        packed = PackedLinear(load_tensor(directory / relative)).to(original.weight.device)
        if (packed.out_features, packed.in_features) != tuple(original.weight.shape):
            raise ValueError(f'Artifact shape mismatch: {name}')
        replacement = packed.fake_linear(original.weight.dtype) if backend == 'fake' else packed
        parent, leaf = name.rsplit('.', 1)
        setattr(model.get_submodule(parent), leaf, replacement)
    del expected  # release old full-precision projection references
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.inference_mode()
def evaluate(model, directory, variant, backend='fake', logit_tokens=32):
    """Full held-out NLL plus bounded sampled-logit comparison with the original."""
    directory = Path(directory)
    blocks = load_tensor(directory / 'blocks.pt')['evaluation']
    ref_dir = directory / 'reference'
    if variant != 'original' and not (ref_dir / 'complete.json').is_file():
        raise ValueError('Run evaluate --variant original first to create held-out reference logits')
    if variant != 'original':
        ref_meta = json.loads((ref_dir / 'complete.json').read_text())
        if ref_meta['logit_tokens'] != logit_tokens:
            raise ValueError('logit-tokens must match the original evaluation')
    device = next(model.parameters()).device
    nll, predictions, sse, sqnorm, kl_sum, agreements, compared, elements = 0., 0, 0., 0., 0., 0, 0, 0
    for i, ids in enumerate(blocks):
        ids = ids.to(device).unsqueeze(0)
        logits = model(input_ids=ids, use_cache=False).logits[0]
        if not torch.isfinite(logits).all():
            raise ValueError('Non-finite model logits')
        for begin in range(0, ids.shape[1] - 1, 64):
            end = min(begin + 64, ids.shape[1] - 1)
            nll += F.cross_entropy(logits[begin:end].float(), ids[0, begin + 1:end + 1], reduction='sum').item()
            predictions += end - begin
        positions = torch.linspace(0, logits.shape[0] - 2, min(logit_tokens, logits.shape[0] - 1)).long().to(device)
        selected = logits[positions].float()
        ref_path = ref_dir / f'{i}.pt'
        if variant == 'original':
            save_tensor(ref_path, selected.cpu())
        else:
            original = load_tensor(ref_path).to(device)
            sse += (selected - original).square().sum().item()
            sqnorm += original.square().sum().item()
            elements += original.numel()
            logp, logq = original.log_softmax(-1), selected.log_softmax(-1)
            kl_sum += (logp.exp() * (logp - logq)).sum().item()
            agreements += (original.argmax(-1) == selected.argmax(-1)).sum().item()
            compared += selected.shape[0]
        del logits, selected
        print(f'evaluate {variant}: {i + 1}/{len(blocks)}', flush=True)
    result = {'variant': variant, 'backend': backend if variant != 'original' else 'original',
              'comparison_scope': 'AWQ per-linear scale-search baseline; no clipping or fused kernels',
              'nll': nll / predictions, 'perplexity': math.exp(min(nll / predictions, 700)),
              'predicted_tokens': predictions, 'resident_tensor_bytes': storage_bytes(model)}
    if variant != 'original':
        result.update(sampled_logit_mse=sse / elements, sampled_logit_relative_mse=sse / max(sqnorm, 1e-30),
                      sampled_kl_original_to_quantized=kl_sum / compared, sampled_top1_agreement=agreements / compared,
                      sampled_positions=compared)
    else:
        write_json(ref_dir / 'complete.json', {'logit_tokens': logit_tokens, 'blocks': len(blocks)})
    write_json(directory / f'eval-{variant}-{backend}.json', result)
    return result


@torch.inference_mode()
def benchmark(model, directory, variant, backend='packed', prompt_length=128,
              new_tokens=32, repeats=5, warmups=2):
    """Single-device, batch-one, fixed-token greedy decoding; EOS does not shorten runs."""
    directory = Path(directory)
    device = next(model.parameters()).device
    ids = load_tensor(directory / 'blocks.pt')['evaluation'][0, :prompt_length].unsqueeze(0).to(device)
    if ids.shape[1] != prompt_length:
        raise ValueError('Benchmark prompt exceeds saved evaluation sequence length')
    if prompt_length + new_tokens > model.config.max_position_embeddings:
        raise ValueError('Benchmark exceeds model context window')
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    def run():
        sync()
        start = time.perf_counter()
        total_start = start
        out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
        nxt, cache = out.logits[:, -1].argmax(-1, keepdim=True), out.past_key_values
        del out
        sync()
        first = time.perf_counter() - start
        start = time.perf_counter()
        for _ in range(new_tokens - 1):
            out = model(input_ids=nxt, past_key_values=cache, use_cache=True, logits_to_keep=1)
            nxt, cache = out.logits[:, -1].argmax(-1, keepdim=True), out.past_key_values
            del out
        sync()
        elapsed = time.perf_counter() - start
        total = time.perf_counter() - total_start
        del cache
        return first, elapsed, total
    for _ in range(warmups):
        run()
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        sync()
        torch.cuda.reset_peak_memory_stats(device)
    times = [run() for _ in range(repeats)]
    prefill = statistics.median(t[0] for t in times)
    decode = statistics.median(t[1] for t in times)
    total = statistics.median(t[2] for t in times)
    result = {
        'variant': variant, 'backend': backend if variant != 'original' else 'original',
        'implementation': 'Unfused PyTorch unpack/dequantize + floating matmul; NOT a fused AWQ kernel' if backend == 'packed' and variant != 'original' else 'floating matmul',
        'resident_tensor_bytes': storage_bytes(model), 'prefill_median_seconds': prefill,
        'decode_median_seconds': decode, 'decode_tokens_per_second': (new_tokens - 1) / decode,
        'end_to_end_median_seconds': total,
        'end_to_end_tokens_per_second': new_tokens / total,
        'prompt_tokens': prompt_length, 'new_tokens': new_tokens, 'batch_size': 1,
        'repeats': repeats, 'warmups': warmups, 'raw_seconds': times,
        'raw_seconds_fields': ['prefill', 'decode', 'end_to_end'],
        'device': str(device), 'torch_version': str(torch.__version__), 'python': platform.python_version(),
    }
    if device.type == 'cuda':
        result.update(gpu=torch.cuda.get_device_name(device),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    else:
        result['peak_allocated_bytes'] = None
        result['memory_note'] = 'CPU tensor bytes only; GPU allocator peaks unavailable'
    write_json(directory / f'benchmark-{variant}-{backend}.json', result)
    return result
