# Output-guided hybrid AWQ experiment

`train.py` is the main entry point for **Qwen/Qwen3-14B-Base**. This research
experiment compares the original floating model, a paper-style AWQ scaling
baseline, and a hybrid that preserves selected **input weight columns in FP16**.
All remaining targeted weights use **INT4, group size 128** by default.

**Status:** mathematical tests and a small random Qwen3 integration run are tested.
Full Qwen3-14B accuracy, GPU memory, and throughput must be measured on your GPU;
no 14B improvement or speedup is claimed. The standard starting dtype is FP16.
A single GPU must fit the original 14B checkpoint (~28 GB of raw FP16 weights)
plus calibration/evaluation working memory. An A100 80 GB is an appropriate test
host; this implementation does not silently offload to CPU or distribute layers.

## What the experiment implements

Use W shaped `[out_features, in_features]` and X shaped `[tokens, in_features]`.
The original projection output is `Y = X @ W.T + bias`.

1. **Detect large original outputs.** At each token, flag output channel i when
   `abs(Y[t,i]) > k * RMS(Y[t,:])`. The default experiment separately fits k=4,
   k=6, and k=8; k=6 is the primary configuration. These are relative multiples
   of the typical output magnitude, **not** Dettmers' absolute input threshold 6.
   The 4–8 range is a heuristic sensitivity analysis: 4 is more permissive, 8 more
   restrictive. It is not a statistical significance test and assumes no Gaussian
   distribution. The RMS includes all output coordinates. Large values themselves
   can raise this threshold. For fewer than k² outputs the test cannot pass.
2. **Apply both persistence conditions.** An output index must be flagged at >=6%
   of all calibration token positions in its projection, and be locally frequent
   in >=ceil(25% * number_of_blocks) blocks of the SAME projection family. A layer
   uses that index only if it is also locally frequent there. These conditions
   apply to output channels **before** attribution, not the subsequently selected
   input columns. Comparing channel indices across blocks is an experimental
   heuristic, not proof that they encode the same semantic feature. No q/k/v/MLP
   index spaces are mixed.
3. **Rank contributing input columns.** On uniformly sampled calibration tokens,
   rank column j by `sum_(t,i flagged and eligible) abs(X[t,j] * W[i,j])`.
   This never allocates a three-dimensional contribution tensor. A large score
   only creates a candidate; it does not prove that FP16 will help.
4. **Measure the benefit of exclusion.** Try each candidate column at the current
   alpha. Exclude its weights from INT4 group extrema; encode their slots as real
   zero; add their original FP16 column contribution once. Accept the best column
   only if projection output MSE improves by more than 0.01% relative to current
   MSE. Re-search all alphas after every accepted column. Repeat until no candidate
   helps or the budget is exhausted. The default candidate pool is eight columns;
   the budget is `floor(0.001 * input_width)` (0.1%), capped by the pool. This is
   a memory guardrail, not a quota. A zero budget or no qualifying outputs gives
   the ordinary AWQ baseline exactly. Use `--max-outlier-fraction 0.0001` to test
   0.01%; with 5,120 inputs that rounds DOWN to zero columns, not one.
5. **Search AWQ scales.** Collect `a_j = mean(abs(X[:,j]))` over ALL calibration
   tokens. On the remaining columns search `s_j = a_j**alpha` with 20 candidates:
   `0, .05, ..., .95`. Clamp raw scales to >=1e-4 and divide by the geometric mean
   of their minimum and maximum, matching upstream's activation-only scale rule.
   Excluded columns have scale 1 and do not affect normalization. Optimize the
   original paper's linear-output MSE. Quantize `W * s`, then fold `1/s` into the
   recovered weight. No backward pass, model training, or runtime outlier detection
   is used.

The baseline and hybrid use the same per-linear search objective, group layout,
zero-inclusive asymmetric INT4 quantizer (codes 0..15), and calibration samples.
Biases stay floating and are added once. Embeddings, LM head, normalizations,
RoPE, attention/softmax, residual additions, SiLU, and elementwise gating stay
unchanged. All seven q/k/v/o/gate/up/down linear projections are covered.
Qwen3 grouped-query attention is handled by its existing forward implementation;
we do not incorrectly fuse o_proj scales backward through repeated KV heads or
move scales through SiLU/QK normalization.

**Baseline scope:** this is an AWQ **scale-search-only research baseline**, not a
bit-for-bit reproduction of the official AWQ package. The upstream implementation
can jointly evaluate attention/MLP submodules, fuse scales into preceding layers,
and run weight clipping searches. Here alpha is searched independently per linear
projection; no clipping search is applied in either arm. Results do not establish
an improvement over a fully tuned production AWQ implementation. The custom
packed files are not AutoAWQ/vLLM checkpoint files.

References:
- [AWQ paper](https://arxiv.org/abs/2306.00978), particularly the scale-search objective.
- [Official activation scale / 20-point search](https://github.com/mit-han-lab/llm-awq/blob/main/awq/quantize/auto_scale.py).
- [Official calibration source](https://github.com/mit-han-lab/llm-awq/blob/main/awq/utils/calib_data.py).
- [LLM.int8()](https://arxiv.org/abs/2208.07339). Its 25%/6% conditions describe its
  analysis of recurring **input** outliers; applying them to our output detector
  is a new experimental rule, not a replication of LLM.int8().

## Setup and existing HF cache

Activate your existing environment, then install dependencies. Retain your
CUDA-compatible PyTorch build. Tested here with PyTorch 2.14.0 and Transformers
4.57.6; this experiment requires the Transformers 4.x Qwen3 adapter.

```bash
conda activate qwen-quant
python -m pip install -r requirements.txt
python -m pytest -q
python train.py --help
```

Model loading defaults to `local_files_only=True`, using the normal HF cache or
`HF_HOME`. If necessary, pass `--cache-dir /path/to/hub` to **each** command, or
use `--model /path/to/local/checkpoint`. `--allow-model-download` explicitly
permits a missing checkpoint to be fetched. `--offline` also disables dataset
network access. The code does not embed access tokens.

## 1. Calibrate and fit

```bash
python train.py calibrate --output outputs/qwen14b-v1
```

The default text source is `mit-han-lab/pile-val-backup`, validation split,
shuffled with seed 42. Separate source documents are assigned to calibration and
held-out evaluation (every fifth document for evaluation); duplicates are
excluded by content hash. Text is concatenated with EOS boundaries into **128
calibration blocks and 32 evaluation blocks of 512 tokens**, without padding.
This block count is not the same as upstream's count of source documents.
The default dataset may be downloaded if it is not cached.

Output event frequencies and activation means use **all 65,536 calibration
tokens**. Expensive scale/exclusion searches use 512 uniformly sampled tokens per
projection. Held-out projection MSE uses 256 separate evaluation tokens. These
limits are configurable and recorded; increase `--search-tokens` for a stronger,
more expensive calibration. Never use the held-out set to fit alpha or columns.
The three k variants are reported separately; the code does not choose a winner
using held-out data. If you select a k using these results, use another test set
for final claims.

For your own data:

```bash
python train.py calibrate --calib-data my_calibration.jsonl --eval-data my_evaluation.jsonl --output outputs/custom-v1
```

Supported: TXT with one document per nonblank line; JSONL with `{"text":"..."}`
objects or strings; JSON lists of those objects or strings. Change the key with
`--text-field`. If `--eval-data` is omitted, the same document-disjoint splitting
rule is used. Insufficient text raises an error rather than reusing training text
for evaluation or inserting padding.

Fitting can be expensive: it evaluates every alpha and candidate on each
projection. For a smaller **pilot** (not a final accuracy claim):

```bash
python train.py calibrate --calib-samples 8 --eval-samples 4 --search-tokens 128 --validation-tokens 64 --candidate-pool 4 --multipliers 6 --output outputs/pilot
```

Each experiment contains token blocks/hashes, checkpoint fingerprint, sampled
projection inputs, output frequencies, per-layer alphas/scales, excluded column
indices, packed INT4 tensors, FP16 columns, selection traces, and held-out
projection errors. Output directories must be new. Allow tens of GB of disk for
14B variants and several GB of CPU activation samples. Identical no-outlier
variants share baseline files. Original checkpoint files are never changed.

If fitting was interrupted **after** the survey completed, restart it:

```bash
python train.py fit --artifact outputs/qwen14b-v1
```

This redoes fitting from the saved survey using the recorded search configuration;
it does not resume individual alpha trials. `manifest.json` is only written when
all projections are finished. A partial run cannot be evaluated as complete.

## 2. Compare accuracy

```bash
python train.py compare --artifact outputs/qwen14b-v1 --backend fake
```

This runs original, AWQ, and all saved hybrid variants in separate processes,
using the same held-out token blocks. It reports:

- Full next-token negative log likelihood and perplexity on held-out blocks.
- Sampled full-vocabulary logit MSE/relative MSE, KL(original || quantized), and
  top-1 agreement against original logits (32 evenly spaced positions per block).
- Per-projection held-out MSE in `manifest.json` from the fitting stage. These
  local measurements use original input activations and do not include propagated
  errors; the end-to-end metrics above do.

Fake mode folds the recovered weights into ordinary floating linear layers. It
is an **accuracy simulation**, not compressed model memory or INT4 runtime speed.
For actual separate-path numerical effects (different FP16 accumulation order),
also evaluate the packed implementation:

```bash
python train.py compare --artifact outputs/qwen14b-v1 --backend packed
```

You can run one variant with `evaluate --variant original`, `--variant awq`, or
`--variant hybrid-6`. Evaluate original first to create reference logits. Both
reference and comparison runs must use the same `--logit-tokens` value. The full
checkpoint fingerprint and dtype must match; recalibration is required after
checkpoint edits. The command prints the actual NLL as well as perplexity; the
exponential is capped at exp(700) only to avoid overflow in degenerate models.

## 3. Measure memory and speed

```bash
python train.py benchmark-all --artifact outputs/qwen14b-v1 --backend packed
```

Each variant gets a fresh process, warmup, CUDA synchronization, and allocator
peak reset. Reports include resident tensor bytes (including quantization
metadata), peak CUDA allocated/reserved bytes during inference, prefill time,
decode tokens/s, raw timings, GPU identity, and versions. Default: batch one,
128-token prompt, 32 generated tokens, two warmups, five repeats. The decode rate
counts 31 post-prefill tokens; end-to-end rate counts all 32. Decoding uses a KV
cache and does not stop early at EOS. It does not time checkpoint loading, but the
original floating checkpoint must still fit during loading. CPU runs report tensor
storage only, not a misleading GPU-memory number.

**Packed backend limitation:** two INT4 codes are actually stored per byte, but
the reference forward unpacks/dequantizes one full weight matrix to floating
point, performs a standard matmul, then adds the FP16-column matmul. It is NOT a
fused CUDA/Triton kernel. It may be slower than the original, and its temporary
expanded weights contribute to peak memory. Baseline AWQ and hybrid use the same
backend so the side-path cost is visible. These timings cannot predict optimized
AWQ kernel throughput. A fused implementation is a separate next step if the
accuracy experiment earns its complexity.

Quantized slots corresponding to FP16 columns remain allocated and encode real
zero. FP16 columns and their indices are therefore additional storage; the code
reports actual bytes rather than assuming ideal removal of packed slots.

## Recalibrate after fine-tuning

Use the new floating checkpoint and new calibration data in a **new directory**:

```bash
python train.py calibrate --model /path/to/finetuned-checkpoint --calib-data new_train.jsonl --eval-data new_eval.jsonl --output outputs/finetuned-v2
python train.py compare --model /path/to/finetuned-checkpoint --artifact outputs/finetuned-v2 --backend fake
```

This recomputes outputs, persistence masks, candidate attribution, retained FP16
columns, alphas, scales, and quantized weights. Reusing an old artifact with changed
weights/dtype is rejected by SHA-256 checks. This is post-training recalibration;
the hybrid model itself is not a fine-tuning implementation.

## Offline smoke test

```bash
python train.py smoke --output outputs/smoke
```

Creates a tiny random two-block Qwen3 with grouped-query attention, Q/K norms,
RoPE and SwiGLU; tests survey, fitting, saving/loading, and full-model evaluation
for both fake and packed variants. It uses k=2 because k>=6 is often impossible
for the tiny output dimensions. Tiny/random results are engineering tests, **not
Qwen3-14B evidence**. Full deterministic tests also cover asymmetric real-zero
encoding, partial groups, restoration without double counting, persistence,
calibration-only fitting, budget rounding, stale artifacts, and decoding.
