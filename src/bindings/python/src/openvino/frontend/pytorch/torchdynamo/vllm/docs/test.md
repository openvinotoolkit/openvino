# Test — vLLM+OV CPU backend

How to run the smoke test, interpret its output, and control OV
plugin behavior via environment variables. See [setup.md](setup.md)
for install instructions.

## Smoke test

```bash
taskset -c 0-39 python -m openvino.frontend.pytorch.torchdynamo.vllm.tests.test_run \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --max-new-tokens 128
```

The plugin's `register()` sets `VLLM_CPU_KVCACHE_SPACE=4` and
`OV_FAST_INFER=1` automatically when they are unset, so a bare
`python -m ...` command is enough on a single-socket box. `taskset`
(or `numactl`) is still required on multi-socket systems to keep
threads and memory on one NUMA node; the plugin emits a warning at
startup when the process is not CPU-pinned.

What it does:

1. Loads the model with `enforce_eager=True` (vLLM's eager path),
   runs warmup + a short run + a full run, reports steady-state
   tok/s and the generated text.
2. Loads the same model with
   `compilation_config={"mode": "STOCK_TORCH_COMPILE", "backend": "openvino"}`,
   repeats, reports tok/s.
3. Compares the two output strings byte-for-byte (greedy decode, so
   they must match) and prints the speedup.

Exit code is non-zero if the texts differ.

### Useful flags

| Flag | Default | Purpose |
|---|---|---|
| `--model` | TinyLlama-1.1B-Chat-v1.0 | HF id or local path. |
| `--prompt` | "The capital of France is " | Generation prompt. |
| `--max-new-tokens` | 64 | Tokens generated per measurement run. |
| `--skip-warmup-tokens` | 5 | First N tokens excluded from steady-state perf. |
| `--dtype` | bfloat16 | One of `bfloat16` / `float16` / `float32`. |
| `--max-model-len` | 2048 | Sequence length for vLLM's KV cache sizing. |
| `--mode` | both | `eager`, `openvino`, or `both`. |

### Recommended environment for stable measurements

```bash
# Pin to a single NUMA socket on a multi-socket box.
numactl --cpunodebind=0 --membind=0 -- \
  taskset -c 0-39 \
  python -m openvino.frontend.pytorch.torchdynamo.vllm.tests.test_run \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --max-new-tokens 128 --mode openvino
```

`taskset` keeps threads on one socket; `numactl --membind` keeps
memory local. The OV plugin widens process affinity once at compile
time (TBB samples affinity on first parallel use), so the wide mask
only takes effect after `Core()` is instantiated; pre-pinning to a
single socket is still honored.

## Environment variables

Env vars documented here control the OV plugin's fast paths and
diagnostics. Set them before running the smoke test or your own
vLLM script.

### Fast paths (perf tuning)

| Variable | Default | Effect |
|---|---|---|
| `OV_FAST_INFER` | `1` (set by plugin) | Bypass `_data_dispatch` dict walk in `req.infer()`; use `set_tensor(port, ...)` directly, cache the ov.Tensor wrappers and output views per `id(InferRequest)`. Measured **+5-15% greedy** across 6 models, no correctness regression on Gemma-4 hybrid attention. Falls back to slow path on any error. Set `OV_FAST_INFER=0` to disable. |
| `OV_NATIVE_SAMPLER` | `0` | When `1`, use a native OV opset13 graph for sampling (topk + softmax + Gumbel-max), bypassing the `torch.compile(backend="openvino")` layer. Measured **+11-30% sampling** across models. Trade-off: skips top_p rejection (uses pure Gumbel-max over top-k values). No effect on greedy (bypassed via eligibility gate). |
| `OV_FUSED_SAMPLER_MIN_VOCAB` | `100000` | Vocab-size gate for the fused / native samplers. Below this threshold, torch's `apply_top_k_top_p` on CPU is faster than round-tripping through a compiled OV graph. Set to `0` to enable the fused sampler for all vocab sizes. |
| `OV_FAST_SAMPLER_HINT` | `f32` | `INFERENCE_PRECISION_HINT` for the native OV sampler compiled model. Options: `f32`, `f16`, `bf16`. |
| `OV_INFERENCE_NUM_THREADS` | (auto) | Thread count for OV inference. Explicit setting overrides OV's automatic detection. |
| `OV_INFERENCE_PRECISION_HINT` | `f16` | OV plugin `INFERENCE_PRECISION_HINT` for the main model compile. Use `bf16` on SPR to get AMX-BF16 kernels for LLMMLP / QKVProjection / FullyConnected. |
| `OV_KV_CACHE_PRECISION` | `f32` | OV plugin `KV_CACHE_PRECISION`. Use `bf16` to halve KV-cache memory bandwidth on SPR. |

### Correctness / diagnostics

| Variable | Default | Effect |
|---|---|---|
| `OV_DISABLE_FUSED_SAMPLER` | (unset) | If set to any non-empty value, skip the `install()` sampler monkey-patch entirely; vLLM uses its own sampler in all cases. Useful for A/B comparing sampler contribution to perf. |
| `OV_PERF_COUNT_OUT` | (unset) | Path to write per-node OV profiling info (one file, one line per node per infer call). Format: `node_type<TAB>node_name<TAB>real_time_us<TAB>cpu_time_us<TAB>exec_type`. Setting this enables `PERF_COUNT=YES` on the OV compile config. |
| `OV_PA_FUSE_UPSTREAM_RESHAPE` | (enabled) | Set to `0` to disable the PA translator's Q-input upstream Reshape fusion. Debug switch — normally leave alone. |
| `VLLM_USE_LAYERNAME` | `0` (plugin forces this) | If `1`, PA translator embeds full vLLM layer names in Parameter names. Kept `0` so translator uses short numeric layer suffixes. |

### vLLM environment (relevant subset)

Not owned by this plugin but relevant to how vLLM+OV runs:

| Variable | Default | Effect |
|---|---|---|
| `VLLM_CPU_KVCACHE_SPACE` | `4` (set by plugin) | GiB of RAM reserved for KV cache. The plugin sets `4` when unset — enough for 1-2B models at 2k context on a shared node. Set explicitly to a larger value for bigger models or longer context. Setting `0` on a shared machine can trigger `Available memory on node 0 ... is less than requested memory for kv`. |
| `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | `300` | Timeout for the worker's `execute_model` RPC. If a first-infer compile is slow, raise this. |
| `VLLM_USE_AOT_COMPILE` | `0` | Keep `0` for OV backend. |
| `OMP_NUM_THREADS` | (varies) | vLLM's internal thread count for its own numpy / torch ops. Typical: `2`. Independent of `OV_INFERENCE_NUM_THREADS`. |
| `MALLOC_ARENA_MAX` | (unset) | Set to `4` to bound glibc heap-arena count. Prevents the `glibc chunk_main_arena assertion failed` crash on some vLLM 0.25.x builds. |

## Running your own benchmark

The `test_run` script covers the golden-path A/B. For custom workloads,
use the same LLM setup:

```python
import os
# OV_FAST_INFER=1 and VLLM_CPU_KVCACHE_SPACE=4 are auto-set by the plugin;
# only override them here if you want non-default behavior.
os.environ["OV_NATIVE_SAMPLER"] = "1"      # native OV sampler (sampling only)
os.environ["OV_INFERENCE_PRECISION_HINT"] = "bf16"
os.environ["OV_KV_CACHE_PRECISION"] = "bf16"

from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    dtype="bfloat16",
    enforce_eager=False,
    max_model_len=2048,
    block_size=32,     # required: OV CPU PagedAttention hard constraint
    compilation_config={
        "mode": "STOCK_TORCH_COMPILE",
        "backend": "openvino",
        "custom_ops": ["none"],  # keep vLLM from expanding RMSNorm/SiLU
    },
)

# Greedy
out = llm.generate(["The capital of France is"],
                   SamplingParams(max_tokens=64, temperature=0.0))
print(out[0].outputs[0].text)

# Sampling — native OV sampler fires if vocab >= 100k
out = llm.generate(["The capital of France is"],
                   SamplingParams(max_tokens=64, temperature=1.0,
                                  top_p=0.95, top_k=50))
print(out[0].outputs[0].text)
```

## Interpreting perf numbers

- **`full` tok/s**: total output tokens divided by wall time of the
  timed `.generate()` call. Includes the first-token cold cost.
- **`steady`** (`skip N`) tok/s: subtracts the first N tokens of
  time and count. This is the amortized decode rate — the number to
  quote for comparisons.
- **`match: True`** — the OV output matches eager output byte-for-byte
  at temperature=0. If False, either a fusion produced semantically
  wrong code, or the model has a bf16 numerical-drift point that
  flips a token at greedy — see the failing text before assuming a
  bug.

## Diagnosing perf regressions

- **Compare against Inductor**: run `bench_compare.py inductor` to
  cross-check whether the workload has a known-good baseline. If
  Inductor also regressed, the issue is vLLM's step loop, not OV.
- **Get per-op OV counters**: set `OV_PERF_COUNT_OUT=/tmp/ov.log`
  and inspect. `LLMMLP` + `FullyConnected` + `QKVProjection` +
  `PagedAttentionExtension` should each fire once per layer per
  decode step. If they're 0, the corresponding fusion did not fire.
- **Bimodal fast/slow tok/s across runs**: usually accumulated ghost
  workers or `/dev/shm` `__KMP_REGISTERED_LIB_*` files from prior
  crashed runs. Kill stragglers and clean shm:
  ```bash
  pkill -9 -f "VLLM::"
  rm -f /dev/shm/__KMP_REGISTERED_LIB_*
  ```
- **First-infer hang** (`Processed prompts: 0%` for minutes): apply
  the KV chunk fix in `side_channel.py` — see banner at top of
  [setup.md](setup.md).
