# OpenVINO backend for vLLM (CPU)

This subpackage wires the OpenVINO `torch.compile` backend into vLLM's CPU
model runner. With it, vLLM's per-step `model.forward()` is dispatched to a
pre-compiled OpenVINO graph instead of PyTorch eager / `torch.compile +
inductor`, while vLLM keeps owning scheduling, paged attention, batching,
sampling, etc.

No source patches are required. `vllm-cpu` pins its own PyTorch exactly, so
the install is one `pip install` plus an OpenVINO wheel.

## Requirements

Exercised on the following stack:

| Component | Version | Notes |
|---|---|---|
| CPU | Intel Xeon (Sapphire Rapids or newer) | AMX-BF16 |
| OS | Linux 5.15+ / glibc 2.34+ | THP=madvise |
| Python | 3.11 | 3.10 also works |
| vLLM | `vllm-cpu` 0.28.0 | v1 engine; CPU wheel |
| PyTorch | 2.13.0+cpu | pulled in by `vllm-cpu`, do not pin separately |
| OpenVINO | 2026.5.0 | `vllm_dev` branch until this integration merges |

### Supported vLLM versions

All four are exercised on this branch and produce identical output.
`vllm-cpu` declares an exact `torch==` pin, so you never choose a PyTorch
version yourself:

| `vllm-cpu` | PyTorch pulled in | `compilation_config` key |
|---|---|---|
| 0.25.0 | 2.11.0+cpu | `mode` (0.25.x also accepted `level=3`) |
| 0.26.0 | 2.11.0+cpu | `mode` |
| 0.27.1 | 2.13.0+cpu | `mode` |
| 0.28.0 | 2.13.0+cpu | `mode` |

Newer is recommended. There is no vLLM release that uses torch 2.12 —
0.26 → 0.27 jumps straight from 2.11 to 2.13.

## Install

### 1. Fresh venv

```bash
python3.11 -m venv ~/ov_vllm_env
source ~/ov_vllm_env/bin/activate
python -m pip install -U pip setuptools wheel
```

### 2. vLLM (CPU)

One command. The PyTorch CPU index has to be reachable because `vllm-cpu`
pins the `+cpu` local-version build of torch; PyPI alone cannot satisfy it.

```bash
python -m pip install "vllm-cpu==0.28.0" \
    --extra-index-url https://download.pytorch.org/whl/cpu
```

Do **not** install torch first and do not pass `--index-url` (which
*replaces* PyPI rather than adding to it) — `vllm-cpu` resolves its own exact
torch pin, and forcing a different one only creates a mismatch.

Optional (Linux): preload tcmalloc before Python starts for a cleaner CPU
allocator profile.

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4
```

### 3. OpenVINO

Option A — build from source (recommended until this integration merges to a
released wheel):

```bash
git clone --recursive https://github.com/ynimmaga/openvino.git
cd openvino
git checkout vllm_dev
git submodule update --init --recursive

# patchelf is required for the wheel build target
python -m pip install patchelf

mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_PYTHON=ON \
      -DENABLE_WHEEL=ON \
      -DENABLE_TESTS=OFF \
      -DENABLE_INTEL_GPU=OFF \
      -DENABLE_INTEL_NPU=OFF \
      -DENABLE_OV_PYTORCH_FRONTEND=ON ..
cmake --build . -j "$(nproc)"

# Build the wheel via the ie_wheel target, then install it
cmake --build . --target ie_wheel -j "$(nproc)"
python -m pip install --force-reinstall ./wheels/openvino-*.whl
```

Option B — install a published wheel that already contains this subpackage
(once `vllm_dev` merges upstream):

```bash
python -m pip install "openvino>=2026.5"
```

### 4. Verify the entry point is registered

```bash
python -c "
import importlib.metadata as md
for e in md.entry_points(group='vllm.general_plugins'):
    print(e.name, '->', e.value)
"
```

You should see
`openvino_vllm_cpu -> openvino.frontend.pytorch.torchdynamo.vllm.plugin:register`.
If it's missing, the OV install did not include this subpackage; rebuild the
OpenVINO Python wheel from the branch that contains this PR.

## Usage

The plugin auto-loads via the `vllm.general_plugins` entry point. To select
the OV backend, pass `compilation_config` to `LLM`:

```python
# Must precede `import vllm`: vLLM latches VLLM_USE_LAYERNAME at import
# time, and the OV backend has no working configuration at 1.
from openvino.frontend.pytorch.torchdynamo.vllm.preset import set_pre_import_env
set_pre_import_env()

from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    dtype="bfloat16",          # "float16" and "float32" also work
    enforce_eager=False,
    max_model_len=2048,
    block_size=32,             # OV CPU PagedAttention hard constraint
    compilation_config={
        "mode": "STOCK_TORCH_COMPILE",
        "backend": "openvino",
        "custom_ops": ["none"],
    },
)
out = llm.generate(["Hello!"], SamplingParams(max_tokens=32, temperature=0.0))
print(out[0].outputs[0].text)
```

All of these are required, not optional:

- **`set_pre_import_env()` before `import vllm`** — otherwise torch hoists
  `layer_name` as an opaque graph input and compilation dies with
  `AttributeError: 'LayerName' object has no attribute 'type'`. The plugin
  cannot do this for you: it loads from `vllm.general_plugins`, after vLLM
  has already frozen the value.
- **`"mode": "STOCK_TORCH_COMPILE"`** — the plugin activates on this mode
  only. `VLLM_COMPILE` is rejected (`Invalid backend for piecewise
  compilation`) because it would double-wrap `torch.compile`;
  `DYNAMO_TRACE_ONCE` fails on `SerializableCallable`.
- **`"custom_ops": ["none"]`** — otherwise vLLM expands the fused
  gated-MLP / RMSNorm custom ops, which the traced graph cannot consume, and
  model load fails with `ValueError: could not broadcast input array from
  shape (N, 4096) into shape (N, 2048)`.
- **`block_size=32`** — a hard constraint of the OV CPU PagedAttention
  kernel.

Without a matching `compilation_config` the plugin's gate
(`compilation_config.backend == "openvino"`) returns False, and vLLM runs its
normal CPU path; OV is not engaged.

### Precision

You do not configure precision. The backend reads the model's own float dtype
off the converted graph and derives `INFERENCE_PRECISION_HINT` and
`KV_CACHE_PRECISION` from it, then redeclares the PagedAttention KV-cache
Parameters to match. Both keys always name the same type, because the OV CPU
PagedAttention kernel is only instantiated for matching (compute, cache)
pairs.

`bfloat16`, `float16` and `float32` models all run. bf16 is the compute type
in every case: it is exact for bf16 models, and for f16 models it avoids an
overflow in vLLM's unfused RMSNorm, whose `x**2` reduction exceeds f16's
65504 ceiling and returns NaN (bf16 has f32's exponent range, so it cannot).
To override, set `OV_INFERENCE_PRECISION_HINT` and `OV_KV_CACHE_PRECISION` —
set **both**, to the same value, or `compile_model` will reject the pair.

## Environment variables

Set these before running your script. The plugin's `register()` fills in
`VLLM_CPU_KVCACHE_SPACE` and `OV_FAST_INFER` when they are unset.

### OV plugin — fast paths

| Variable | Default | Effect |
|---|---|---|
| `OV_LM_HEAD` | `0` | When `1`, compile `lm_head` on OV instead of leaving it on torch/oneDNN. `lm_head` runs outside the compiled graph (vLLM calls `compute_logits()` separately), so on torch it is governed by `OMP_NUM_THREADS`, and at 1–2 threads it costs far more than it needs to; on OV that dependency disappears. Off by default anyway: the OV path adds a second compiled model whose InferRequest is invoked between main-graph infers, and that interleaving roughly doubles the **main graph's** per-step time — a larger loss than the `OMP_NUM_THREADS` mistuning it avoids. Not caused by the lm_head kernel, and not fixable by capping the second model's threads, disabling its pinning, or sharing one `ov.Core`. Enable it where `OMP_NUM_THREADS` cannot be tuned per model and per machine. |
| `OV_FAST_INFER` | `1` (set by plugin) | Bypass the `_data_dispatch` dict walk in `req.infer()`; use `set_tensor(port, ...)` directly and cache the `ov.Tensor` wrappers and output views per `id(InferRequest)`. Falls back to the slow path on any error. Set `0` to disable. |
| `OV_NATIVE_SAMPLER` | `0` | When `1`, use a native OV opset13 graph for sampling (topk + softmax + Gumbel-max), bypassing the `torch.compile(backend="openvino")` layer. Trade-off: skips top_p rejection (pure Gumbel-max over top-k values). No effect on greedy. |
| `OV_FUSED_SAMPLER_MIN_VOCAB` | `100000` | Vocab-size gate for the fused / native samplers. Below this threshold torch's `apply_top_k_top_p` on CPU beats round-tripping through a compiled OV graph. Set `0` to enable for all vocab sizes. |
| `OV_FAST_SAMPLER_HINT` | `f32` | `INFERENCE_PRECISION_HINT` for the native OV sampler's compiled model. One of `f32`, `f16`, `bf16`. |
| `OV_INFERENCE_NUM_THREADS` | (auto) | Thread count for OV inference. Overrides OV's automatic detection. |
| `OV_INFERENCE_PRECISION_HINT` | (derived) | `INFERENCE_PRECISION_HINT` for the main model compile. Derived from the model dtype when unset (see Precision). Escape hatch only. |
| `OV_KV_CACHE_PRECISION` | (derived) | `KV_CACHE_PRECISION`. Must equal `OV_INFERENCE_PRECISION_HINT`; the OV CPU PagedAttention kernel only exists for matching pairs and `compile_model` throws otherwise. Set both or neither. |

### OV plugin — correctness / diagnostics

| Variable | Default | Effect |
|---|---|---|
| `OV_DISABLE_FUSED_SAMPLER` | (unset) | Any non-empty value skips the `install()` sampler monkey-patch entirely; vLLM uses its own sampler in all cases. Useful for A/B comparing the sampler's contribution. |
| `OV_PERF_COUNT_OUT` | (unset) | Path to write per-node OV profiling info (one line per node per infer call): `node_type<TAB>node_name<TAB>real_time_us<TAB>cpu_time_us<TAB>exec_type`. Enables `PERF_COUNT=YES` on the compile config. |
| `OV_PA_FUSE_UPSTREAM_RESHAPE` | (enabled) | Set `0` to disable the PA translator's Q-input upstream Reshape fusion. Debug switch — normally leave alone. |
| `VLLM_USE_LAYERNAME` | `0` (**forced, not a knob**) | The OV backend has no working configuration at `1`: torch hoists `layer_name` as an opaque `LayerName` graph input instead of a constant, and `torchdynamo/compile.py` raises `'LayerName' object has no attribute 'type'`. `preset.set_pre_import_env()` pins it to `0`, overriding any value you export; call it before `import vllm`. The plugin cannot do this for you — it loads from `vllm.general_plugins`, after `vllm.utils.torch_utils` has latched the value — so it only warns. |

### vLLM environment (relevant subset)

Not owned by this plugin, but relevant to how vLLM+OV runs:

| Variable | Default | Effect |
|---|---|---|
| `VLLM_CPU_KVCACHE_SPACE` | `4` (set by plugin) | GiB of RAM reserved for the KV cache. The plugin sets `4` when unset — enough for 1–2B models at 2k context on a shared node. Raise it for bigger models or longer context. Setting `0` on a shared machine can trigger `Available memory on node 0 ... is less than requested memory for kv`. |
| `OMP_NUM_THREADS` | (varies) | vLLM's thread count for its own numpy / torch ops. Under the default `OV_LM_HEAD=0`, `lm_head` runs on torch/oneDNN and this knob matters: leaving it unset is right for small models, while larger ones need it set explicitly (the exact value matters far less than setting it at all). Under `OV_LM_HEAD=1` it is close to irrelevant. Independent of `OV_INFERENCE_NUM_THREADS`. |
| `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | `300` | Timeout for the worker's `execute_model` RPC. Raise it if a first-infer compile is slow. |
| `VLLM_USE_AOT_COMPILE` | `0` | Keep `0` for the OV backend. |
| `MALLOC_ARENA_MAX` | (unset) | Set `4` to bound the glibc heap-arena count. Prevents the `glibc chunk_main_arena assertion failed` crash on some vLLM 0.25.x builds. |

## Tuning

Two settings dominate the rest:

- **Pinning.** Decode is memory-bandwidth-bound. Confine the process to one
  NUMA node's physical cores (`taskset` / `numactl`); spanning sockets or
  including SMT siblings costs more than most other tuning. Note that OV
  sizes its thread pool from the socket's core count and does not narrow it
  to the affinity mask, so a mask smaller than the socket oversubscribes.
- **`OMP_NUM_THREADS`.** Under the default `OV_LM_HEAD=0`, `lm_head` runs on
  torch/oneDNN and this governs it. Leaving it unset suits small models;
  larger ones need it set explicitly.

Relative standing against vLLM eager and `torch.compile + inductor` depends
on the model, dtype, core count and pinning, so measure your own workload
rather than relying on a quoted number.

## Benchmarking

`scripts/bench_run.py` runs the eager-vs-OV A/B: it loads the model on each
path, runs warmup + a short run + a full run, reports steady-state throughput
for both, and compares the two output strings byte-for-byte (greedy decode,
so they must match). Exit code is non-zero if the texts differ.

```bash
taskset -c 0-39 python -m openvino.frontend.pytorch.torchdynamo.vllm.scripts.bench_run \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --max-new-tokens 128
```

`taskset` (or `numactl`) is required on multi-socket systems to keep threads
and memory on one NUMA node; the plugin warns at startup when the process is
not pinned. To also bind memory:

```bash
numactl --cpunodebind=0 --membind=0 -- \
  taskset -c 0-39 \
  python -m openvino.frontend.pytorch.torchdynamo.vllm.scripts.bench_run \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --max-new-tokens 128 --mode openvino
```

| Flag | Default | Purpose |
|---|---|---|
| `--model` | TinyLlama-1.1B-Chat-v1.0 | HF id or local path. |
| `--prompt` | "The capital of France is " | Generation prompt. |
| `--max-new-tokens` | 64 | Tokens generated per measurement run. |
| `--skip-warmup-tokens` | 5 | First N tokens excluded from steady-state throughput. |
| `--dtype` | bfloat16 | One of `bfloat16` / `float16` / `float32`. |
| `--max-model-len` | 2048 | Sequence length for vLLM's KV-cache sizing. |
| `--mode` | both | `eager`, `openvino`, or `both`. |

Reading the output:

- **`full`** — total output tokens divided by the wall time of the timed
  `.generate()` call. Includes the first-token cold cost.
- **`steady`** (`skip N`) — subtracts the first N tokens of time and count.
  This is the amortized decode rate, and the number to quote.
- **`match: True`** — OV output matches eager byte-for-byte at
  temperature=0. If False, either a fusion produced semantically wrong code
  or the model has a bf16 numerical-drift point that flips a token at greedy;
  read the failing text before assuming a bug. bf16 is not a stable regime
  for byte-exact comparison — the regression test pins float32 for that
  reason.

## Tests

```bash
python -m pytest tests/vllm_tests -v
```

Covers eager-vs-OV output equality, dynamic shapes, lm_head weight reload,
and the advanced-feature surface (logprobs, continuous batching, grammar,
beam search, speculative decoding, prefix caching, custom logits processors).

## Troubleshooting

- **`KeyError: 'openvino'` from torch.compile**: the OV pytorch frontend was
  not installed, or `entry_points.txt` doesn't list
  `[torch_dynamo_backends] openvino = openvino.frontend.pytorch.torchdynamo.backend:openvino`.
  Reinstall the OV Python wheel.
- **`AttributeError: 'LayerName' object has no attribute 'type'`**: you did
  not call `set_pre_import_env()` before `import vllm`. See Usage.
- **`ValueError: could not broadcast input array from shape (N, 4096) into
  shape (N, 2048)`**: `custom_ops: ["none"]` is missing from
  `compilation_config`.
- **`ValueError: Field 'level' not found in CompilationConfig`**: you are on
  a newer vLLM that renamed `level` to `mode`. Use `{"mode":
  "STOCK_TORCH_COMPILE", ...}`. vLLM 0.25.x used `level=3`; 0.26+ uses
  `mode`.
- **`expect kvcache type bf16, current: f16`** (or any other pair) from
  `compile_model`: you set only one of `OV_INFERENCE_PRECISION_HINT` /
  `OV_KV_CACHE_PRECISION`, or set them to different types. Set both to the
  same value, or unset both.
- **First `generate()` hangs at `Processed prompts: 0%`**, with
  `shm_broadcast.py` warning "No available shared memory broadcast block
  found in 60 seconds": the worker is stuck inside `posix_memalign` / TBB /
  glibc heap-lock contention during the first OV infer. This was a rank-4
  KV-cache layout bug fixed in `side_channel.py` — rebuild OpenVINO from a
  current `vllm_dev`. To confirm before rebuilding, set
  `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800` and `py-spy dump --native` the
  worker; the stack pins in `libopenvino_intel_cpu_plugin.so`.
- **`free(): invalid size` under `LD_PRELOAD=libtcmalloc`**: seen on some
  vLLM 0.25.x releases. Unset `LD_PRELOAD` and use `MALLOC_ARENA_MAX=4`.
- **`libnvrtc.so.13: cannot open shared object file`**: a CUDA-flavored torch
  or torchvision was resolved. Reinstall with
  `--extra-index-url https://download.pytorch.org/whl/cpu` and check that
  `python -c "import torch; print(torch.__version__)"` ends in `+cpu`.
- **Fused sampler didn't fire** (grep `[OV plugin] Fused sampler compiled` in
  stderr): the vocab-size gate skips small-vocab models. Override with
  `OV_FUSED_SAMPLER_MIN_VOCAB=0`.
- **Bimodal fast/slow throughput across runs**: usually accumulated ghost
  workers or `/dev/shm/__KMP_REGISTERED_LIB_*` files from prior crashed runs:
  ```bash
  pkill -9 -f "VLLM::"
  rm -f /dev/shm/__KMP_REGISTERED_LIB_*
  ```
- **A fusion did not fire**: set `OV_PERF_COUNT_OUT=/tmp/ov.log` and inspect.
  `LLMMLP`, `FullyConnected`, `QKVProjection` and `PagedAttentionExtension`
  should each appear once per layer per decode step.

## Limitations

- CPU-only for now. The OV GPU/NPU paths are not tested by this integration
  yet.
- `compilation_config["mode"]` must be `STOCK_TORCH_COMPILE`. `VLLM_COMPILE`
  invokes vLLM's own `torch.compile` wrapper and would double-wrap.
- f16 models execute with bf16 compute, not f16 (see Precision).
- Speculative decoding, beam search, grammar-constrained decoding, custom
  logit processors, and `logprobs > 0` use the slower vLLM Python sampler
  fallback (the OV-fused sampler eligibility check rejects these).
- Continuous batching, prefix caching, and paged attention are unaffected and
  continue to work.
- **Tensor parallelism is not supported.** `tensor_parallel_size > 1` fails
  during graph compilation.

## Layout

| File | Role |
|---|---|
| `plugin.py` | `vllm.general_plugins` entry point. Patches `CPUModelRunner.load_model` to wrap `model.forward` with `torch.compile(backend="openvino", options={"vllm": True})`. |
| `paged_attention.py` | Custom `torch.ops.openvino.paged_attention` op + FX rewrite that converts vLLM's `auto_functionalized_v2(unified_attention_with_output, ...)` HOP nodes into it, so the OV pytorch frontend translates the call to `PagedAttentionExtension`. |
| `side_channel.py` | At infer time, binds the `__pa__<layer>__<field>` Parameters (KV cache, block tables, past_lens, ...) from `vllm.forward_context.get_forward_context()`. |
| `lm_head.py` | Optional OV-compiled `lm_head`, gated on `OV_LM_HEAD`. |
| `sampler.py` | Optional OV-fused fast path for `vllm.v1.sample.sampler.Sampler`. |
| `preset.py` | `options["vllm"] = True` mega-preset that expands into per-flag defaults plus OV CPU-config defaults. |
| `compile_hooks.py` / `runtime_hooks.py` | Helpers called from the generic `torchdynamo/compile.py` and `torchdynamo/execute.py` so the standalone `torch.compile(backend="openvino")` path stays free of vLLM-specific code. |
| `scripts/bench_run.py` | Manual A/B: runs the same prompt through vLLM eager and vLLM+OV, compares output text and reports steady-state throughput. |
