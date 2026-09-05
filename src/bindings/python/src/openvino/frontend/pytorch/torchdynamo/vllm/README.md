# OpenVINO backend for vLLM (CPU)

This subpackage wires the OpenVINO `torch.compile` backend into vLLM's CPU
model runner. With it, vLLM's per-step `model.forward()` is dispatched to a
pre-compiled OpenVINO graph instead of PyTorch eager / `torch.compile +
inductor`, while vLLM keeps owning scheduling, paged attention, batching,
sampling, etc.

> ⚠️ **Setup:** requires a source patch and a pinned pip lockfile.
> See [setup.md](docs/setup.md) before running.


## Documentation

- **[setup.md](docs/setup.md)** — install (venv, PyTorch, vLLM, OpenVINO), verify entry point, select the OV backend, troubleshooting.
- **[test.md](docs/test.md)** — run the smoke test, environment variables, interpret perf numbers, diagnose regressions.
- **[requirements-known-good.txt](docs/requirements-known-good.txt)** — pinned pip lockfile for the reproducible working stack.

## Layout

| File | Role |
|---|---|
| `plugin.py` | `vllm.general_plugins` entry point. Patches `CPUModelRunner.load_model` to wrap `model.forward` with `torch.compile(backend="openvino", options={"vllm": True})`. |
| `paged_attention.py` | Custom `torch.ops.openvino.paged_attention` op + FX rewrite that converts vLLM's `auto_functionalized_v2(unified_attention_with_output, ...)` HOP nodes into it, so the OV pytorch frontend translates the call to `PagedAttentionExtension`. |
| `side_channel.py` | At infer time, binds the `__pa__<layer>__<field>` Parameters (KV cache, block tables, past_lens, ...) from `vllm.forward_context.get_forward_context()`. |
| `sampler.py` | Optional OV-fused fast path for `vllm.v1.sample.sampler.Sampler`. |
| `preset.py` | `options["vllm"] = True` mega-preset that expands into per-flag defaults plus OV CPU-config defaults. |
| `compile_hooks.py` / `runtime_hooks.py` | Helpers called from the generic `torchdynamo/compile.py` and `torchdynamo/execute.py` so the standalone `torch.compile(backend="openvino")` path stays free of vLLM-specific code. |
| `tests/test_run.py` | Smoke test: runs the same prompt through vLLM eager and vLLM+OV, compares output text and reports steady-state tok/s. |

## Measured performance

Isolated single-model runs on Llama-3.2-1B-Instruct, TinyLlama-1.1B,
DeepSeek-R1-Distill-Qwen-1.5B, Qwen2.5-{0.5,1.5}B. bf16 weights + KV
cache, 128 decode tokens, `numactl --cpunodebind=0 --membind=0
--taskset -c 0-39`, all backends built against `dddcff2cc70`.

### Greedy (temperature=0)

| Model | Eager | Inductor | vLLM+OV |
|---|---|---|---|
| Llama-3.2-1B | 63.8 | **76.2** | 60.1 |
| TinyLlama-1.1B | 64.2 | **81.3** | 49.1 |
| DeepSeek-R1-1.5B | 38.6 | **49.0** | 45.3 |
| Qwen2.5-0.5B | 70.1 | **98.2** | 83.3 |
| Qwen2.5-1.5B | 38.8 | **50.3** | 45.1 |

Inductor wins every model under greedy. OV forward pass is
memory-bound on the QKV/MLP/lm_head GEMMs and sits behind Inductor's
kernel selection by 8–40%.

### Sampling (temperature=1.0, top_p=0.95, top_k=50)

| Model | Eager | Inductor | vLLM+OV |
|---|---|---|---|
| Llama-3.2-1B | 52.4 | **57.5** | 48.8 |
| TinyLlama-1.1B | 55.8 | **65.7** | 52.8 |
| DeepSeek-R1-1.5B | 31.6 | 35.9 | **37.6** |
| Qwen2.5-0.5B | 50.1 | 57.0 | **60.8** |
| Qwen2.5-1.5B | 31.6 | 34.1 | **37.7** |

Under sampling, OV wins on 3 of 5 models. The OV-fused sampler (see
`sampler.py`, gated at vocab ≥ 100k) recovers most of the torch
`apply_top_k_top_p` cost, so backends without it (Eager, Inductor) pay
a larger sampler-tax:

| Backend | Median tok/s drop, greedy → sampling |
|---|---|
| Eager | −19% |
| Inductor | −27% |
| vLLM+OV | −17% |

## Changes vs `dddcff2cc70` base

Since the `dddcff2cc70` baseline, `vllm_dev` has landed:

**Perf-improving:**
- **OV-fused sampler for vLLM v1** (`2af320b9aa`, wired in `21ac11be7e`,
  gated on vocab in `9dedb29901`): Compiles topk + softmax + Gumbel-max
  as an OV graph. Fires when `top_k ≤ 128`, no logprobs, no per-request
  seed. Skips the O(vocab·log(vocab)) torch sort. Sampler kernel ~25×
  faster than torch baseline on Llama vocab=128k. End-to-end +11–29%
  under sampling on large-vocab models (DeepSeek, both Qwens, Gemma).
  See `VLLM_OV_FAST_SAMPLER` and `OV_FUSED_SAMPLER_MIN_VOCAB`.
- **Q-input Reshape fusion in PA translator** (`431e29654b`): retargets
  the upstream `Reshape` shape input for the Q branch instead of
  emitting a second `Reshape`. −16 Reshape ops/iter on Llama-1B. K/V
  slices left alone (they have a Result consumer that needs rank-3).
- **QKV projection rank-2 stride fix** (`431e29654b`): the executor
  was hardcoding `srcStrides[1]` which is the innermost element stride
  (== 1) for rank-2 activations; use `strides.size()-2` so both rank-2
  `[M, H]` and rank-3 `[B, S, H]` work. Correctness fix, not a speedup,
  but without it vLLM+OV produced garbage.
- **NormalizeVLLMMLP: Gelu activation** (`db83a034b3`): Gemma-3 support
  — the MLP-fusion pattern now matches models that use Gelu instead of
  Swish. Without it Gemma-3 falls back to unfused MLP.
- **NormalizeVLLMQKV: sink Convert past VariadicSplit** (`7b3bb8ccdd`):
  enables QKV fusion on the bf16 → f16 Convert-annotated graphs
  produced by newer vLLM versions.
- **NormalizeVLLMMLP absorbs bf16 narrow-Convert pair** (`73eb29e35b`):
  eliminates the `f32 → bf16 → f32` envelope around fused MLP that
  otherwise adds two Convert nodes and blocks weight-in-place
  reordering.
- **MADV_HUGEPAGE for large PlainTensor allocations** (`431e29654b`):
  best-effort 2 MB huge-page hint for tensors ≥ 2 MB. Reduces TLB
  misses on LLM weight streaming — measurable on machines with THP
  enabled (`echo always > /sys/kernel/mm/transparent_hugepage/enabled`).
- **oneDNN re-enabled for `lm_head` outside the OV-traced graph**
  (`bc4699c2f7`, pre-baseline but relevant): the lm_head GEMM
  (`[hidden, vocab]`, ~500 MB weight) runs via oneDNN's AMX-prepacked
  path, saving ~3–5 ms/step at decode.

**Refactor/hygiene (no perf effect):**
- Move vLLM glue into a `vllm/` subpackage (`0aa932c741`) and split
  it further (`f28a4bbe9b`, `c64f88ca4f`, `9b9bfd6925`, `1059c0e7fe`,
  `3029753de4`).
- Per-layer PA rt_info + per-layer block_indices / KV geometry
  (`fcad67382f`, `df498641d5`) — required for hybrid attention
  (Gemma-3/4), no effect on uniform-attention models.
- Reverts of experimental PT-frontend workarounds (`b95511a2e0`,
  `b463e6efa9`).

### 1. Fresh venv

```bash
python3.11 -m venv ~/ov_vllm_env
source ~/ov_vllm_env/bin/activate
python -m pip install -U pip setuptools wheel
```

### 2. PyTorch (CPU-only)

Install this FIRST so vLLM picks it up rather than pulling CUDA-enabled
torch during its own resolve.

```bash
python -m pip install "torch==2.11.0+cpu" \
    --index-url https://download.pytorch.org/whl/cpu
```

### 3. vLLM (CPU) — use the known-good lockfile

Install the pinned requirements from this directory. This is required —
see the "First-infer hang" warning at the top of this file. Loose
constraints (`vllm-cpu==0.25.0` alone) resolve to different dep
versions over time and some of those combinations hang.

```bash
python -m pip install -r requirements-known-good.txt \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu
```

The lockfile pins all ~200 packages that reproduced a working stack
(SPR, bf16, 66 tok/s TinyLlama). Notably it pins `apache-tvm-ffi==0.1.9`
(later versions trigger a `__cxa_thread_atexit → posix_memalign`
lock-contention with OV's intel_cpu plugin allocator) and torchvision
`0.26.0+cpu` / torchaudio `2.11.0+cpu` (non-`+cpu` variants fail to
load with `libnvrtc.so.13: cannot open shared object file`).

Verified reproducer: `cp -a` the known-working `venv/` tree to a new
location and rewrite hardcoded paths in `bin/activate*`. This runs the
smoke test at 66 tok/s. A fresh `pip install "vllm-cpu==0.25.0"` today
does NOT.

If you don't have this lockfile handy, install `vllm-cpu==0.25.0` then
downgrade `apache-tvm-ffi` explicitly:

```bash
python -m pip install "vllm-cpu==0.25.0"
python -m pip install --force-reinstall "apache-tvm-ffi==0.1.9"
python -m pip install --force-reinstall \
    "torch==2.11.0+cpu" "torchvision==0.26.0+cpu" "torchaudio==2.11.0+cpu" \
    --index-url https://download.pytorch.org/whl/cpu
python -m pip uninstall -y torchcodec
```

But this shortcut has NOT been verified end-to-end — apache-tvm-ffi
alone was not the whole cause in our fresh-env test. Use the lockfile.

Optional (perf, Linux):

```bash
# Preload tcmalloc before Python starts for a cleaner CPU allocator profile.
# WARNING: some vLLM 0.25.x releases have a `free(): invalid size` crash
# under tcmalloc; if you hit it, unset LD_PRELOAD and use MALLOC_ARENA_MAX=4
# as a fallback.
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4
```

### 4. OpenVINO

Option A — build from the known-good commit `dddcff2cc70` on `vllm_dev`
(required for this integration until the changes merge to a released
wheel; see "Known good build" above for why not the tip):

```bash
git clone --recursive https://github.com/ynimmaga/openvino.git
cd openvino
git checkout dddcff2cc70
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
cmake --build . -j $(nproc)

# Build the wheel via the ie_wheel target, then install it
cmake --build . --target ie_wheel -j $(nproc)
python -m pip install --force-reinstall ./wheels/openvino-*.whl
```

Option B — install a published wheel that already contains this
subpackage:

```bash
python -m pip install "openvino>=2026.2"
```

### 5. Verify the entry point is registered

```bash
python -c "
import importlib.metadata as md
eps = md.entry_points(group='vllm.general_plugins')
for e in eps:
    print(e.name, '->', e.value)
"
```

You should see `openvino_vllm_cpu -> openvino.frontend.pytorch.torchdynamo.vllm.plugin:register`.
If it's missing, the OV install did not include this subpackage; rebuild
the OpenVINO Python wheel from the branch that contains this PR.
