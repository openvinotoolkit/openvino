# OpenVINO backend for vLLM (CPU)

This subpackage wires the OpenVINO `torch.compile` backend into vLLM's CPU
model runner. With it, vLLM's per-step `model.forward()` is dispatched to a
pre-compiled OpenVINO graph instead of PyTorch eager / `torch.compile +
inductor`, while vLLM keeps owning scheduling, paged attention, batching,
sampling, etc.

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

## Setup

The integration assumes a recent OpenVINO build with this PR applied and a
matching CPU-only vLLM install. SPR / Granite Rapids / similar AMX-bf16 CPUs
are the primary target.

### Tested environment

The tables and perf numbers below were measured on this stack:

| Component | Version | Notes |
|---|---|---|
| CPU | Intel Xeon Platinum 8580 (Sapphire Rapids) | AMX-BF16, 40 cores per socket |
| OS | Linux 5.15+ / glibc 2.34+ | THP=madvise |
| Python | 3.11 | 3.10 also works |
| PyTorch | 2.11.0+cpu | CPU-only build; MUST match vLLM's torch pin |
| vLLM | 0.25.0 | v1 engine; CPU wheel |
| OpenVINO | 2026.2.0 (branch `vllm_dev`) | must include this subpackage |

Older/newer combinations may work but are not benched. In particular vLLM
0.24.x targets torch 2.10 and 0.26.x moves to torch 2.12 — do not mix minor
versions across vLLM and PyTorch.

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
python -m pip install "torch==2.11.*+cpu" \
    --index-url https://download.pytorch.org/whl/cpu
```

### 3. vLLM (CPU)

```bash
# Recent vLLM CPU wheels ship on PyPI and don't require CUDA.
python -m pip install "vllm==0.25.*"
```

If the resolver tries to reinstall CUDA torch, use
`--no-deps` for vllm and install the runtime deps manually, or use vLLM's
official CPU installation guide:
https://docs.vllm.ai/en/latest/getting_started/installation/cpu/

Optional (perf, Linux):

```bash
# Preload tcmalloc before Python starts for a cleaner CPU allocator profile.
# WARNING: some vLLM 0.25.x releases have a `free(): invalid size` crash
# under tcmalloc; if you hit it, unset LD_PRELOAD and use MALLOC_ARENA_MAX=4
# as a fallback.
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4
```

### 4. OpenVINO

Option A — build from the `vllm_dev` branch (required for this integration
until the changes merge to a released wheel):

```bash
git clone --recursive https://github.com/openvinotoolkit/openvino.git
cd openvino
git checkout vllm_dev
git submodule update --init --recursive

mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_PYTHON=ON \
      -DENABLE_TESTS=OFF \
      -DENABLE_INTEL_GPU=OFF \
      -DENABLE_INTEL_NPU=OFF \
      -DENABLE_OV_PYTORCH_FRONTEND=ON ..
cmake --build . -j $(nproc)

# Build & install the Python wheel into the venv
cd ..
python -m pip wheel ./src/bindings/python -w /tmp/ov_wheel
python -m pip install --force-reinstall /tmp/ov_wheel/openvino-*.whl
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

## Running the smoke test

```bash
python -m openvino.frontend.pytorch.torchdynamo.vllm.tests.test_run \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --max-new-tokens 64
```

What it does:

1. Loads the model with `enforce_eager=True` (vLLM's eager path), runs
   warmup + a short run + a full run, reports steady-state tok/s and the
   generated text.
2. Loads the same model with `compilation_config={"mode": "STOCK_TORCH_COMPILE", "backend": "openvino"}`,
   repeats, reports tok/s.
3. Compares the two output strings byte-for-byte (greedy decode, so they
   must match) and prints the speedup.

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
  taskset -c 0-23 \
  python -m openvino.frontend.pytorch.torchdynamo.vllm.tests.test_run \
    --model meta-llama/Llama-3.2-1B-Instruct
```

`taskset` keeps everything on one socket; `numactl --membind` keeps memory
local. The OV plugin widens process affinity once at compile time (TBB
samples affinity on first parallel use), so the wide mask only takes effect
after `Core()` is instantiated; pre-pinning to a single socket is still
honored.

## Selecting the OV backend at runtime

The plugin auto-loads via the `vllm.general_plugins` entry point. To select
the OV backend, pass `compilation_config` to `LLM`:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    dtype="bfloat16",
    enforce_eager=False,
    max_model_len=2048,
    compilation_config={
        "mode": "STOCK_TORCH_COMPILE",
        "backend": "openvino",
    },
)
out = llm.generate(["Hello!"], SamplingParams(max_tokens=32, temperature=0.0))
print(out[0].outputs[0].text)
```

Without that `compilation_config`, the plugin's gate
(`compilation_config.backend == "openvino"`) returns False and vLLM uses
its default backend; OV is not engaged.

## Troubleshooting

- **`KeyError: 'openvino'` from torch.compile**: the OV pytorch frontend was not installed or the entry-point file `entry_points.txt` doesn't list `[torch_dynamo_backends] openvino = openvino.frontend.pytorch.torchdynamo.backend:openvino`. Reinstall the OV Python wheel.
- **Output text differs in the smoke test at temperature=0**: a fusion produced semantically wrong code. Re-run with `OV_DISABLE_FUSED_SAMPLER=1` to isolate the sampler.
- **Per-step latency much higher than expected**: enable `OV_PERF_COUNT_OUT=/tmp/ov.log` and inspect counts. `LLMMLP`, `QKVProjection`, `PagedAttentionExtension` should each appear once per layer per decode step. If they're 0 the corresponding fusion did not fire on this model.
- **`ValueError: Field 'level' not found in CompilationConfig`**: You are on a newer vLLM that renamed `level` to `mode`. Use `{"mode": "STOCK_TORCH_COMPILE", "backend": "openvino"}` (as shown above). vLLM 0.25.x used `level=3`; 0.26+ uses `mode`.
- **Fused sampler didn't fire (grep `[OV plugin] Fused sampler compiled` in stderr)**: the vocab-size gate (default: 100000) skips small-vocab models. Override with `OV_FUSED_SAMPLER_MIN_VOCAB=0` if you want it on for all models.

## Limitations

- CPU-only. The OV GPU/NPU paths are not exercised by this integration.
- `compilation_config["mode"]` must be `STOCK_TORCH_COMPILE`. `VLLM_COMPILE`
  mode invokes vLLM's own `torch.compile` wrapper and would double-wrap.
- Speculative decoding, beam search, grammar-constrained decoding, custom
  logit processors, and `logprobs > 0` use the slower vLLM Python sampler
  fallback (the OV-fused sampler eligibility check rejects these).
- Continuous batching, prefix caching, paged attention, and tensor
  parallelism are unaffected and continue to work.
