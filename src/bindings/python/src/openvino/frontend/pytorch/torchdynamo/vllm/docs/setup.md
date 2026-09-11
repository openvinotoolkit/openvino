# Setup — vLLM+OV CPU backend

Step-by-step install for the OpenVINO backend to vLLM on CPU. Once
setup is done, see [test.md](test.md) for how to run the smoke test,
what environment variables control behavior, and how to interpret
output.

No source patches are required. `vllm-cpu` pins its own PyTorch
exactly, so the install is one `pip install` plus an OpenVINO wheel.

## Tested environment

The perf numbers in the [main README](../README.md) were measured on
this stack:

| Component | Version | Notes |
|---|---|---|
| CPU | Intel Xeon Platinum 8580 (Sapphire Rapids) | AMX-BF16, 40 cores per socket |
| OS | Linux 5.15+ / glibc 2.34+ | THP=madvise |
| Python | 3.11 | 3.10 also works |
| vLLM | `vllm-cpu` 0.28.0 | v1 engine; CPU wheel |
| PyTorch | 2.13.0+cpu | pulled in by `vllm-cpu`, do not pin separately |
| OpenVINO | 2026.5.0 | `vllm_dev` branch until this integration merges |

### Supported vLLM versions

All four are exercised on this branch and produce identical output.
`vllm-cpu` declares an exact `torch==` pin, so you never choose a
PyTorch version yourself:

| `vllm-cpu` | PyTorch pulled in | `compilation_config` key |
|---|---|---|
| 0.25.0 | 2.11.0+cpu | `mode` (0.25.x also accepted `level=3`) |
| 0.26.0 | 2.11.0+cpu | `mode` |
| 0.27.1 | 2.13.0+cpu | `mode` |
| 0.28.0 | 2.13.0+cpu | `mode` |

Newer is recommended. There is no vLLM release that uses torch 2.12 —
0.26 → 0.27 jumps straight from 2.11 to 2.13.

## 1. Fresh venv

```bash
python3.11 -m venv ~/ov_vllm_env
source ~/ov_vllm_env/bin/activate
python -m pip install -U pip setuptools wheel
```

## 2. vLLM (CPU)

One command. The PyTorch CPU index has to be reachable because
`vllm-cpu` pins the `+cpu` local-version build of torch; PyPI alone
cannot satisfy it.

```bash
python -m pip install "vllm-cpu==0.28.0" \
    --extra-index-url https://download.pytorch.org/whl/cpu
```

Do **not** install torch first and do not pass `--index-url` (which
*replaces* PyPI rather than adding to it) — `vllm-cpu` resolves its own
exact torch pin, and forcing a different one only creates a mismatch.

Optional (perf, Linux):

```bash
# Preload tcmalloc before Python starts for a cleaner CPU allocator profile.
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4
```

## 3. OpenVINO

Option A — build from source (recommended until this integration
merges to a released wheel):

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

Option B — install a published wheel that already contains this
subpackage (once `vllm_dev` merges upstream):

```bash
python -m pip install "openvino>=2026.5"
```

## 4. Verify the entry point is registered

```bash
python -c "
import importlib.metadata as md
for e in md.entry_points(group='vllm.general_plugins'):
    print(e.name, '->', e.value)
"
```

You should see `openvino_vllm_cpu -> openvino.frontend.pytorch.torchdynamo.vllm.plugin:register`.
If it's missing, the OV install did not include this subpackage;
rebuild the OpenVINO Python wheel from the branch that contains this PR.

## Selecting the OV backend at runtime

The plugin auto-loads via the `vllm.general_plugins` entry point. To
select the OV backend, pass `compilation_config` to `LLM`:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    dtype="bfloat16",          # "float16" and "float32" also work
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
(`compilation_config.backend == "openvino"`) returns False and vLLM
uses its default backend; OV is not engaged.

### Precision

You do not configure precision. The backend reads the model's own float
dtype off the converted graph and derives `INFERENCE_PRECISION_HINT`
and `KV_CACHE_PRECISION` from it, then redeclares the PagedAttention
KV-cache Parameters to match. Both keys always name the same type,
because the OV CPU PagedAttention kernel is only instantiated for
matching (compute, cache) pairs.

`bfloat16`, `float16` and `float32` models all run. bf16 is the compute
type in every case: it is exact for bf16 models, and for f16 models it
avoids an overflow in vLLM's unfused RMSNorm, whose `x**2` reduction
exceeds f16's 65504 ceiling and returns NaN (bf16 has f32's exponent
range, so it cannot). To override, set `OV_INFERENCE_PRECISION_HINT`
and `OV_KV_CACHE_PRECISION` — set **both**, to the same value, or
`compile_model` will reject the pair.

## Troubleshooting

- **`KeyError: 'openvino'` from torch.compile**: the OV pytorch
  frontend was not installed or the entry-point file
  `entry_points.txt` doesn't list
  `[torch_dynamo_backends] openvino = openvino.frontend.pytorch.torchdynamo.backend:openvino`.
  Reinstall the OV Python wheel.
- **`ValueError: Field 'level' not found in CompilationConfig`**:
  You are on a newer vLLM that renamed `level` to `mode`. Use
  `{"mode": "STOCK_TORCH_COMPILE", "backend": "openvino"}` (as shown
  above). vLLM 0.25.x used `level=3`; 0.26+ uses `mode`.
- **`expect kvcache type bf16, current: f16`** (or any other pair) from
  `compile_model`: you set only one of `OV_INFERENCE_PRECISION_HINT` /
  `OV_KV_CACHE_PRECISION`, or set them to different types. Set both to
  the same value, or unset both and let the backend derive them.
- **First `generate()` hangs at `Processed prompts: 0%`,
  `shm_broadcast.py:705` warns "No available shared memory broadcast
  block found in 60 seconds"**: the worker is stuck inside
  `posix_memalign` / TBB / glibc heap-lock contention during the very
  first OV infer. This was a rank-4 KV-cache layout bug, fixed in
  `side_channel.py` (`e9cfaabc28`) — rebuild OpenVINO from a current
  `vllm_dev`. To confirm before rebuilding, set
  `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800` and `py-spy dump --native`
  the worker; the stack pins in `libopenvino_intel_cpu_plugin.so`.
- **`free(): invalid size` under `LD_PRELOAD=libtcmalloc`**: seen on
  some vLLM 0.25.x releases. Unset `LD_PRELOAD` and use
  `MALLOC_ARENA_MAX=4` instead.
- **`libnvrtc.so.13: cannot open shared object file`**: a CUDA-flavored
  torch or torchvision was resolved. Reinstall with
  `--extra-index-url https://download.pytorch.org/whl/cpu` and check
  `python -c "import torch; print(torch.__version__)"` ends in `+cpu`.
- **Fused sampler didn't fire (grep `[OV plugin] Fused sampler
  compiled` in stderr)**: the vocab-size gate (default 100000) skips
  small-vocab models. Override with `OV_FUSED_SAMPLER_MIN_VOCAB=0` if
  you want it on for all models.

## Limitations

- CPU-only. The OV GPU/NPU paths are not exercised by this integration.
- `compilation_config["mode"]` must be `STOCK_TORCH_COMPILE`.
  `VLLM_COMPILE` mode invokes vLLM's own `torch.compile` wrapper and
  would double-wrap.
- f16 models execute with bf16 compute, not f16 (see Precision above).
- Speculative decoding, beam search, grammar-constrained decoding,
  custom logit processors, and `logprobs > 0` use the slower vLLM
  Python sampler fallback (the OV-fused sampler eligibility check
  rejects these).
- Continuous batching, prefix caching, paged attention, and tensor
  parallelism are unaffected and continue to work.
