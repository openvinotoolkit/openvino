# Setup — vLLM+OV CPU backend

Step-by-step install for the OpenVINO backend to vLLM on CPU. Once
setup is done, see [test.md](test.md) for how to run the smoke test,
what environment variables control behavior, and how to interpret
output.

> ⚠️ **Required source patch: KV chunk for vLLM 0.25 rank-4 layout.**
> Without it, OV's side-channel does `kv_cache.unbind(0)` which is
> wrong for vLLM 0.25's `[num_blocks, num_kv_heads, block_size,
> 2*head_size]` layout and the first `generate()` hangs silently
> (`EngineDeadError` / `Processed prompts: 0%`, worker in
> `posix_memalign` per `py-spy --native`).
>
> Location: `vllm/side_channel.py` around `kc, vc = kv_cache.unbind(0)`.
>
> Fix (already committed on `vllm_dev` as `e9cfaabc28`):
>
> ```python
> if kv_cache.ndim == 4:
>     _nb, _hk, _bs, _last = kv_cache.shape
>     _view = kv_cache.view(_nb, _hk, _bs * 2, _last // 2)
>     kc, vc = _view.chunk(2, dim=2)
>     kc = kc.contiguous()
>     vc = vc.contiguous()
> else:
>     kc, vc = kv_cache.unbind(0)
> ```

## Tested environment

The perf numbers in the [main README](README.md) were measured on
this stack:

| Component | Version | Notes |
|---|---|---|
| CPU | Intel Xeon Platinum 8580 (Sapphire Rapids) | AMX-BF16, 40 cores per socket |
| OS | Linux 5.15+ / glibc 2.34+ | THP=madvise |
| Python | 3.11 | 3.10 also works |
| PyTorch | 2.11.0+cpu | CPU-only build; must match vLLM's torch pin |
| vLLM | vllm-cpu 0.25.0 | v1 engine; CPU wheel |
| OpenVINO | 2026.2.0 `vllm_dev` tip | with KV chunk source patch |

Older/newer combinations may work but are not benched. vLLM 0.24.x
targets torch 2.10 and 0.26.x moves to torch 2.12 — do not mix minor
versions across vLLM and PyTorch.

## Verified working configuration

Both work on this branch:

- `vllm_dev` **tip** — WORKS (contains KV chunk fix, native sampler, fast infer)
- `dddcff2cc70` — WORKS (requires KV chunk patch applied manually)

Tip is measurably faster on Llama sampling (+30% vs `dddcff2cc70`,
because the OV-fused sampler landed in `21ac11be7e` / `9dedb29901`).
Use tip unless you have a specific reason to pin.

## 1. Fresh venv

```bash
python3.11 -m venv ~/ov_vllm_env
source ~/ov_vllm_env/bin/activate
python -m pip install -U pip setuptools wheel
```

## 2. PyTorch (CPU-only)

Install this FIRST so vLLM picks it up rather than pulling CUDA-enabled
torch during its own resolve.

```bash
python -m pip install "torch==2.11.0+cpu" \
    --index-url https://download.pytorch.org/whl/cpu
```

## 3. vLLM (CPU) — use the known-good lockfile

Install the pinned requirements from this directory. This is required.
Loose constraints (`vllm-cpu==0.25.0` alone) resolve to different dep
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

## 4. OpenVINO

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
cmake --build . -j $(nproc)

# Build the wheel via the ie_wheel target, then install it
cmake --build . --target ie_wheel -j $(nproc)
python -m pip install --force-reinstall ./wheels/openvino-*.whl
```

Option B — install a published wheel that already contains this
subpackage (once `vllm_dev` merges upstream):

```bash
python -m pip install "openvino>=2026.2"
```

## 5. Verify the entry point is registered

```bash
python -c "
import importlib.metadata as md
eps = md.entry_points(group='vllm.general_plugins')
for e in eps:
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
(`compilation_config.backend == "openvino"`) returns False and vLLM
uses its default backend; OV is not engaged.

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
- **First `generate()` hangs at `Processed prompts: 0%`,
  `shm_broadcast.py:705` warns "No available shared memory broadcast
  block found in 60 seconds"**: worker is stuck inside
  `posix_memalign` / TBB / glibc heap-lock contention during the very
  first OV infer. Verified with `py-spy dump --native`. Cause: KV
  chunk fix not applied (see banner at top). Apply the source patch
  and rebuild. If you already applied it, set
  `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800` and `py-spy` the worker
  to confirm the stack pins in `libopenvino_intel_cpu_plugin.so`.
- **Fused sampler didn't fire (grep `[OV plugin] Fused sampler
  compiled` in stderr)**: the vocab-size gate (default 100000) skips
  small-vocab models. Override with `OV_FUSED_SAMPLER_MIN_VOCAB=0` if
  you want it on for all models.

## Limitations

- CPU-only. The OV GPU/NPU paths are not exercised by this integration.
- `compilation_config["mode"]` must be `STOCK_TORCH_COMPILE`.
  `VLLM_COMPILE` mode invokes vLLM's own `torch.compile` wrapper and
  would double-wrap.
- Speculative decoding, beam search, grammar-constrained decoding,
  custom logit processors, and `logprobs > 0` use the slower vLLM
  Python sampler fallback (the OV-fused sampler eligibility check
  rejects these).
- Continuous batching, prefix caching, paged attention, and tensor
  parallelism are unaffected and continue to work.
