# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Smoke test: vLLM + OpenVINO backend vs vLLM eager.

Greedy decode only (HuggingFace's `do_sample=False`; vLLM expresses it as
`temperature=0`). No sampling code path is exercised: the eager and OV
runs both go through `Sampler.greedy_sample` which is a plain argmax over
logits, so any output divergence is attributable to the model.forward
implementation alone.

Reports for each path:
  1. Output text (must match byte-for-byte under greedy).
  2. Steady-state decode tok/s.

Usage:
  python -m openvino.frontend.pytorch.torchdynamo.vllm.tests.test_run \
      --model meta-llama/Llama-3.2-1B-Instruct \
      --prompt "The capital of France is " \
      --max-new-tokens 64

Both paths share a process so the output comparison is meaningful (same
tokenizer state, same prompt encoding). The OV path is selected by passing
compilation_config={"mode": "STOCK_TORCH_COMPILE", "backend": "openvino"};
the eager path uses enforce_eager=True.

Returns nonzero exit code if the texts diverge; perf numbers print regardless.
"""

import argparse
import os
import sys
import time


# Force CPU platform: this env may have both `vllm` and `vllm-cpu` installed,
# so auto-detection can pick the wrong one. Pre-init the platform to CPU
# before any vLLM import. Mirrors the standard CPU-only vLLM bootstrapping
# documented in vllm/getting_started/installation/cpu/.
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")


def _select_cpu_platform():
    import vllm.platforms as _vp
    from vllm.platforms.cpu import CpuPlatform as _CpuPlatform
    _vp._current_platform = _CpuPlatform()


def _generate(llm, prompt, params):
    out = llm.generate([{"prompt": prompt}], params)
    return out[0].outputs[0]


def _run(label, llm, prompt, max_new_tokens, skip_warmup_tokens):
    from vllm import SamplingParams

    # All runs use temperature=0 (do_sample=False) -> greedy argmax. No
    # sampling code path is exercised in either backend.

    # Warmup (compiles the model; not counted in the steady-state measurement).
    warm_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
        ignore_eos=True,
    )
    _generate(llm, prompt, warm_params)

    # Short run captures fixed overhead + first few tokens (first decode step
    # is often slower).
    short_params = SamplingParams(
        max_tokens=skip_warmup_tokens,
        temperature=0.0,
        ignore_eos=True,
    )
    t0 = time.perf_counter()
    _generate(llm, prompt, short_params)
    t_short = time.perf_counter() - t0

    # Full run.
    full_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
        ignore_eos=True,
    )
    t0 = time.perf_counter()
    out = _generate(llm, prompt, full_params)
    t_full = time.perf_counter() - t0

    steady_tokens = max_new_tokens - skip_warmup_tokens
    steady_time = max(t_full - t_short, 1e-9)
    steady_tps = steady_tokens / steady_time
    print(
        f"[{label}] {max_new_tokens} tokens / {t_full:.2f}s = {max_new_tokens / t_full:.2f} tok/s  "
        f"steady (skip {skip_warmup_tokens}t): {steady_tokens}t / {steady_time:.2f}s = {steady_tps:.2f} tok/s"
    )
    return out.text, steady_tps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Hugging Face model id or local path.",
    )
    parser.add_argument(
        "--prompt",
        default="The capital of France is ",
        help="Generation prompt.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Tokens to generate per pass.",
    )
    parser.add_argument(
        "--skip-warmup-tokens",
        type=int,
        default=5,
        help="First N tokens excluded from steady-state perf measurement.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="Maximum sequence length.",
    )
    parser.add_argument(
        "--mode",
        default="both",
        choices=["both", "eager", "openvino"],
        help="Which path(s) to run.",
    )
    args = parser.parse_args()

    _select_cpu_platform()
    from vllm import LLM

    eager_text = ov_text = None
    eager_tps = ov_tps = None

    if args.mode in ("eager", "both"):
        print(f"[eager] loading {args.model} ...")
        llm = LLM(
            model=args.model,
            dtype=args.dtype,
            enforce_eager=True,
            max_model_len=args.max_model_len,
            distributed_executor_backend="uni",
        )
        eager_text, eager_tps = _run(
            "eager", llm, args.prompt, args.max_new_tokens, args.skip_warmup_tokens
        )
        print(f"[eager] text: {eager_text!r}")
        del llm

    if args.mode in ("openvino", "both"):
        print(f"[openvino] loading {args.model} ...")
        # OV CPU PagedAttention requires block_size=32 (hard kernel constraint).
        # custom_ops=["none"] keeps vLLM from expanding RMSNorm/SiLU into custom
        # CUDA ops that the CPU torch.compile path can't handle.
        llm = LLM(
            model=args.model,
            dtype=args.dtype,
            enforce_eager=False,
            max_model_len=args.max_model_len,
            distributed_executor_backend="uni",
            block_size=32,
            compilation_config={
                "mode": "STOCK_TORCH_COMPILE",
                "backend": "openvino",
                "custom_ops": ["none"],
            },
        )
        ov_text, ov_tps = _run(
            "openvino", llm, args.prompt, args.max_new_tokens, args.skip_warmup_tokens
        )
        print(f"[openvino] text: {ov_text!r}")
        del llm

    if args.mode == "both":
        print()
        print("=" * 60)
        if eager_text == ov_text:
            print("PASS: outputs match byte-for-byte")
        else:
            print("FAIL: outputs differ")
            print(f"  eager:    {eager_text!r}")
            print(f"  openvino: {ov_text!r}")
        speedup = ov_tps / eager_tps if eager_tps else float("inf")
        print(f"steady tok/s   eager={eager_tps:.2f}   openvino={ov_tps:.2f}   speedup={speedup:.2f}x")
        if eager_text != ov_text:
            sys.exit(1)


if __name__ == "__main__":
    main()
