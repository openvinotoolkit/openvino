# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Benchmark a GGUF model on two backends:
#   1. OpenVINO GGUF frontend  (frontend convert + compile on CPU)
#   2. llama.cpp               (llama-simple binary, CPU only, no GPU layers)
#
# Metrics reported per backend:
#   - load (convert / llama load) time [s]
#   - compile time [s]  (OV only)
#   - prefill throughput [tok/s]
#   - decode throughput [tok/s]
#
# Usage:
#   PYTHONPATH=<ov>/bin/intel64/Release/python \
#   LD_LIBRARY_PATH=<ov>/bin/intel64/Release \
#   python3 bench_gguf.py --gguf model.gguf --tokenizer <hf_dir> \
#       [--llama-simple <path>] [--prompt "..."] [--prefill-tokens 128] [--gen-tokens 64]
#
# llama-simple is expected to print on stderr:
#   "prompt eval time = ... / N tokens (...  ms per token, PP tok/s tokens per second)"
#   "eval time = ... / N runs   (... ms per token, TG tok/s tokens per second)"

import argparse
import re
import subprocess
import sys
import time

import numpy as np
import openvino as ov

from gguf_io_utils import KVCache, build_inputs, convert_gguf


# ---------------------------------------------------------------------------
# OpenVINO helpers
# ---------------------------------------------------------------------------

def bench_openvino(gguf_path, prompt_ids, gen_tokens, device="CPU"):
    core = ov.Core()

    t0 = time.perf_counter()
    model = convert_gguf(gguf_path)
    t_read = time.perf_counter() - t0

    t0 = time.perf_counter()
    compiled = core.compile_model(model, device)
    t_compile = time.perf_counter() - t0

    req = compiled.create_infer_request()
    model_inputs = {n for p in compiled.inputs for n in p.get_names()}
    kv_cache = KVCache(compiled)

    def run(tokens, past):
        raw = build_inputs(tokens, past)
        feed = {}
        for k, v in raw.items():
            if k in model_inputs:
                feed[k] = ov.Tensor(v)
        feed.update({k: v for k, v in kv_cache.inputs_for_step(len(tokens)).items() if k in model_inputs})
        req.infer(feed)
        kv_cache.update_from(req)
        return np.array(req.get_output_tensor(0).data)

    # warm-up: 1-token prefill + 1 decode
    run(prompt_ids[:1], 0)
    run([0], 1)
    # reset the (Python-side) KV cache so the timed run below starts from an empty context.
    kv_cache = KVCache(compiled)

    # prefill
    t0 = time.perf_counter()
    next_id = int(run(prompt_ids, 0).reshape(-1).argmax())
    t_prefill = time.perf_counter() - t0
    pp_tps = len(prompt_ids) / t_prefill

    # decode
    past = len(prompt_ids)
    t0 = time.perf_counter()
    for _ in range(gen_tokens - 1):
        next_id = int(run([next_id], past).reshape(-1).argmax())
        past += 1
    t_decode = time.perf_counter() - t0
    tg_tps = (gen_tokens - 1) / t_decode

    return {
        "load_s": t_read,
        "compile_s": t_compile,
        "pp_tps": pp_tps,
        "tg_tps": tg_tps,
        "prefill_tokens": len(prompt_ids),
        "gen_tokens": gen_tokens,
    }


# ---------------------------------------------------------------------------
# llama.cpp helpers
# ---------------------------------------------------------------------------

def bench_llamacpp(llama_simple, gguf_path, prompt_text, gen_tokens):
    """Run llama-simple and parse its timing lines from stderr."""
    import os
    env = dict(os.environ)
    # ensure the llama.cpp libs are found
    lib_dir = str(__import__("pathlib").Path(llama_simple).parent)
    old = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{lib_dir}:{old}" if old else lib_dir

    cmd = [
        llama_simple,
        "-m", gguf_path,
        "-n", str(gen_tokens),
        prompt_text,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    combined = result.stdout + result.stderr

    # llama_perf output lines:
    #   "llama_perf_context_print:        load time =   167.53 ms"
    #   "llama_perf_context_print: prompt eval time =    37.42 ms /   N tokens (...  PP tok/s tokens per second)"
    #   "llama_perf_context_print:        eval time =   251.15 ms /   N runs   (...  TG tok/s tokens per second)"
    load_ms = _parse_perf_line(combined, "load time")
    pp_tps = _parse_tps(combined, "prompt eval time")
    tg_tps = _parse_tps(combined, r"        eval time")

    return {
        "load_s": (load_ms / 1000) if load_ms else None,
        "compile_s": None,
        "pp_tps": pp_tps,
        "tg_tps": tg_tps,
        "gen_tokens": gen_tokens,
        "returncode": result.returncode,
        "stderr": combined,
    }


def _parse_perf_line(text, label):
    m = re.search(rf"{re.escape(label)}\s*=\s*([\d.]+)\s*ms", text)
    return float(m.group(1)) if m else None


def _parse_tps(text, label_pattern):
    m = re.search(rf"{label_pattern}.*?([\d.]+)\s+tokens per second", text)
    return float(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tokenizer", required=True, help="HF tokenizer dir")
    ap.add_argument("--prompt", default="The capital of France is Paris. Tell me about this city.")
    ap.add_argument("--prefill-tokens", type=int, default=128,
                    help="Pad/truncate prompt to this many tokens")
    ap.add_argument("--gen-tokens", type=int, default=64)
    ap.add_argument("--llama-simple",
                    default="/home/vmaxim/llama.cpp/build-ref/bin/llama-simple")
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--skip-ov", action="store_true")
    ap.add_argument("--skip-llama", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    prompt_ids = tok(args.prompt, return_tensors="np")["input_ids"][0].tolist()

    # pad / trim to prefill_tokens
    if len(prompt_ids) < args.prefill_tokens:
        # repeat prompt until we hit target
        while len(prompt_ids) < args.prefill_tokens:
            prompt_ids = prompt_ids + prompt_ids
    prompt_ids = prompt_ids[:args.prefill_tokens]

    print(f"Model : {args.gguf}")
    print(f"Prompt: {args.prefill_tokens} tokens (prefill), {args.gen_tokens} tokens (gen)")
    print()

    ov_res = None
    lc_res = None

    if not args.skip_ov:
        print("=== OpenVINO GGUF frontend ===")
        ov_res = bench_openvino(args.gguf, prompt_ids, args.gen_tokens, args.device)
        print(f"  convert     : {ov_res['load_s']:.2f}s")
        print(f"  compile     : {ov_res['compile_s']:.2f}s")
        print(f"  prefill     : {ov_res['pp_tps']:.1f} tok/s  ({ov_res['prefill_tokens']} tokens)")
        print(f"  decode      : {ov_res['tg_tps']:.1f} tok/s  ({ov_res['gen_tokens']} tokens)")
        print()

    if not args.skip_llama:
        print("=== llama.cpp (CPU) ===")
        lc_res = bench_llamacpp(
            args.llama_simple,
            args.gguf,
            args.prompt,
            args.gen_tokens,
        )
        if lc_res["load_s"]:
            print(f"  load        : {lc_res['load_s']:.2f}s")
        if lc_res["pp_tps"]:
            print(f"  prefill     : {lc_res['pp_tps']:.1f} tok/s")
        else:
            print(f"  prefill     : N/A (1-token prompt in llama-simple)")
        if lc_res["tg_tps"]:
            print(f"  decode      : {lc_res['tg_tps']:.1f} tok/s  ({lc_res['gen_tokens']} tokens)")
        if lc_res["returncode"] != 0:
            print(f"  [WARN] llama-simple returned {lc_res['returncode']}")
        print()

    if ov_res and lc_res and lc_res["tg_tps"]:
        ratio = ov_res["tg_tps"] / lc_res["tg_tps"]
        print(f"=== Summary ===")
        print(f"  OV decode   : {ov_res['tg_tps']:.1f} tok/s")
        print(f"  llama decode: {lc_res['tg_tps']:.1f} tok/s")
        print(f"  OV / llama  : {ratio:.2f}x")


if __name__ == "__main__":
    sys.exit(main())
