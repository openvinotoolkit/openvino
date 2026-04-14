# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Measure perplexity of an OpenVINO LLM with different KV cache settings.
#
# Uses autoregressive token-by-token decode so the KV cache codec is
# exercised: each token's attention reads from previously quantized
# KV cache entries stored by Assign nodes.
#
# For each chunk of `window` tokens from WikiText-2:
#   1. Prefill first `prefill` tokens at once (fast, codec-transparent)
#   2. Decode remaining tokens one-by-one (reads quantized KV cache)
#   3. Compute NLL of ground-truth next token at each decode step
#   4. Reset state, advance to next chunk
#
# Usage:
#   source install_RelWithDebInfo/setupvars.sh
#   source openvino.genai/setupvars.sh
#   source openvino.genai/.venv/bin/activate
#
#   # Baseline (f32)
#   python measure_ppl.py --model ~/models/Qwen3-8B/
#
#   # With OV config file
#   python measure_ppl.py --model ~/models/Qwen3-8B/ --ov-config openvino.genai/config_tbq4.json
#
#   # Quick test with limited tokens
#   python measure_ppl.py --model ~/models/Qwen3-8B/ --max-tokens 1000 --window 256

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import openvino as ov
import openvino_genai as ov_genai
from datasets import load_dataset


def tokenize_wikitext(tokenizer, max_tokens=0):
    """Load and tokenize WikiText-2 test set."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])

    encoded = tokenizer.encode(text)
    tokens = encoded.input_ids.data.flatten().astype(np.int64)

    if max_tokens > 0:
        tokens = tokens[:max_tokens]
    return tokens


def log_softmax_target(logits_row, target):
    """Compute log-softmax of logits_row at target index (numerically stable)."""
    row = logits_row.astype(np.float64)
    max_logit = np.max(row)
    log_sum_exp = max_logit + np.log(np.sum(np.exp(row - max_logit)))
    return row[target] - log_sum_exp


def batch_nll(logits_2d, targets):
    """Compute NLL for multiple positions at once. logits_2d: (N, vocab), targets: (N,)."""
    logits = logits_2d.astype(np.float64)
    max_logits = np.max(logits, axis=1, keepdims=True)
    log_sum_exp = max_logits.squeeze(1) + np.log(
        np.sum(np.exp(logits - max_logits), axis=1))
    target_logits = logits[np.arange(len(targets)), targets]
    return np.sum(log_sum_exp - target_logits)


def measure_perplexity(model_path, tokens, window_size, prefill_size,
                       ov_properties):
    """
    Autoregressive perplexity measurement.

    Within each chunk:
    - Prefill: feed first `prefill_size` tokens at once. KV cache is written
      but attention uses f32 K/V directly (no codec read-back).
    - Decode: feed remaining tokens one-by-one. Each step reads from the
      quantized KV cache, so codec error is captured.
    - NLL is accumulated for all positions where we have logits.
    """
    core = ov.Core()

    # Detect model layout: text-only (openvino_model.xml) vs VLM (language + embeddings)
    is_vlm = False
    model_xml = f"{model_path}/openvino_model.xml"
    if not os.path.exists(model_xml):
        model_xml = f"{model_path}/openvino_language_model.xml"
        is_vlm = True

    model = core.read_model(model_xml)
    compiled = core.compile_model(model, "CPU", ov_properties)
    infer_request = compiled.create_infer_request()

    # For VLM models, we need a separate embeddings model to convert input_ids -> inputs_embeds
    embed_request = None
    if is_vlm:
        embed_xml = f"{model_path}/openvino_text_embeddings_model.xml"
        embed_model = core.read_model(embed_xml)
        embed_compiled = core.compile_model(embed_model, "CPU")
        embed_request = embed_compiled.create_infer_request()

    # Check if the language model expects position_ids
    lm_input_names = {inp.any_name for inp in model.inputs}
    has_position_ids = "position_ids" in lm_input_names

    def get_embeddings(token_ids):
        """Convert token IDs to embeddings via the text embeddings model."""
        embed_request.infer({"input": ov.Tensor(token_ids)})
        return embed_request.get_output_tensor(0).data.copy()

    def build_lm_inputs(token_ids, attention_mask, position_ids):
        """Build input dict for the language model, handling both layouts."""
        if is_vlm:
            inputs = {
                "inputs_embeds": ov.Tensor(get_embeddings(token_ids)),
                "attention_mask": attention_mask,
                "beam_idx": ov.Tensor(np.array([0], dtype=np.int32)),
            }
        else:
            inputs = {
                "input_ids": ov.Tensor(token_ids),
                "attention_mask": attention_mask,
                "beam_idx": ov.Tensor(np.array([0], dtype=np.int32)),
            }
        if has_position_ids:
            inputs["position_ids"] = position_ids
        return inputs

    seq_len = len(tokens)
    n_chunks = max(1, seq_len // window_size)

    total_nll = 0.0
    total_tokens = 0
    prefill_nll = 0.0
    prefill_tokens = 0
    decode_nll = 0.0
    decode_tokens = 0
    t0 = time.time()

    config_str = json.dumps(ov_properties) if ov_properties else "(baseline)"
    layout = "VLM (embeddings + language)" if is_vlm else "text-only"
    print(f"\nDataset: {seq_len} tokens")
    print(f"Model layout: {layout}")
    print(f"Window: {window_size}, prefill: {prefill_size}, chunks: {n_chunks}")
    print(f"OV config: {config_str}")
    print()

    for chunk_idx in range(n_chunks):
        begin = chunk_idx * window_size
        end = min(begin + window_size, seq_len)
        chunk_tokens = tokens[begin:end]
        chunk_len = len(chunk_tokens)

        if chunk_len < 2:
            continue

        infer_request.reset_state()

        pf_len = min(prefill_size, chunk_len - 1)
        attn_len = 0

        # --- Phase 1: Prefill ---
        if pf_len > 0:
            infer_request.infer(build_lm_inputs(
                chunk_tokens[:pf_len].reshape(1, -1),
                ov.Tensor(np.ones((1, pf_len), dtype=np.int64)),
                ov.Tensor(np.arange(pf_len, dtype=np.int64).reshape(1, -1)),
            ))
            attn_len = pf_len

            # Score ALL prefill positions: logits[i] predicts token[i+1]
            logits = infer_request.get_output_tensor(0).data.copy()
            if logits.ndim == 3:
                logits = logits[0]
            # logits shape: (pf_len, vocab). Position i predicts token i+1.
            chunk_nll = batch_nll(logits[:pf_len],
                                  chunk_tokens[1:pf_len + 1])
            total_nll += chunk_nll
            total_tokens += pf_len
            prefill_nll += chunk_nll
            prefill_tokens += pf_len

        # --- Phase 2: Autoregressive decode (one token at a time) ---
        for pos in range(pf_len, chunk_len - 1):
            attn_len += 1
            infer_request.infer(build_lm_inputs(
                np.array([[chunk_tokens[pos]]], dtype=np.int64),
                ov.Tensor(np.ones((1, attn_len), dtype=np.int64)),
                ov.Tensor(np.array([[pos]], dtype=np.int64)),
            ))

            logits = infer_request.get_output_tensor(0).data
            if logits.ndim == 3:
                logits = logits[0]
            nll = -log_softmax_target(logits[-1], chunk_tokens[pos + 1])
            total_nll += nll
            total_tokens += 1
            decode_nll += nll
            decode_tokens += 1

        ppl_so_far = math.exp(total_nll / total_tokens)
        elapsed = time.time() - t0
        eta = elapsed / (chunk_idx + 1) * (n_chunks - chunk_idx - 1)
        tps = total_tokens / elapsed
        print(f"  [{chunk_idx + 1}/{n_chunks}] "
              f"PPL={ppl_so_far:.4f} "
              f"({total_tokens} tok, "
              f"{tps:.1f} tok/s, "
              f"ETA {eta:.0f}s)")

    final_ppl = math.exp(total_nll / total_tokens) if total_tokens > 0 else float("inf")
    elapsed = time.time() - t0
    pf_ppl = math.exp(prefill_nll / prefill_tokens) if prefill_tokens > 0 else float("inf")
    dec_ppl = math.exp(decode_nll / decode_tokens) if decode_tokens > 0 else float("inf")
    print(f"\nFinal PPL: {final_ppl:.4f} ({total_tokens} tokens, {elapsed:.1f}s)")
    if prefill_tokens > 0 and decode_tokens > 0:
        print(f"  Prefill PPL: {pf_ppl:.4f} ({prefill_tokens} tokens, no codec read)")
        print(f"  Decode  PPL: {dec_ppl:.4f} ({decode_tokens} tokens, codec-impacted)")
    return {
        "ppl": final_ppl,
        "decode_ppl": dec_ppl,
        "prefill_ppl": pf_ppl,
        "total_tokens": total_tokens,
        "decode_tokens": decode_tokens,
        "prefill_tokens": prefill_tokens,
        "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Measure perplexity of an OpenVINO LLM with KV cache settings")
    parser.add_argument("--model", required=True,
                        help="Path to OpenVINO model directory")
    parser.add_argument("--ov-config", default=None,
                        help="Path to JSON file with OV compile properties "
                             "(e.g. openvino.genai/config_tbq4.json)")
    parser.add_argument("--window", type=int, default=2048,
                        help="Chunk size in tokens (default: 2048)")
    parser.add_argument("--prefill", type=int, default=0,
                        help="Tokens to prefill at once per chunk (default: window-1). "
                             "Prefill scores from a single forward pass (fast). "
                             "Use 1 for full codec-impacted decode, 0 for auto.")
    parser.add_argument("--max-tokens", type=int, default=0,
                        help="Max tokens from dataset (0=all)")
    parser.add_argument("--inference-precision", default=None,
                        help="Inference precision (e.g. f32, bf16). "
                             "Sets INFERENCE_PRECISION_HINT property.")
    args = parser.parse_args()

    if args.prefill == 0:
        args.prefill = args.window - 1

    ov_properties = {}
    if args.ov_config:
        with open(args.ov_config) as f:
            raw = json.load(f)
        genai_only = {"ATTENTION_BACKEND", "SCHEDULER_CONFIG"}
        ov_properties = {k: str(v) for k, v in raw.items() if k not in genai_only}

    if args.inference_precision:
        ov_properties["INFERENCE_PRECISION_HINT"] = args.inference_precision

    print(f"Loading tokenizer from {args.model}...")
    tokenizer = ov_genai.Tokenizer(args.model)

    print("Tokenizing WikiText-2 test set...")
    tokens = tokenize_wikitext(tokenizer, args.max_tokens)
    print(f"  {len(tokens)} tokens")

    measure_perplexity(
        args.model, tokens, args.window, args.prefill, ov_properties)

    return 0


if __name__ == "__main__":
    sys.exit(main())
