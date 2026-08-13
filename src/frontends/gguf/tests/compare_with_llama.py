# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Compare the OpenVINO GGUF frontend against native llama.cpp on a GGUF model.
#
# Greedy-decodes a prompt through the model produced by the GGUF frontend
# (the GGUF frontend) and prints the generated text / token ids, so it can be diffed
# against `llama-simple -m model.gguf <prompt>` (which also greedy-decodes).
#
# The frontend's model uses the gguf IO contract (inp_tokens / inp_pos / self_kq_mask /
# token_len_per_seq + a stateful KV cache). This script builds those inputs from a token
# sequence -- i.e. it is a standalone version of the genai IO adapter. The same logic moved
# into the graph (or a genai-side wrapper) lets the model run under genai's LLMPipeline.
#
# Usage:
#   PYTHONPATH=<ov>/bin/intel64/Release/python \
#   LD_LIBRARY_PATH=<ov>/bin/intel64/Release \
#   python3 compare_with_llama.py --gguf model.gguf --tokenizer <hf_dir> \
#       --prompt "The capital of France is" --n 16

import argparse
import sys

import numpy as np
import openvino as ov
from openvino.frontend import FrontEndManager


def convert_gguf(model_path: str):
    """Convert a .gguf through the GGUF frontend.

    The frontend is not auto-selectable (see is_hidden_frontend in
    src/frontends/common/src/manager.cpp), so core.read_model(".gguf") does not reach it and it
    has to be requested by name.
    """
    fe = FrontEndManager().load_by_framework("gguf")
    return fe.convert(fe.load(model_path))


def build_inputs(tokens, past_len):
    """Build the gguf-IO tensors for one decode step.

    tokens   : list[int] of the new token ids for this step.
    past_len : number of tokens already in the KV cache.
    Returns a dict of input name -> ov.Tensor.
    """
    n = len(tokens)
    total = past_len + n
    inp_tokens = np.array(tokens, dtype=np.int32).reshape(1, 1, 1, n)
    inp_pos = np.arange(past_len, past_len + n, dtype=np.int32).reshape(1, 1, 1, n)
    # last-token logits only (matches llama-simple's per-step argmax on the final token)
    inp_out_ids = np.array([n - 1], dtype=np.int32).reshape(1, 1, 1, 1)
    # causal mask [1, 1, n, total]: 0 where attended, -inf where masked.
    mask = np.zeros((1, 1, n, total), dtype=np.float32)
    for i in range(n):
        # query token i (absolute position past_len + i) may attend to keys 0..past_len+i
        allowed = past_len + i + 1
        mask[0, 0, i, allowed:] = -np.inf
    token_len = np.array([n], dtype=np.int64)
    # beam_idx: identity beam reorder for the (single-beam, batch-1) stateful KV cache.
    beam_idx = np.zeros((1,), dtype=np.int32)
    return {
        "inp_tokens": ov.Tensor(inp_tokens),
        "inp_pos": ov.Tensor(inp_pos),
        "inp_out_ids": ov.Tensor(inp_out_ids),
        "self_kq_mask": ov.Tensor(mask),
        # gpt-oss sliding-window mask: for prompts shorter than the window it equals the
        # full causal mask, so the same tensor is correct here.
        "self_kq_mask_swa": ov.Tensor(mask.copy()),
        "token_len_per_seq": ov.Tensor(token_len),
        "beam_idx": ov.Tensor(beam_idx),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tokenizer", required=True, help="HF tokenizer dir/json for the model")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--n", type=int, default=16, help="tokens to generate")
    ap.add_argument("--device", default="CPU")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    prompt_ids = tok(args.prompt, return_tensors="np")["input_ids"][0].tolist()
    print(f"prompt: {args.prompt!r}")
    print(f"prompt token ids: {prompt_ids}")

    core = ov.Core()
    model = convert_gguf(args.gguf)
    compiled = core.compile_model(model, args.device)
    req = compiled.create_infer_request()
    # only feed the inputs the (pruned) model actually exposes
    model_inputs = {n for p in compiled.inputs for n in p.get_names()}

    def run(tokens, past):
        feed = {k: v for k, v in build_inputs(tokens, past).items() if k in model_inputs}
        out = req.infer(feed)
        return list(out.values())[0]

    # ---- prefill ----
    next_id = int(run(prompt_ids, 0).reshape(-1).argmax())
    generated = [next_id]
    past = len(prompt_ids)

    # ---- decode ----
    for _ in range(args.n - 1):
        next_id = int(run([next_id], past).reshape(-1).argmax())
        generated.append(next_id)
        past += 1

    text = tok.decode(generated)
    print(f"\ngenerated token ids: {generated}")
    print(f"generated text: {text!r}")
    print(f"\nfull: {args.prompt}{text}")


if __name__ == "__main__":
    sys.exit(main())
