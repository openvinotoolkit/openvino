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
# token_len_per_seq + a per-layer stateless KV cache, since no DecoderTransformationExtension is
# registered here -- see convert_gguf in gguf_io_utils.py). This script builds those inputs from a
# token sequence and round-trips the KV cache itself (see KVCache) -- i.e. it is a standalone
# version of the genai IO adapter plus the state GGUFMakeStateful would otherwise fold into the
# graph. The same logic moved into the graph (or a genai-side wrapper) lets the model run under
# genai's LLMPipeline.
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

from gguf_io_utils import KVCache, build_inputs, convert_gguf


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
    kv_cache = KVCache(compiled)

    def run(tokens, past):
        feed = {k: ov.Tensor(v) for k, v in build_inputs(tokens, past).items() if k in model_inputs}
        feed.update({k: v for k, v in kv_cache.inputs_for_step(len(tokens)).items() if k in model_inputs})
        req.infer(feed)
        kv_cache.update_from(req)
        return np.array(req.get_output_tensor(0).data)

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
