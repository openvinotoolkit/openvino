# GGUF Frontend — Supported Models

This document lists the model architectures the GGUF frontend can convert and run
end-to-end. An architecture is listed as **Supported** only when at least one *real*
(non-synthetic) model of that architecture has been verified to load, convert, and
produce coherent output through the frontend.

Verification is done by running a real `.gguf` through the OpenVINO backend swap in
`llama.cpp` (`llama-completion`, CPU device, stateful execution) and confirming the
generated text is coherent and matches the pure-ggml CPU reference for the same prompt.

## Supported architectures

Each row was verified with the named real model.

| Architecture | Verified model | Notes |
|---|---|---|
| `llama`   | TinyLlama-1.1B-Chat v1.0 (Q4_K_M) | Dense; standard RoPE + GQA. |
| `qwen2`   | Qwen2.5-0.5B-Instruct (Q8_0)      | Dense. |
| `qwen3`   | Qwen3-0.6B (Q8_0)                 | Dense; QK-norm. |
| `qwen3moe`| Qwen3-0.9B-A0.6B (Q4_K_M), Qwen3-4B (Q4_K_M) | Mixture-of-experts (`mul_mat_id`). |
| `olmoe`   | OLMoE-1B-7B-0924-Instruct (Q4_0)  | Mixture-of-experts. |
| `gemma3`  | gemma-3 family                    | Mixed sliding-window / global RoPE. |
| `gemma4`  | gemma-4-E4B-it (Q4_K_M)           | Per-op RoPE (SWA vs global); f16 KV cache. |

`llama`, `qwen2`, `qwen3`, `olmoe`, and `gemma4` were verified with a fresh end-to-end
run of the model named above. `qwen3moe` and `gemma3` were verified in earlier
development on the models named above.

Quantization formats verified in the above runs: `Q2_K`, `Q4_0`, `Q4_K_M`, `Q6_K`,
`Q8_0`. The frontend weight path also handles `Q4_1`, `Q5_0`, `Q5_1` and the F16/F32
paths; these are exercised by the unit tests but have not each been tied to a specific
end-to-end real-model run.

## How verification was performed

```sh
GGML_OPENVINO_DEVICE=CPU GGML_OPENVINO_STATEFUL_EXECUTION=1 \
  llama-completion -m <model>.gguf -p "The capital of France is" -n 12 -no-cnv --no-warmup
```

A run counts as verification only when the output is coherent (e.g. completes
"...is Paris") and consistent with the pure-ggml CPU backend on the same prompt. A model
that loads but emits garbage (e.g. `hunyuan`) is **not** counted as supported.

## Adding a new architecture

Support for a new architecture is a combination of:
1. **Ops** — every ggml op in the model's compute graph must have a frontend translator
   (`src/op/<name>.cpp`) and backend admission.
2. **Weights** — every quantization format used by the model's tensors must be handled by
   the weight path (`src/quant/weights.cpp`).
3. **Real-model verification** — run a real `.gguf` end-to-end as above before adding the
   architecture to the Supported table.
