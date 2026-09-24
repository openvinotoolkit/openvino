# Native architecture accuracy references

Each NPZ contains a small F32 GGUF (`model`, uint8 bytes) and three complete last-token
logit vectors (`logits`, float32). `gen_arch_accuracy.py` creates the weights;
`architecture_oracle.cpp` evaluates them with the real llama.cpp CPU backend.
OpenVINO does not participate in reference generation.

The 32 fixtures, including Muse Glimmer and Qwen3.5 dense/MoE (separate and fused
expert projections) and Gemma4 mixed-head MQA/MoE variants, use the CPU oracle from upstream
[ggml-org/llama.cpp `03fa73cb27f5c251b9528489b18d303b1366aca4`](https://github.com/ggml-org/llama.cpp/commit/03fa73cb27f5c251b9528489b18d303b1366aca4)
(2026-09-08). Muse Glimmer support is included upstream; no fork is needed.

The cases use distinct nonzero weights and nonuniform norm scales, four query heads,
grouped-query attention (single KV head for Gemma), and token batches `[1,2,3]`, `[4]`,
`[5,6]`. The same inference request preserves state across all three batches.
Gemma2 and Muse Glimmer cross a two-token sliding window; SmolLM3 exercises its fourth,
NoPE layer. ERNIE omits `expert_shared_count`, as the real 21B checkpoint does. Bailing
covers sigmoid routing, biased selection, group filtering, and a shared expert.

Devstral adds three configurations under the existing `llama` and `mistral3` families.
Small models use unequal embedding/query widths. Devstral Small 2 crosses reduced original-context
boundaries at positions 2 and 4 to exercise attention temperature; Devstral 2 uses non-default
YaRN correction parameters. Each runs through the native frontend. Devstral 2's real
123B checkpoint and full-context workloads are not verified by these small fixtures.

`GGUFArchitectureAccuracy` compiles the native frontend graph on CPU with F32 inference,
F16 KV state and dynamic activation quantization disabled. It checks every logit using
normalized MSE below `1e-5`. Missing references fail the test. These fixtures run in the
regular frontend suite, including offline CI; llama.cpp is only needed to regenerate them.

To regenerate all fixtures, check out the revision above and build it with
`GGML_OPENVINO=OFF`. Run from `src/frontends/gguf/tests`, with `LLAMA_SRC` and
`LLAMA_BUILD` pointing to that checkout and its build directory:

```sh
c++ -std=c++17 architecture_oracle.cpp \
    -I "$LLAMA_SRC/include" -I "$LLAMA_SRC/ggml/include" \
    -L "$LLAMA_BUILD/bin" -Wl,-rpath,"$LLAMA_BUILD/bin" \
    -lllama -lggml -lggml-base -o architecture_oracle
PYTHONPATH="$LLAMA_SRC/gguf-py" python3 gen_arch_accuracy.py \
    --oracle ./architecture_oracle
```

For an additional real-checkpoint check, create `<arch>.gguf` in a separate directory,
then run `architecture_oracle <arch>.gguf <arch>.bin "The capital of France is"`.
Set `OV_GGUF_ACCURACY_DATA` to that directory and filter the test to that architecture.
This performs prefill plus twelve reference-token decode steps, requires the same first
prediction and at least 90% matching greedy choices, and records per-step normalized MSE.
It does not claim exact logit agreement for the frontend's lossy quantized-weight path.
Use `OV_GGUF_Q4_K_ZP_F16=1` for Q4_K accuracy comparisons.

### Q8_0_C requantization accuracy check

A separate CPU GenAI check compared the default Q8_0_C weight requantization with a
no-requantization build on publisher-precision references for Qwen3.5 0.8B/2B/4B/9B,
Qwen3.6 35B, Qwen3.8 27B, Gemma 4 E2B/E4B/12B/26B/31B and Muse Glimmer 30B. It covered
Q4_0 and Q4_K_M with both PA and SDPA (48 cases), F32 inference, F16 KV cache and disabled
dynamic activation quantization. For this comparison, Q4_K used F16 zero points in both runs
(`OV_GGUF_Q4_K_ZP_F16=1`) to isolate the Q8_0_C change. The references retain publisher
BF16/F32 tensor precision.

All 48 predicted-token sequences and matching-choice counts were identical between builds:
36 cases passed the first-token plus 90% matching-choice criterion, and 12 failed in both
builds. There is no evidence from this checkpoint matrix that Q8_0_C requantization reduces
accuracy. This conclusion is limited to the listed checkpoints and the fixed 13-choice
reference history for `The capital of France is`; it is not a guarantee for other models or
prompts. Q8_0_C requantization is selected by tensor name/type and has no runtime opt-out.

A follow-up run left `OV_GGUF_Q4_K_ZP_F16` unset to exercise the default Q4_K conversion to
integer-zero-point u4 weights. Across the same 48 cases, all predicted-token sequences and
matching-choice counts were identical to the F16-zero-point comparison: 36 passed and the
same 12 remained below threshold. No model degraded under this Q4_K requantization check.
The below-threshold configurations in both runs were Qwen3.5 0.8B Q4_K_M, Qwen3.6 35B
Q4_0, Gemma 4 12B Q4_0, Gemma 4 E2B Q4_0, and Gemma 4 26B Q4_0 and Q4_K_M (each with
both attention backends).
This result is limited to these checkpoints and reference histories; it does not establish
that Q4_K requantization is lossless for arbitrary weights or prompts. The F16-zero-point
override remains available for targeted comparisons that isolate Q4_K requantization.
