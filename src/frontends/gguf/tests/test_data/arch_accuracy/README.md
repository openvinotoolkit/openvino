# Native architecture accuracy references

Each NPZ contains a small F32 GGUF (`model`, uint8 bytes) and three complete last-token
logit vectors (`logits`, float32). `gen_arch_accuracy.py` creates the weights;
`architecture_oracle.cpp` evaluates them with the real llama.cpp CPU backend.
OpenVINO does not participate in reference generation.

The original 23 fixtures, including Muse Glimmer, reproduce byte-for-byte with upstream
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
It does not claim exact logit agreement across different runtime arithmetic. Native Q8_0
preserves the original integer codes and F16 scales, but the llama.cpp CPU Q8_0 matmul
also quantizes activations, whereas this OpenVINO configuration disables that step.
Use `OV_GGUF_Q4_K_ZP_F16=1` for Q4_K accuracy comparisons.

## Mamba 2 verification

The three additional fixtures (`mamba2`, `mamba2-tied`, `nemotron_h`) were generated with
[llama.cpp `476c01efe88aad7880a8132d5d3a415f2ca75139`](https://github.com/ggml-org/llama.cpp/commit/476c01efe88aad7880a8132d5d3a415f2ca75139).
The same revision supplies the CPU reference for the checkpoints below. The fixtures exercise
grouped B/C projections, convolution and SSM state updates, tied embeddings and the hybrid
attention/FFN schedule. `GGUFMambaPagedAccuracy` also runs these references through the actual
paged CPU kernels, with independent state rows for two packed sequences of unequal length.
`GGUFArchitectureAccuracy` checks state reset and replay for both architectures.

Real checkpoints checked on 2026-09-15:

| Test case | Repository / file | Pinned revision | SHA-256 |
|---|---|---|---|
| `mamba2` | [Nicholas55555/mamba2-2.7b-Q8_0-GGUF](https://huggingface.co/Nicholas55555/mamba2-2.7b-Q8_0-GGUF), `mamba2-2.7b-q8_0.gguf` | `2dfe52573da68dff5f01996d5ed25fccebf61a06` | `9201a17cad8fa4e084bdf132ebf44f206137ba9c0d03ec903ccd261055a795bf` |
| `nemotron_h` | [bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF](https://huggingface.co/bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF), `nvidia_Nemotron-H-8B-Reasoning-128K-Q4_K_M.gguf` | `cf45b6ff44dcd5c2105a23abeb9e77797f1a4014` | `312ebd50999058707868f49011d87225fcc92af0f5a526cd3553f821dc161956` |

Download those revisions, verify their hashes, and place them as `mamba2.gguf` and
`nemotron_h.gguf` in a separate `mamba_references` directory. Build `architecture_oracle`
as shown above, then run:

```sh
./architecture_oracle mamba_references/mamba2.gguf mamba_references/mamba2.bin "The capital of France is"
./architecture_oracle mamba_references/nemotron_h.gguf mamba_references/nemotron_h.bin "The capital of France is"
OV_GGUF_ACCURACY_DATA="$PWD/mamba_references" OV_GGUF_Q4_K_ZP_F16=1 \
  ./ov_gguf_frontend_tests \
  --gtest_filter='*GGUFArchitectureAccuracy.*/*mamba2:*GGUFMambaPagedAccuracy.*/*mamba2:*Accuracy.*/*nemotron_h' \
  --gtest_output=xml:mamba_verification.xml
```

Both models match the first prediction and 12/13 greedy choices on identical reference
histories, in both stateful and paged execution. The test's existing 90% criterion is unchanged.
CPU settings are F32 inference, disabled dynamic activation quantization, and F16 KV cache.
For Q4_K, the environment setting retains the original fractional zero point. This is not a
claim about default low-precision inference, every checkpoint/quantization, or GPU execution.

The smaller [Mamba2-130M Q8_0 checkpoint](https://huggingface.co/rpatel622/mamba2-130m-hf-Q8_0-GGUF)
(revision `0daf70963405439f2102f6fefe92d7584e2c76eb`, `mamba2-130m-q8_0.gguf`) matches
11/13 choices against the Q8_0 CPU reference. It remains a lightweight model-hub state smoke
test; it does not pass the token-agreement criterion against that reference configuration.

An additional controlled comparison isolates activation quantization for both Q8_0 models.
OpenVINO still loads the original Q8_0 file. Only the llama.cpp reference uses weights decoded
with `llama-quantize --allow-requantize input.gguf output.gguf F32`, preserving the represented
weight values while selecting F32 matmul arithmetic. Both runtimes replay the original Q8_0
reference's identical token batches at every step, even when greedy predictions differ.

| Q8_0 checkpoint | Greedy matches against Q8_0 arithmetic | Greedy matches against F32 arithmetic | Maximum per-step logit NMSE against F32 arithmetic | Maximum absolute logit difference |
|---|---|---|---|---|
| Mamba2-130M | 11/13 | 13/13 | `1.19e-10` | `2.68e-4` |
| Mamba2-2.7B | 12/13 | 13/13 | `1.81e-10` | `2.31e-4` |

The logit measurements above use stateful OpenVINO CPU execution. Both stateful and paged
execution match 13/13 greedy choices against the F32 reference. The 130M prefill comparison
also checks all 24 mixer outputs, with maximum normalized MSE below `6e-12`. These results
attribute the larger Q8_0-reference discrepancy to activation quantization, not loss of the
native Q8_0 weight representation. Small floating-point differences remain; the measurements
cover the tested histories and do not establish bitwise equality or all-prompt token agreement.
