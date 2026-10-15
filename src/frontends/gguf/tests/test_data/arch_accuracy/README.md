# Native architecture accuracy references

Each decoder NPZ contains a small F32 GGUF (`model`, uint8 bytes) and three complete last-token
logit vectors (`logits`, float32). `gen_arch_accuracy.py` creates the weights;
`architecture_oracle.cpp` evaluates them with the real llama.cpp CPU backend.
Embedding fixtures contain `embeddings` instead of `logits`, and cover causal/bidirectional
per-token output plus mean, first-token and last-token pooling.
OpenVINO does not participate in reference generation.

The 40 decoder fixtures and five embedding fixtures include Muse Glimmer, Qwen3.5 dense/MoE (separate and fused
expert projections, and F16 Gated-DeltaNet gate/beta projections that disable the frontend's
projection merges as mixed quantization does in real checkpoints), Gemma4 mixed-head MQA/MoE variants and an E2B/E4B-style Gemma4 with
per-layer embeddings and shared KV layers that omit unused K/V weights (`gemma4-ple`). Reference revisions are:

| Fixtures | Upstream llama.cpp CPU oracle |
|---|---|
| Decoder/embedding fixtures other than the Mamba cases below | `03fa73cb27f5c251b9528489b18d303b1366aca4` (2026-09-08; includes Muse Glimmer) |
| `mamba2`, `mamba2-tied`, `nemotron_h` | `476c01efe88aad7880a8132d5d3a415f2ca75139`, recorded when these fixtures were added |

The cases use distinct nonzero weights and nonuniform norm scales, four query heads,
grouped-query attention (single KV head for Gemma), and token batches `[1,2,3]`, `[4]`,
`[5,6]`. Decoder requests preserve state across all three batches. Embedding fixtures evaluate `[1,2,3]`
without caches; their tests also change input lengths on the same request.
Gemma2 and Muse Glimmer cross a two-token sliding window; SmolLM3 exercises its fourth,
NoPE layer. ERNIE omits `expert_shared_count`, as the real 21B checkpoint does. Bailing
covers sigmoid routing, biased selection, group filtering, and a shared expert. EXAONE MoE
also covers sigmoid routing with scale 2.5 and trailing NextN tensors; MiniMax and Hunyuan
omit expert-normalization metadata to exercise their mandatory normalization defaults.

Devstral adds three configurations under the existing `llama` and `mistral3` families.
Small models use unequal embedding/query widths. Devstral Small 2 crosses reduced original-context
boundaries at positions 2 and 4 to exercise attention temperature; Devstral 2 uses non-default
YaRN correction parameters. Each runs through the native frontend. Devstral 2's real
123B checkpoint and full-context workloads are not verified by these small fixtures.

`GGUFArchitectureAccuracy` compiles the native frontend graph on CPU with F32 inference,
F16 KV state and dynamic activation quantization disabled. It checks every logit using
normalized MSE below `1e-5`. Missing references fail the test. These fixtures run in the
regular frontend suite, including offline CI; llama.cpp is only needed to regenerate them.

To regenerate selected fixtures, check out their revision above and build it with
`GGML_OPENVINO=OFF`. Run from `src/frontends/gguf/tests`, with `LLAMA_SRC` and
`LLAMA_BUILD` pointing to that checkout and its build directory:

```sh
c++ -std=c++17 architecture_oracle.cpp \
    -I "$LLAMA_SRC/include" -I "$LLAMA_SRC/ggml/include" \
    -L "$LLAMA_BUILD/bin" -Wl,-rpath,"$LLAMA_BUILD/bin" \
    -lllama -lggml -lggml-base -o architecture_oracle
c++ -std=c++17 embedding_oracle.cpp \
    -I "$LLAMA_SRC/include" -I "$LLAMA_SRC/ggml/include" \
    -L "$LLAMA_BUILD/bin" -Wl,-rpath,"$LLAMA_BUILD/bin" \
    -lllama -lggml -lggml-base -o embedding_oracle
PYTHONPATH="$LLAMA_SRC/gguf-py" python3 gen_arch_accuracy.py \
    --oracle ./architecture_oracle --embedding-oracle ./embedding_oracle --architectures llama qwen3
```

Choose other names from `CASES` in `gen_arch_accuracy.py`. For the Mamba group, build the oracle
against its recorded revision and use `--architectures mamba2 mamba2-tied nemotron_h`.
Omitting `--architectures` attempts every case with the supplied oracle; use it only for a
deliberate full reference refresh and review the numerical changes and provenance together.

For an additional real-checkpoint check, create `<arch>.gguf` in a separate directory,
then run `architecture_oracle <arch>.gguf <arch>.bin "The capital of France is"`.
Set `OV_GGUF_ACCURACY_DATA` to that directory and filter the test to that architecture.
This performs prefill plus twelve reference-token decode steps, requires the same first
prediction and at least 90% matching greedy choices, and records per-step normalized MSE.
It does not claim exact logit agreement for the frontend's lossy quantized-weight path.
Use `OV_GGUF_Q4_K_ZP_F16=1` for Q4_K accuracy comparisons.

For the embedding checkpoint, put `llama-embed.gguf` and the output of
`embedding_oracle llama-embed.gguf llama-embed.bin "<prompt>"` in a directory and set
`OV_GGUF_EMBEDDING_DATA` to it; `GGUFEmbeddingAccuracy` then compares per-token embeddings.
