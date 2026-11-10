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

| Fixtures | llama.cpp CPU oracle |
|---|---|
| Decoder/embedding fixtures other than the Mamba and Prism cases below | Upstream `03fa73cb27f5c251b9528489b18d303b1366aca4` (2026-09-08; includes Muse Glimmer) |
| `mamba2`, `mamba2-tied`, `nemotron_h` | `476c01efe88aad7880a8132d5d3a415f2ca75139`, recorded when these fixtures were added |
| `bonsai2-hadamard`, `dspark` | [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp/tree/e8fd209c975c80aa7fb36a8423571ff3b1b7e955), `e8fd209c975c80aa7fb36a8423571ff3b1b7e955` |

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

For the Prism fixtures, use the exact fork revision above. From `src/frontends/gguf/tests`,
set `LLAMA_SRC` and `LLAMA_BUILD` to separate source and build directories, then run:

```sh
git clone https://github.com/PrismML-Eng/llama.cpp "$LLAMA_SRC"
git -C "$LLAMA_SRC" checkout e8fd209c975c80aa7fb36a8423571ff3b1b7e955
cmake -S "$LLAMA_SRC" -B "$LLAMA_BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DGGML_OPENVINO=OFF -DGGML_CUDA=OFF -DGGML_NATIVE=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_TOOLS=OFF
cmake --build "$LLAMA_BUILD" --target llama --parallel
c++ -std=c++17 -O2 architecture_oracle.cpp \
    -I "$LLAMA_SRC/include" -I "$LLAMA_SRC/ggml/include" \
    -L "$LLAMA_BUILD/bin" -Wl,-rpath,"$LLAMA_BUILD/bin" \
    -lllama -lggml -lggml-base -o architecture_oracle_prism
c++ -std=c++17 -O2 dspark_oracle.cpp \
    -I "$LLAMA_SRC/include" -I "$LLAMA_SRC/ggml/include" -I "$LLAMA_SRC/src" \
    -L "$LLAMA_BUILD/bin" -Wl,-rpath,"$LLAMA_BUILD/bin" \
    -lllama -lggml -lggml-base -o dspark_oracle
PYTHONPATH="$LLAMA_SRC/gguf-py" python3 gen_arch_accuracy.py \
    --oracle ./architecture_oracle_prism --architectures bonsai2-hadamard
PYTHONPATH="$LLAMA_SRC/gguf-py" python3 gen_dspark_accuracy.py --oracle ./dspark_oracle
```

`bonsai2-hadamard.npz` uses the architecture oracle with explicit Hadamard signs and
Gated-DeltaNet feature ordering. `dspark.npz` uses the DSpark API oracle for noncausal
draft blocks with log-SNR conditioning; it also stores Markov and confidence projection
weights for the head checks. The upstream oracle revision does not supply these Prism contracts.

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

LFM2 fixtures (`lfm2`, `lfm2-tied`, `lfm2-swa`, `lfm2-attention-first`, `lfm2moe`,
`lfm2moe-all-experts`) use the decoder oracle revision `03fa73cb27f5c251b9528489b18d303b1366aca4`
and random seed 20261014. They cover four hybrid layers, four query/two KV heads,
three/four-tap convolution caches, a two-token attention window, dense leading layers,
selection bias and normalized softmax/sigmoid expert weights. The sigmoid case includes
expert-scale metadata that llama.cpp's LFM2 MoE loader ignores.

`afmoe.npz` was extracted from OpenVINO donor `c4a023aeac0302c283295ba241ddee5af5ca4a0b`.
It covers four layers (one dense, three routed/shared MoE), four query/two KV heads,
pre/post RMS norms, QK norm, sigmoid attention gating and biased normalized sigmoid
expert selection. The three cached batches cross a two-token sliding window; every
fourth layer uses global attention without RoPE. The generator uses seed 20260907. Regeneration against the local CPU oracle built
from llama.cpp `4b3951de9c87f66768e0fce4043756cc49efc20b` reproduced the NPZ byte for byte.
