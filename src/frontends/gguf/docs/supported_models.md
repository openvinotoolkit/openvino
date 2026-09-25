# GGUF Frontend — Supported Models

The frontend accepts graphs from two paths:

- **Native GGUF:** builds a graph directly from a `.gguf` file using the architecture catalog.
- **llama.cpp cgraph:** converts a graph constructed by llama.cpp through its OpenVINO backend.

The paths share operation converters, but architecture coverage is validated separately.
Verification applies to a tested checkpoint and configuration; it does not guarantee support
for every model, modality, context length or quantization using the same architecture name.

## llama.cpp cgraph path

Validated architectures and model examples:

| Architecture | Verified model | Notes |
|---|---|---|
| `llama`   | TinyLlama-1.1B-Chat v1.0 (Q4_K_M) | Dense; standard RoPE + GQA. |
| `qwen2`   | Qwen2.5-0.5B-Instruct (Q8_0)      | Dense. |
| `qwen3`   | Qwen3-0.6B (Q8_0)                 | Dense; QK-norm. |
| `qwen3moe`| Qwen3-0.9B-A0.6B (Q4_K_M), Qwen3-4B (Q4_K_M) | Mixture-of-experts (`mul_mat_id`). |
| `olmoe`   | OLMoE-1B-7B-0924-Instruct (Q4_0)  | Mixture-of-experts. |
| `gemma3`  | gemma-3 family                    | Mixed sliding-window / global RoPE. |
| `gemma4`  | gemma-4-E4B-it (Q4_K_M)           | Per-op RoPE (SWA vs global); f16 KV cache. |
| `qwen35`  | Qwen3.5-4B (Q4_K_M)               | Hybrid GatedDeltaNet + full-attention layers; partial-rotary IMROPE; interleaved Q/gate joint projection; f16 KV cache. |

Checkpoint validation for this path covers: `Q2_K`, `Q4_0`, `Q4_K_M`, `Q6_K`,
`Q8_0`. The frontend weight path also handles `Q4_1`, `Q5_0`, `Q5_1` and the F16/F32
paths; these are exercised by the unit tests but have not each been tied to a specific
end-to-end real-model run.

### Verification

```sh
GGML_OPENVINO_DEVICE=CPU GGML_OPENVINO_STATEFUL_EXECUTION=1 \
  llama-completion -m <model>.gguf -p "The capital of France is" -n 12 -no-cnv --no-warmup
```

A run counts as verification only when the output is coherent (e.g. completes
"...is Paris") and consistent with the pure-ggml CPU backend on the same prompt. A model
that loads but emits garbage is **not** counted as supported.

## Native GGUF path

The native architecture catalog is defined in
[`src/builder/arch_registry.cpp`](../src/builder/arch_registry.cpp):

- **Verified** — conversion, compilation and decoding checked against a reference on a real
  checkpoint. Numerical regression coverage is described below.
- **Experimental** — accepted by the registry, but end-to-end support is not established or
  known limitations remain. The frontend emits a warning; acceptance does not guarantee
  correct output.

Without additional architecture registrations, names outside this catalog are rejected at load
time. External definitions and custom-family catalog entries extend the same registry.

### Verified — 25 architectures

| Architecture | Notes |
|---|---|
| `bailingmoe2` | sigmoid routing, biased/grouped expert selection and shared experts |
| `deepseek2-ocr` | language backbone: dense lead layers and shared/routed experts |
| `ernie4_5-moe` | interleaved MoE, biased selection, normalized weights and shared experts |
| `exaone4` | post-norm-only attention and FFN |
| `gemma` | GeGLU and multi-query attention |
| `gemma2` | post-norms, SWA, attention and final logit soft-caps |
| `gemma3` | post-norms and final logit soft-cap |
| `gemma4` | SWA, per-layer embeddings and shared KV |
| `gpt-oss` | MoE, attention sinks, SWA and OAI gated activation |
| `hunyuan-dense` | learned QK-norm after RoPE |
| `llama` | llama-2 / llama-3 |
| `maincoder` | NORMAL RoPE; learned QK-norm after RoPE |
| `mamba2` | Mamba 2 mixer, tied or separate output embeddings; stateful greedy decoding with one sequence |
| `mellum` | MoE with normalized expert weights |
| `minicpm` | NORMAL RoPE; embedding/residual scales and inverse logit scale |
| `mistral3` | dense decoder, NORMAL RoPE |
| `muse-glimmer` | RoPE on SWA layers, attention output gate, pre/post norms |
| `nemotron_h` | dense hybrid Mamba 2 / attention / ReLU-squared FFN; stateful greedy decoding with one sequence |
| `olmoe` | full-width QK-norm and MoE |
| `phi3` | fused QKV |
| `qwen2` | qwen2 / qwen2.5 |
| `qwen3` | QK-norm before RoPE |
| `qwen35` | hybrid GatedDeltaNet + full attention, interleaved M-RoPE; greedy / batch 1 only |
| `qwen3moe` | NEOX RoPE, per-head QK-norm and normalized expert weights |
| `smollm3` | NORMAL RoPE, skipped on every fourth layer |

### Experimental — 7 architectures

| Architecture | Status / limitation |
|---|---|
| `exaone-moe` | no real checkpoint validated |
| `glm4moe` | no real checkpoint validated |
| `hunyuan-moe` | small numerical fixture passes; no real checkpoint validated |
| `jais2` | no real checkpoint validated |
| `llama-embed` | needs embedding-specific numerical and pooling tests; completion is not a suitable test |
| `minimax-m2` | no real checkpoint validated |
| `plamo3` | native builder disagrees with the real checkpoint reference (0/13 matching choices); post-norm tensors without `.weight` are not recognized |

### Numerical regression coverage

[`GGUFArchitectureAccuracy`](../tests/test_arch_accuracy.cpp) contains 26 small, nonzero F32
fixtures covering 21 verified architecture identifiers and experimental `hunyuan-moe`.
Additional model variants exercise YaRN and position-dependent attention scaling under
`llama` and `mistral3`. The suite does not cover `gemma3`, `gemma4`, `gpt-oss` or `qwen35`.

Each fixture compares complete last-token logits with llama.cpp CPU through multi-token
prefill, one-token decode and a two-token cache append. The normalized MSE limit is `1e-5`.
F32 inference, F16 KV cache and disabled activation quantization isolate architecture behavior
from optional CPU approximations.

The fixtures include single-head KV for Gemma, nonuniform QK-norm weights, Gemma2/Muse
sliding-window boundaries, SmolLM3's fourth NoPE layer, ERNIE shared experts without
`expert_shared_count`, and Bailing's sigmoid/grouped routing. Configuration tests also
cover Gemma2-27B's attention scale, EXAONE4's 64-layer SWA defaults and ERNIE's interleaved
dense/MoE schedule. Zero-weight conversion fingerprints provide structural smoke coverage.

### Mamba 2 execution

`GGML_OP_SSM_SCAN` uses the existing `SelectiveSSM` operation. `GGUFMakeStateful` normalizes
convolution caches for OpenVINO's recurrent fusions. Stateful execution supports greedy
decoding with one sequence and resettable convolution/SSM caches.

CPU tests cover prefill, decode and state reset through the stateful frontend path.
GenAI paged integration is not included. Beam search and prefix-cache reuse are not verified.
Mamba 1 and `nemotron_h_moe` are not supported.

Regenerate the Mamba references with `gen_arch_accuracy.py --oracle <oracle>
--architectures mamba2 mamba2-tied nemotron_h`, using
`architecture_oracle.cpp` built against llama.cpp `476c01efe88aad7880a8132d5d3a415f2ca75139`.

### Reference-checked checkpoints

The following checkpoint comparisons use the prompt `The capital of France is`, native
conversion, `GGUFMakeStateful`, `AdaptToGenAI` and CPU compilation.
The reference's next token is fed back for twelve decode steps, allowing comparison on
identical histories after a token choice differs. The table counts matching greedy
choices across prefill and those twelve steps. Q4_K checks use
`OV_GGUF_Q4_K_ZP_F16=1`; inference uses the precision settings above.

| Architecture | Real checkpoint | Matching choices |
|---|---|---|
| `bailingmoe2` | Ling-mini-2.0 Q2_K | 12/13 |
| `deepseek2-ocr` | DeepSeek-OCR-2 Q4_K_M | 13/13 |
| `ernie4_5-moe` | ERNIE-4.5-21B-A3B-PT Q4_K_M | 12/13 |
| `exaone4` | EXAONE-4.0-1.2B Q4_K_M | 13/13 |
| `gemma` | gemma-2b Q4_K_M | 12/13 |
| `gemma2` | gemma-2-2b-it Q4_K_M | 13/13 |
| `hunyuan-dense` | Hunyuan-0.5B-Instruct Q8_0 | 13/13 |
| `llama` | Devstral Small 2507, 24B Q4_K_M | 13/13 |
| `maincoder` | Maincoder-1B Q4_K_M | 13/13 |
| `mamba2` | Mamba2-2.7B Q8_0 | 12/13 |
| `mellum` | Mellum2-12B-A2.5B-Instruct Q4_K_M | 12/13 |
| `mistral3` | Devstral Small 2, 24B Q4_K_M (text) | 13/13 |
| `mistral3` | Ministral-3-3B-Instruct-2512 Q4_K_M | 13/13 |
| `muse-glimmer` | Muse-Glimmer-30B Q4_0 | 13/13 |
| `nemotron_h` | NVIDIA Nemotron-H-8B-Reasoning-128K Q4_K_M | 12/13 |
| `qwen3moe` | Qwen3-0.9B-A0.6B Q4_K_M | 13/13 |
| `smollm3` | SmolLM3-3B Q4_K_M | 12/13 |

The real-model check requires the same first prediction and at least 90% matching
choices. It records full-logit errors but does not apply the F32 fixture tolerance to
lossy weight conversions. This is bounded decoder validation, not a quality benchmark
or a guarantee for every checkpoint, context length, quantization or device. In particular,
default integer-zero-point Q4_K and U8 KV-cache approximations can change output.

Native Q8_0 preserves weight codes and scales. These comparisons disable OpenVINO's dynamic
activation quantization (`DYNAMIC_QUANTIZATION_GROUP_SIZE=0`) and use F32 inference.
llama.cpp's Q8 CPU path quantizes activations. Running llama.cpp with F32 arithmetic on
the same decoded weights, while keeping OpenVINO's settings unchanged, gives 13/13 matching
choices for Mamba2-130M and Mamba2-2.7B. The 130M checkpoint has model-hub smoke coverage but
falls below the agreement threshold against Q8 CPU arithmetic with OpenVINO dynamic
quantization disabled (11/13). Enabling it with group size 32 gives 13/13 matching choices
for both checkpoints against llama.cpp Q8 CPU in stateful execution. Full logits still
differ; their mean normalized error increases despite the improved token agreement.

See [reference generation and reproduction instructions](../tests/test_data/arch_accuracy/README.md).
These references are shipped with the frontend tests; no model download or llama.cpp
build is needed in the regular test run. Real-model hub tests separately exercise
additional checkpoint/quantization combinations in precommit/nightly jobs.

### Runtime limitations

- **`qwen35`: greedy decoding, batch size 1.** Recurrent states have no batch axis and are
  not reordered by `beam_idx`. Beam search, larger batches, prefix caching and PagedAttention
  are unsupported. Verified checkpoints include Qwen3.5-0.8B Q8_0 and
  Ternary-Bonsai-27B Q2_g64.
- **Multimodal models:** a verified language backbone does not establish support for its
  vision or audio components, preprocessing or full application pipeline.
- **Ternary Bonsai packaging:** `Ternary-Bonsai-27B-Q2_0.gguf` uses g128 packing that does
  not match upstream `Q2_0` (g64). Use `Ternary-Bonsai-27B-Q2_g64.gguf`. The frontend
  rejects the mismatched file.

## Extending support

See [adding an architecture](adding_an_architecture.md) for catalog entries and shared decoder
features, and [porting a model from llama.cpp](porting_a_llama_cpp_model.md) for custom builders
and runtime extensions. Verified status requires a real-checkpoint reference comparison;
successful conversion alone is insufficient.
