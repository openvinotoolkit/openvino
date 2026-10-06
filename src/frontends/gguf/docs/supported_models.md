# GGUF frontend supported models

The frontend converts graphs from two paths that share operation converters but are validated
separately:

- **Native GGUF:** builds the graph from a `.gguf` file through the architecture catalog.
- **llama.cpp cgraph:** converts graphs built by llama.cpp's OpenVINO backend.

Validation applies to the tested checkpoint and configuration, not to every model, modality,
context length or quantization sharing an architecture name. Acceptance limits are defined in
[testing.md](testing.md#acceptance). Projector files (`mmproj-*.gguf`) are covered in [mmproj.md](mmproj.md).

## Native GGUF path

[`arch_registry.cpp`](../src/builder/arch_registry.cpp) holds one catalog; other names are rejected at
`load()` unless an [extension](extensions.md) registers them on that frontend. `clip` selects the
[mmproj coordinator](mmproj.md).

The checkpoint column counts matching greedy choices over prefill plus twelve reference-token decode
steps for the prompt `The capital of France is`, through `GGUFMakeStateful`, `AdaptToGenAI` and CPU
with the [strict accuracy settings](quantization.md#accuracy-controls) (`OV_GGUF_Q4_K_ZP_F16=1` for Q4_K).
`—` means no recorded real-checkpoint comparison.

| Architecture | Notes | Real checkpoint | Choices |
|---|---|---|---|
| `bailingmoe2` | Sigmoid routing, biased/grouped selection, shared experts | Ling-mini-2.0 Q2_K | 12/13 |
| `deepseek2-ocr` | Language backbone: dense lead layers, shared/routed experts | DeepSeek-OCR-2 Q4_K_M | 13/13 |
| `ernie4_5-moe` | Interleaved MoE, biased selection, normalized weights, shared experts | ERNIE-4.5-21B-A3B-PT Q4_K_M | 12/13 |
| `exaone-moe` | Dense lead, shared/routed experts, RoPE on SWA layers only | K-EXAONE-236B-A23B Q2_K | 11/13 ¹ |
| `exaone4` | Post-norm-only attention and FFN | EXAONE-4.0-1.2B Q4_K_M | 13/13 |
| `gemma` | GeGLU, multi-query attention | gemma-2b Q4_K_M | 12/13 |
| `gemma2` | Post-norms, SWA, attention and final soft-caps | gemma-2-2b-it Q4_K_M | 13/13 |
| `gemma3` | Post-norms, final soft-cap, separate global/local RoPE scaling | — | |
| `gemma4` | SWA, per-layer embeddings, shared KV, dense plus routed experts | — | |
| `glm4moe` | Biased sigmoid selection, shared experts, optional QK-norm | GLM-4.5-Air REAP50 Q2_K | 12/13 |
| `gpt-oss` | MoE, attention sinks, SWA, OAI gated activation; **no numerical fixture** | — | |
| `hunyuan-dense` | Learned QK-norm after RoPE | Hunyuan-0.5B-Instruct Q8_0 | 13/13 |
| `hunyuan-moe` | QK-norm after RoPE, routed and shared experts | Hunyuan-A13B-Instruct Q2_K | 13/13 |
| `jais2` | Biased LayerNorm, ungated ReLU² FFN | Jais-2-8B-Chat Q4_K_M | 13/13 ² |
| `llama` | llama-2 / llama-3, YaRN | Devstral Small 2507 24B Q4_K_M | 13/13 |
| `llama-embed` | Per-token embeddings; mean/first/last pooling; causal or bidirectional | Llama-Nemotron-Embed-1B-v2 Q4_K_M | NMSE 3e-6 ² |
| `maincoder` | NORMAL RoPE, QK-norm after RoPE | Maincoder-1B Q4_K_M | 13/13 |
| `mamba2` | Mamba 2 mixer, tied or separate output; one-sequence greedy decoding | Mamba2-2.7B Q8_0 | 12/13 ³ |
| `mellum` | MoE with normalized expert weights | Mellum2-12B-A2.5B-Instruct Q4_K_M | 12/13 |
| `minicpm` | Embedding/residual scales, inverse logit scale | — | |
| `minimax-m2` | Full-width QK-norm, partial RoPE, normalized experts | MiniMax-M2.1-REAP-50 Q2_K | 12/13 ⁴ |
| `mistral3` | Dense, NORMAL RoPE, attention temperature | Devstral Small 2 24B; Ministral-3-3B-Instruct-2512 Q4_K_M | 13/13; 13/13 |
| `muse-glimmer` | RoPE on SWA layers, attention output gate, pre/post norms | Muse-Glimmer-30B Q4_0 | 13/13 |
| `nemotron_h` | Hybrid Mamba 2 / attention / ReLU² FFN; one-sequence greedy decoding | Nemotron-H-8B-Reasoning-128K Q4_K_M | 12/13 |
| `olmoe` | Full-width QK-norm, MoE | — | |
| `phi3` | Fused QKV | — | |
| `plamo3` | Fused QKV and gate/up, bare post-norms, SWA | plamo-3-nict-2b-base Q4_K_M | 12/13 |
| `qwen2` | qwen2 / qwen2.5 | — | |
| `qwen3` | QK-norm before RoPE; also Bonsai-8B (Q1_0) | — | |
| `qwen35` | Hybrid GatedDeltaNet + attention, interleaved M-RoPE; batching and beam search with SDPA and PA; also Bonsai-27B (Q1_0), Ternary-Bonsai-27B (Q2_0 g64) | — | |
| `qwen35moe` | Hybrid GatedDeltaNet, separate or fused routed experts, shared expert | Qwen3.6-35B-A3B Q4_K_M / Q4_0 | 13/13 / 12/13 |
| `qwen3moe` | NEOX RoPE, per-head QK-norm, normalized experts | Qwen3-0.9B-A0.6B Q4_K_M | 13/13 |
| `smollm3` | NORMAL RoPE, skipped on every fourth layer | SmolLM3-3B Q4_K_M | 12/13 |

1. Below the acceptance threshold (same first prediction, differences at steps 4–5). Its F32
   fixtures, including NextN, pass. A staged CPU runner that compiles one layer at a time (u2 expert
   weights expand) produced coherent chat answers; full-model compilation is not validated.
2. Against llama.cpp F32 arithmetic on the represented weights. Against its quantized CPU kernels,
   Jais gives 11/13 with the same first prediction; this remains a validation limitation.
3. llama.cpp's Q8 path quantizes activations. Against F32 arithmetic on the same weights, Mamba2-130M
   and 2.7B give 13/13; with OpenVINO dynamic quantization group size 32, both give 13/13 against Q8 CPU.
4. Peaked at about 222 GiB RAM plus swap because u2 experts expand at CPU compilation; see
   [memory limits](quantization.md#memory-and-packaging-limits).

### Numerical regression coverage

[`GGUFArchitectureAccuracy`](../tests/test_arch_accuracy.cpp) runs 40 small nonzero F32 decoder
fixtures (every decoder above except `gpt-oss`, plus Devstral YaRN/temperature, Gemma4 MQA/MoE/PLE,
Qwen3.5 mixed and fused variants) and `GGUFEmbeddingAccuracy` five embedding fixtures. Each compares
complete logits with llama.cpp CPU through prefill, decode and a two-token cache append. Fixtures
cover sliding-window boundaries, SmolLM3's NoPE layer, ERNIE without `expert_shared_count` and
Bailing grouped routing; see the [fixture README](../tests/test_data/arch_accuracy/README.md).
`GGUFMultimodalBackboneAdaptation` checks padded batches, beam reordering and PagedAttention for
`qwen35`, `qwen35moe` and `gemma4`. Zero-weight fingerprints add structural coverage.

### Runtime limitations

- **Recurrent states:** Gated-DeltaNet states support batching and beams through `AdaptToGenAI`;
  Mamba 2 and `nemotron_h` support one sequence with greedy decoding and resettable caches, without
  GenAI paged integration; beam search and prefix-cache reuse are untested. Mamba 1 and
  `nemotron_h_moe` are not supported. See [runtime.md](runtime.md#stateful-and-genai-conversion).
- **Multimodal:** a tested language backbone does not cover its vision or audio encoders,
  preprocessing or application pipeline.
- **Quantization and memory:** see [quantization.md](quantization.md).

## llama.cpp cgraph path

Manually validated with the backend's stateful execution:

| Architecture | Tested model | Notes |
|---|---|---|
| `llama` | TinyLlama-1.1B-Chat v1.0 Q4_K_M | Dense, GQA |
| `qwen2` | Qwen2.5-0.5B-Instruct Q8_0 | Dense |
| `qwen3` | Qwen3-0.6B Q8_0 | QK-norm |
| `qwen3moe` | Qwen3-0.9B-A0.6B Q4_K_M, Qwen3-4B Q4_K_M | `mul_mat_id` |
| `olmoe` | OLMoE-1B-7B-0924-Instruct Q4_0 | MoE |
| `gemma3` | gemma-3 (checkpoint not recorded) | Mixed SWA/global RoPE |
| `gemma4` | gemma-4-E4B-it Q4_K_M | Per-op RoPE, F16 KV cache |
| `qwen35` | Qwen3.5-4B Q4_K_M | GatedDeltaNet, partial-rotary IMROPE, interleaved Q/gate; F16 KV cache |

These runs record no llama.cpp revision. [llama.cpp compatibility CI](../../../../.github/workflows/job_gguf_llamacpp_validation.yml)
pins the backend revision and runs its operator tests plus state scenarios on generated
`llama`, `qwen2` and `qwen3` models only. A run passes when the output is coherent and consistent
with the ggml CPU backend:

```sh
GGML_OPENVINO_DEVICE=CPU GGML_OPENVINO_STATEFUL_EXECUTION=1 \
  llama-completion -m <model>.gguf -p "The capital of France is" -n 12 -no-cnv --no-warmup
```

Checkpoints used `Q2_K`, `Q4_0`, `Q4_K_M`, `Q6_K` and `Q8_0`; other formats are covered by unit tests only.
