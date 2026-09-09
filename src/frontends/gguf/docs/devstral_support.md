# Devstral support and integration effort

Devstral reuses existing GGUF families; no new architecture alias is required. The native
frontend implements its behavior in shared decoder blocks. The external example independently
implements the text graph in a `ModelBuilder`, demonstrating a route without those shared fixes.
This work covers text decoding, not Small 2's vision encoder/projector or agent/tool orchestration.

## Model coverage

| Model | GGUF family | Evidence in this branch |
|---|---|---|
| Devstral Small 2505 | `llama` | Same decoder configuration as 2507; checkpoint not independently tested |
| Devstral Small 2507, 24B | `llama` | Small numerical fixture and real Q4_K_M checkpoint, native and external routes |
| Devstral Small 2, 24B | `mistral3` | Small numerical fixture and real Q4_K_M text checkpoint, native and external routes |
| Devstral 2, 123B | `mistral3` | Published configuration and small numerical fixture; real 123B checkpoint not tested |

Published configurations: [Small 2505](https://huggingface.co/mistralai/Devstral-Small-2505/blob/main/config.json),
[Small 2507](https://huggingface.co/mistralai/Devstral-Small-2507/blob/main/config.json),
[Small 2](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512/blob/main/config.json),
[Devstral 2](https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512/blob/main/config.json).
The GGUF family mapping follows llama.cpp's Mistral and Mistral3 converters.

## What changed

The first Small 2 numerical reference failed on the previous builder. Four existing
production files needed changes:

- `quant/gguf.cpp`: read YaRN correction parameters, its log multiplier and attention temperature.
- `builder/decoder_config.{hpp,cpp}`: resolve the RoPE magnitude and temperature configuration.
- `builder/blocks/attention.cpp`: apply the position-dependent query scale, preserving dynamic token shapes.

The query scale follows llama.cpp: `1 + temperature * log(1 + floor(position / original_context))`.
Existing copy/conversion, scale, log, reshape and multiply translators implement it. YaRN's
log multiplier adjusts the magnitude factor already computed by the RoPE translator.
These changes also benefit other models using the same metadata. No new operator translator,
builder class or architecture registration was added.

## External implementation without shared-builder changes

The [external example](../examples/devstral_extension) now contains a standalone
[`DevstralBuilder`](../examples/devstral_extension/devstral.cpp). It replaces the earlier
registration-only demonstration, which simply called `make_decoder_architecture` and therefore
depended on the shared fixes above.

The custom builder reads raw metadata, supplies its own `RopeConfig`, inserts the query-scale
operations and assembles the attention and dense decoder graph. It reuses the generic SDK,
weight loading, normalization and existing operation converters. It does not use `DecoderConfig`,
`configure_decoder`, `decoder_attention` or `decoder_ffn`; even the short SwiGLU expression is
local to avoid a dependency on shared decoder configuration. No shared model builder or operation
converter was changed to implement this example.

The library entry point replaces the `llama` and `mistral3` handlers in its own frontend instance.
It is intentionally scoped to the Devstral dense text topology, not every model using those
family names. The current native frontend runs the supported Devstral text checkpoints without
this plugin. The extension demonstrates independent behavior, not additional architecture count.

The work also exposed a consumer-adapter issue: generic graphs can prune an unused
`token_len_per_seq` input, but `AdaptToGenAI` previously required it and silently skipped adaptation.
That pass now accepts its absence. This correction is separate from the model builder; the
example's GenAI integration requires it. Libraries still need to be built against the SDK version
used by the application.

## Validation (2026-09-09)

The standalone example builds against installed headers from this branch. All 327 frontend
tests and the external-library test pass, including the optional-token-count adapter regression.

Three committed F32 fixtures cover the Devstral generations. Each runs through the native
frontend and the loaded library: six checks, all comparing complete logits to real llama.cpp
CPU with normalized MSE below `1e-5`. The cases cover unequal embedding/query widths,
non-default YaRN parameters, prefill and two cache appends. Small 2 uses a reduced original
context of two tokens so the test crosses the temperature boundaries at positions 2 and 4.
A configuration test also checks metadata propagation and rejects a zero original context
when temperature scaling is enabled.

Two real 24B Q4_K_M checkpoints were checked with the prompt `The capital of France is`.
The independent external builder was rerun on September 9; native results are from September 7:

| Checkpoint | Native matching choices | External matching choices | Maximum logit NMSE |
|---|---|---|---|
| Devstral Small 2507 | 13/13 | 13/13 | 0.007764 |
| Devstral Small 2 | 13/13 | 13/13 | 0.008226 |

These are prefill plus twelve decode steps on identical reference-fed histories. Settings:
CPU F32 inference, F16 KV cache, activation quantization disabled and
`OV_GGUF_Q4_K_ZP_F16=1`. The reference is llama.cpp CPU at
`476c01efe88aad7880a8132d5d3a415f2ca75139`. Quantized checkpoints use the documented
first-choice/90%-agreement criterion, not the strict F32-fixture error limit.

Checkpoint repositories and pinned revisions:

- [Small 2507 Q4_K_M](https://huggingface.co/bartowski/mistralai_Devstral-Small-2507-GGUF/tree/3aaee38f9345e582cca40a0d2a320dc23e21dc32)
- [Small 2 Q4_K_M](https://huggingface.co/bartowski/mistralai_Devstral-Small-2-24B-Instruct-2512-GGUF/tree/027695770ae1de77c2f6fb19f8e1ba9d65fcd15d)

Both downloaded files were checked against their repository SHA256 values. The original combined
real-checkpoint run peaked at approximately 39.3 GiB RSS on the local CPU host. Nightly
model-hub entries cover conversion and finite-output smoke checks for both checkpoints;
the committed numerical fixtures run offline in the regular frontend suite.

Reproduce with the [architecture oracle instructions](../tests/test_data/arch_accuracy/README.md),
using data stems `devstral-small` and `devstral-small2`. Filter with
`--gtest_filter='*GGUFArchitectureAccuracy*devstral_small*'` to run native and external checks.

This establishes bounded CPU text-decoder agreement, not coding benchmark quality, full
128K/384K/256K context capacity, vision support, or coverage of every device and quantization.
The architecture catalog remains at 23 verified names and seven experimental names because
Devstral uses families that were already registered.
