# Devstral behavior implemented in an external builder

This example implements the Devstral **text graph** in a loadable `ModelBuilder`, using only
installed GGUF SDK headers. It reads the original GGUF metadata and builds YaRN and
position-dependent attention scaling itself. It does not call `make_decoder_architecture`,
`configure_decoder`, `decoder_attention` or `decoder_ffn`, and does not depend on the shared
`DecoderConfig` resolver's Devstral fixes. No new operation converter is required.

The current frontend already supports these behaviors natively. This plugin demonstrates how
to implement them externally when the shared decoder does not, rather than adding names to
the supported-model list. It requires the generic-node SDK in this branch; it is not binary
compatible with older SDKs.

## What the extension owns

[`devstral.cpp`](devstral.cpp) contains the graph and its `ArchitectureDefinition` factory:

- Read dimensions, RoPE parameters and attention temperature directly from `GgufMetadata`.
- Configure YaRN correction and magnitude through `RopeConfig` and `configure_rope`.
- Emit Q/K/V projections, RoPE and the query multiplier
  `1 + temperature * log(1 + floor(position / original_context))`.
- Describe KV updates with `GGML_OP_SET_ROWS` and attention with `GGML_OP_FLASH_ATTN_EXT`.
- Assemble residuals, normalization, the dense SwiGLU FFN and the output projection.

Weight loading, generic converters, OpenVINO shape/type inference, normalization helpers and
input helpers are reused. The four-operation SwiGLU expression is local so the example does
not need shared decoder configuration at all. This costs a complete small decoder recipe,
not just the registration wrapper. There are no builder-side shape formulas.

[`extension.cpp`](extension.cpp) wraps the same definition for external loading:

```cpp
std::make_shared<ArchitectureExtension>(
    example::devstral_decoder("mistral3"), RegistrationMode::Replace)
```

## Build and load

Build against an installation of this branch:

```sh
cmake -S src/frontends/gguf/examples/devstral_extension -B /tmp/devstral-extension \
    -DOpenVINO_DIR=/path/to/openvino/runtime/cmake
cmake --build /tmp/devstral-extension
```

Load on the explicitly selected frontend:

```cpp
ov::frontend::FrontEndManager manager;
auto frontend = manager.load_by_framework("gguf");
frontend->add_extension("/tmp/devstral-extension/libgguf_devstral_extension.so");
auto model = frontend->convert(frontend->load("Devstral-Small-2-Q4_K_M.gguf"));
```

Devstral Small 2505/2507 uses `llama`; Devstral Small 2 and Devstral 2 use `mistral3`. The library explicitly
replaces both existing family handlers in this frontend instance. **Use a dedicated frontend
instance for the intended Devstral models**: this is a dense, full-attention text example, not a
replacement supporting every Llama/Mistral variant. Vision, MoE, sliding-window variants and
additional family-specific behavior are outside its scope.

For cached GenAI decoding, the consumer selects `GGUFMakeStateful` followed by `AdaptToGenAI`.
The latter must tolerate a pruned, unused `token_len_per_seq` input; this branch includes that
consumer-adapter correction and its regression test. GenAI must forward the library to the
GGUF frontend it creates. Registering on an unrelated `Core` or frontend does not forward it.

## Validation and built-in integration

The three existing Devstral F32 fixtures run through the loaded library as well as the native
builder. They compare complete logits to independent llama.cpp CPU references, including
prefill, cached decode, nondefault YaRN correction and temperature boundaries. See
[supported models](../../docs/supported_models.md#devstral-text-models) for checkpoint scope and
[reference generation](../../tests/test_data/arch_accuracy/README.md) to reproduce the fixtures.
Run both native and external cases with
`--gtest_filter='*GGUFArchitectureAccuracy*devstral*'`.

To include this implementation in the frontend, move `devstral.cpp` and its header into the
builder sources and register the same `devstral_decoder` definition. Replace the intended
handler or partition matching predicates; do not add a second handler claiming the same files.
Keep the builder unchanged and omit the library entry point. For the current built-in catalog,
the smaller shared fixes already provide the behavior, so duplicating this example in-tree is
unnecessary.
