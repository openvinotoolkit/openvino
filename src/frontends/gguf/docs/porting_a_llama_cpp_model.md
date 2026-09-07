# Architecture definitions: external plugins and built-in support

An architecture is described by an `ArchitectureDefinition`. The native catalog and runtime
extensions consume the **same definition**, invoke the same factory, and use the same conversion
and normalization pipeline. `ArchitectureExtension` is only the runtime registration adapter.

The SDK is a developer API. Build an extension against the OpenVINO release it will run with;
compatibility with future SDK revisions is not promised. Loading an extension does not require
rebuilding that OpenVINO release.

## Choose the smallest implementation

| Requirement | Implementation |
|---|---|
| Existing decoder topology, new architecture name | `make_decoder_architecture(name, rope)` |
| Decoder needs different architecture facts | Supply a callback returning `DecoderOptions` |
| Custom layer order using existing decoder sublayers | `ModelBuilder` using `configure_decoder`, `decoder_attention`, `decoder_ffn` |
| Different model family | `ModelBuilder` using generic graph operations and its own metadata |

Prefer structural detection to overriding an option. Tensor presence, metadata, and the native
configuration resolver already handle the supported decoder features. A custom topology owns its
embedding, normalization, residual, and output ordering; the shared sublayer methods do not assemble
those automatically. `decoder_layer_parameters(layer)` supplies resolved head counts, attention
scale and RoPE parameters when implementing a custom attention fragment.

## Define a decoder

```cpp
#include "openvino/frontend/gguf/extension/architecture.hpp"

ov::frontend::gguf::ArchitectureDefinition my_architecture() {
    using namespace ov::frontend::gguf;
    return make_decoder_architecture(
        "my-arch", RopeMode::Neox,
        [](const GgufMetadata& metadata) {
            DecoderOptions options;
            options.geglu = true;
            return options;
        });
}
```

Omit the callback when no overrides are required. `RopeMode` explicitly distinguishes consecutive
pairs (`Normal`), rotate-halves (`Neox`), and interleaved multimodal RoPE (`Interleaved`). Verify the
mode against the architecture's reference implementation.

`DecoderOptions` contains supported architectural overrides, not mutable dimensions or execution
plans. The callback runs before configuration resolution. The native resolver validates the options
and derives the SWA RoPE configuration and KV plan afterward. There is no second SDK hyperparameter
reader: both built-in and custom decoder topologies use `decoder_config_from_meta` and
`DecoderConfig`, including their defaults and RoPE scaling rules. The resolved configuration stays
internal to the frontend.

## Define a custom builder

Implement `ModelBuilder::build()` with a `GgufGraphContext`. The `BuildContext` is a borrowed view of
metadata and weights, valid for the synchronous factory/build call. Do not retain it afterward.

Read scalar metadata with `get_int`, `get_float`, `get_bool`, or `get_str`; they return
`std::optional`, so defaults use standard `value_or`, e.g.
`metadata.get_int("my-arch.block_count").value_or(1)`. Array getters return typed vectors.
Numeric conversion reuses OpenVINO's tensor utilities rather than a separate SDK type dispatcher.

The [projector example](../examples/architecture_extension/projector.cpp) is a complete small
non-decoder family. It reads a weight, accepts a variable number of input embeddings and projects
them into another space. Its architecture definition is separate from the plugin entry point.

For custom decoder layer order, initialize the shared decoder blocks once:

```cpp
auto dimensions = graph.configure_decoder(RopeMode::Neox, options);
graph.build_inp_pos();
graph.build_attn_inp_kv();
// Inside the custom layer loop, after constructing the appropriate normalization:
auto attention = graph.decoder_attention(layer, normalized_input);
auto feed_forward = graph.decoder_ffn(layer, normalized_ffn_input);
```

These methods call the existing `blocks::attention` / `blocks::gated_delta_net` and dense/GeGLU/MoE
FFN implementations. Configuration chooses the applicable sublayer. There is no parallel SDK
implementation of those blocks. Decoder dimensions are returned as a value snapshot; modifying it
does not mutate the resolved model. See the custom decoder in
[`test_architecture_extension.cpp`](../tests/test_architecture_extension.cpp).

## Tensor operations and shapes

The SDK uses GGML operand order, e.g. `mul_mat(weight, activation)`. Value shapes use OpenVINO order;
`value.ne(i)` accesses the reversed, GGML dimension order. Weight values preserve all logical
axes, including vectors and expert dimensions. Looking up a missing optional weight returns an
empty value; `tensors.require(name)` reports missing mandatory weights.

`reshape(value, {width, heads, -1})` takes a target in GGML dimension order. It is an explicit
reshape with at most one inferred dimension. It does not infer an attention operation from the
number of requested dimensions. Use `split_heads(value, heads, width)` and `merge_heads(value)`
for decoder attention layouts that must preserve the leading batch and dynamic token axis.

Use `-1` for a variable extent. SDK values retain dynamic dimensions; representative static
metadata is internal bookkeeping for existing GGML translators, not a token-count API.
`permute` uses OpenVINO axis numbering and requires each of the four axes exactly once.

The SDK is not a drop-in implementation of llama.cpp's `llm_graph_context`. When porting a model,
map its sublayers onto existing blocks first, and use generic operations for new structure. The
removed `GgufHparams`, `LayerTensors`, and `build_lora_mm` facades should not be recreated in each
architecture. Tensor names can be kept in a model-specific helper when that improves readability.

`raw_op` supports operations outside the convenience vocabulary. It describes an existing GGML
translator's shape, case, and attribute contract; a new operation can be supplied with a
`ConversionExtension`. It does not make every ggml memory-view or stride operation interchangeable
with an OpenVINO tensor operation.

## Declare model contracts

Graph nodes alone do not describe how consumers should maintain state or form masks:

- `configure_decoder` records the resolved RoPE and sliding-window metadata automatically.
- `rope_ext` records per-operation RoPE use and the multimodal position contract.
- `set_sliding_window(tokens)` records the window for a custom attention implementation.
- `add_recurrent_state(input, update)` declares an overwritten state and marks the update as an
  output. `GGUFMakeStateful` consumes this relationship. Its existing batch/beam restrictions still
  apply.
- `set_output(value)` declares other outputs. `finish()` validates references and outputs and seals
  the context against further emission.

These declarations are passed through the same `GgufGraph` and `TranslateSession` pipeline as the
built-in decoder. A consumer selects `GGUFMakeStateful` and `AdaptToGenAI` through transformation
extensions; an architecture definition does not select a device or force stateful execution.

## Selection and replacement

A definition has both a unique handler **id** and the file's **architecture**. Matching requires
`general.architecture` to agree, followed by the optional metadata predicate. For example, two
handlers may share `architecture = "clip"` but use ids `"clip.vision"` and `"clip.audio"`, with
disjoint modality predicates. They coexist without overwriting each other.

Two matching handlers are an error. Duplicate ids are also an error. To intentionally replace an
existing handler, including a built-in, register with `RegistrationMode::Replace`:

```cpp
frontend.add_extension(std::make_shared<ArchitectureExtension>(
    my_architecture(), RegistrationMode::Replace));
```

Replacement requires the id to already exist. Registries belong to individual frontends, so a
replacement never changes another frontend's catalog.

## Build and load an external plugin

The plugin entry point only wraps the definition:

```cpp
OPENVINO_CREATE_EXTENSIONS(std::vector<ov::Extension::Ptr>{
    std::make_shared<ov::frontend::gguf::ArchitectureExtension>(my_architecture())});
```

Link against `openvino::frontend::gguf`. The
[standalone CMake example](../examples/architecture_extension/CMakeLists.txt) builds against the
installed SDK:

```sh
cmake -S src/frontends/gguf/examples/architecture_extension -B /tmp/gguf-extension \
    -DOpenVINO_DIR=/path/to/openvino/runtime/cmake
cmake --build /tmp/gguf-extension
```

Load by framework name: GGUF is currently hidden from automatic frontend selection.

```cpp
ov::frontend::FrontEndManager manager;
auto frontend = manager.load_by_framework("gguf");
frontend->add_extension("/path/to/libgguf_projector_extension.so");
auto model = frontend->convert(frontend->load("model.gguf"));
```

A consumer integrating through `Core` must arrange for its extensions to reach the explicitly
selected GGUF frontend. Registering on an unrelated frontend instance does not forward them.

## Promote the same implementation into OpenVINO

1. Move the architecture source/header (for example, `projector.cpp` / `projector.hpp`) into
   `src/builder/arch/`. Keep the definition factory and `ModelBuilder` unchanged. The frontend
   source collection includes files under this directory.
2. Include its header in `src/builder/arch_registry.cpp` and add
   `definitions.push_back(my_architecture());` to `builtin_architectures()`.
3. Add the source to the explicit frontend-source list in `tests/CMakeLists.txt`, and retain its
   conversion and accuracy tests. The plugin-only `OPENVINO_CREATE_EXTENSIONS` source is not needed.
4. Update supported-model documentation and declare `Maturity::Verified` only after real-model
   accuracy validation. Synthetic conversion tests do not establish model support.

For a plain decoder, add a row to the decoder catalog with name, RoPE mode, and maturity. For a
decoder with options, add the unchanged definition factory instead. Do not also register a plain
decoder row under the same id.

No new family-dispatch branch, frontend registration method, or alternate graph implementation is
needed. The tests compile the projector source both as a library and into a catalog test to exercise
this migration path.

## Validate

- Compile an external plugin using installed headers and load the actual library.
- Compare numerical results against a reference, including multiple token lengths and subsequent
  decoding with previously generated state. Check SWA beyond its window and non-default RoPE scaling.
- Run `ov_gguf_frontend_tests` architecture fixtures to check built-in graph fingerprints.
- For a real architecture, also compare generation to llama.cpp on the same checkpoint before
  marking it verified.
