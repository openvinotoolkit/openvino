# Architecture definitions: external plugins and built-in support

An architecture is described by an `ArchitectureDefinition`. The native catalog and runtime
extensions consume the **same definition**, invoke the same factory, and use the same conversion
and normalization pipeline. `ArchitectureExtension` is only the runtime registration adapter.

The SDK is a developer API. Build an extension against the OpenVINO release it will run with;
compatibility with future SDK revisions is not promised. Loading an extension does not require
rebuilding that OpenVINO release.

## Choose the smallest implementation

The [Devstral extension](../examples/devstral_extension) implements YaRN and position-dependent
attention scaling directly from metadata, without using the shared decoder resolver or blocks.

| Requirement | Implementation |
|---|---|
| Existing decoder topology, new architecture name | `make_decoder_architecture(name, rope)` |
| Decoder needs different architecture facts | Supply a callback returning `DecoderOptions` |
| Custom layer order using existing decoder sublayers | `ModelBuilder` using `configure_decoder`, `decoder_attention`, `decoder_ffn` |
| Different model family | `ModelBuilder` using generic graph operations and its own metadata; follow the [full porting walkthrough](#port-a-fully-new-architecture-from-llamacpp) |

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
internal to the frontend. Options also cover QK-norm placement (`qk_norm_after_rope`),
post-norm-only blocks (`post_norm_only`), selected expert normalization
(`normalize_expert_weights`), and periodic NoPE layers (`rope_skip_period`, zero to disable).

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

## Port a fully new architecture from llama.cpp

Use this route when the model's computation cannot be expressed by the shared decoder blocks:
for example, a new encoder, a different recurrent network, or an encoder-decoder family. Implement
its whole graph in a `ModelBuilder`. The native frontend reads the GGUF file; your builder describes
the computation using `node` and any applicable shared blocks. llama.cpp supplies the reference
implementation and numerical oracle, but is not a dependency of the resulting extension.

### 1. Trace the reference from the file format to the outputs

Start with a llama.cpp revision that loads the target checkpoint correctly. Record its commit,
checkpoint, quantization, input preprocessing and runtime options so comparisons are reproducible.
The source locations below are relative to that llama.cpp checkout; older revisions may combine
these implementations in larger files.

| Source to inspect | What to extract |
|---|---|
| `src/models/<architecture>.cpp`: `load_arch_hparams` | Required metadata, defaults, architectural variants and constraints |
| The same file: `load_arch_tensors` | Tensor names and dimensions; required/optional weights; aliases and tied weights |
| The same file: `build_arch_graph` and its graph constructor | Inputs, layer order, branches, residuals, final projection and output selection |
| `src/llama-model.cpp`, `src/llama-arch.cpp` and `src/llama-hparams.h` | Common metadata loading, expansion of `LLM_KV_*` / `LLM_TENSOR_*` names and derived dimensions |
| `src/llama-graph.cpp` and the relevant memory implementation | Bodies of called helpers, attention masks, cache reads/writes and recurrent updates |
| `tools/mtmd/models/` and their shared helpers | Vision/audio graph construction, patch processing, pooling and projection, when outside the language model |
| GGUF conversion code and `gguf-py/gguf/` | Actual exported names, tensor packing, transpositions and weight transformations |

For example, in a checkout with separate model files:

```sh
rg -n 'load_arch_hparams|load_arch_tensors|build_arch_graph|::graph' src/models/<architecture>.cpp
rg -n 'build_lora_mm|build_norm|build_ffn|build_attn' src/llama-graph.cpp
rg -n 'LLM_KV_<KEY>|LLM_TENSOR_<TENSOR>' src/llama-arch.cpp
```

Follow every helper used by the architecture. `build_lora_mm`, for example, can include an
additional weight scale and active LoRA updates after the base matrix multiplication. Replacing
it with `MUL_MAT` reproduces the base path only when those additions are absent. Likewise,
`build_norm` selects RMS, layer or group normalization and may apply both a weight and a bias.
A helper's name alone does not establish its semantics.

Write down the complete input-to-output sequence before coding. Include operations outside the
layer loop: embedding scales, learned positions, final normalization, pooling, output-row selection
and logit transforms. A GGUF model name or matching tensor names do not imply the same computation.
For multimodal models, identify which file holds each graph and which preprocessing steps happen
on the host; supporting a projection graph does not automatically support the whole pipeline.

### 2. Define the file and model interface

Create a small contract alongside the port:

- The literal `general.architecture`, plus metadata predicates needed to distinguish variants.
- Required metadata and its types, source-defined defaults, and per-layer arrays or schedules.
- Required tensors, optional tensors and their fallback behavior, with **logical** GGML dimensions.
- Each model input's name, element type, axis meanings and variable dimensions.
- Outputs and how the consumer interprets them: logits, embeddings, encoded features or state.
- Cache/state inputs, their updates, position indices and masks, if applicable.

Read a new family's own keys with `graph.metadata()`. Reject missing mandatory fields rather than
silently inventing dimensions; use `value_or` only for defaults present in the reference. Derive
weight-dependent dimensions from `GgufValue::ne` or its inferred shape, not from packed byte sizes.
`tensors.require(name)` loads a mandatory weight, `tensors(name)` returns an empty value when an
optional weight is absent, and `tensors.has(name)` checks presence without emitting a node.
Implement tied-weight fallbacks explicitly where the reference does so.

The SDK accepts named model inputs with a known rank of at most four; dimensions may be dynamic.
Use the layouts expected by the selected converters. For example, GGML `[width, items, 1, 1]`
corresponds to OpenVINO `[1, 1, items, width]`. This is not an implicit batch-capability guarantee:
the model's operations, state and consumer must all support the batch layout you expose.

A new encoder need not call `configure_decoder`, create token/position inputs, or emit logits.
An encoder-decoder port must describe both its encoding and decoding computations and their
connection, including cross-attention and state where needed; registration does not generate them.

### 3. Translate graph construction, preserving operation semantics

Port one reference fragment at a time, retaining its operand order and branches. Use shared
attention/FFN blocks only where their semantics match. Otherwise express the fragment with generic
nodes. Local functions can name repeated architectural fragments without adding methods to the SDK.

| llama.cpp construction | Builder equivalent or porting action |
|---|---|
| `create_tensor(tn(...), ...)` / a layer weight pointer | Look up the actual GGUF name through `GgufTensors`; parsing and dequantization are already handled |
| `ggml_new_tensor_*` for runtime data | `add_input(name, type, shape)`; supply values at inference time |
| `ggml_add`, `ggml_sub`, `ggml_mul` | `node("GGML_OP_ADD" / "GGML_OP_SUB" / "GGML_OP_MUL", {a, b})` |
| `ggml_mul_mat(ctx, weight, x)` | `node("GGML_OP_MUL_MAT", {weight, x})` |
| `ggml_silu(ctx, x)` | `node("GGML_UNARY_OP_SILU", {x})` |
| `ggml_rms_norm(ctx, x, eps)` | `node("GGML_OP_RMS_NORM", {x}, 0, {{"eps", eps}})`; learned scaling remains a separate multiply |
| `ggml_reshape_*` | `GGML_OP_RESHAPE`, case 6, with an OpenVINO-order `reshape_target`; use `special_zero` when copying an input dimension |
| `ggml_permute` | `GGML_OP_PERMUTE`, case 1, with an OpenVINO-order `perm`; translate the axis mapping |
| `ggml_view_*` | Describe the logical slice/layout with the converter's attributes; do not reproduce a ggml pointer or allocator |
| `ggml_top_k(ctx, scores, k)` | `GGML_OP_TOP_K` with explicit `int64_t` attribute `k`; its converter selects I32 indices |
| `ggml_cpy`, `ggml_set*` used for state | Preserve the updated tensor and its consumer/state relationship, not an in-place memory side effect |
| `cb(...)`, graph expansion and backend scheduling | Reference diagnostics and execution infrastructure; no corresponding model computation to emit |

Inspect [`src/op_table.cpp`](../src/op_table.cpp) and the selected converter in
[`src/op/`](../src/op) for the accepted attributes and `op_case`. The case is a semantic variant,
not a llama.cpp operation enum or an arbitrary tag. The generic builder forwards it without
interpreting the operation. See [Tensor operations and shapes](#tensor-operations-and-shapes).

Pay particular attention to layouts:

- `ggml_permute` specifies the destination of each source axis, while OpenVINO's `perm` lists
  source axes in output order. With four GGML axes `axes[i]`, the corresponding mapping is
  `perm[3 - axes[i]] = 3 - i`. Derive it from axis meanings instead of copying four integers.
- A ggml view's byte offset and strides describe a logical selection. For a simple contiguous
  subrange, VIEW case 3 accepts `view_slice = {ov_axis, start, length}` and optional
  `view_reshape`. More complex strided views need an appropriate converter or an explicit
  composition of layout operations; a slice alone is not a general replacement.
- OpenVINO has no ggml contiguity requirement. CONT case 1 is a passthrough; a combined
  `ggml_cont_2d/3d/4d` call still has a reshape to preserve. Do not remove real transposes.
- Preserve dynamic sequence/item dimensions. A successful one-token graph does not validate the
  layout: incorrect broadcasting can appear only with several tokens, heads or experts.

### 4. Implement a whole-model builder

The example below is a complete builder for an **illustrative** `example-encoder` file contract,
not a claim of support for a named llama.cpp model. It shows the mechanics for a family with its
own metadata and inputs, without using the decoder configuration or attention blocks.

Assume the reference reads `example-encoder.block_count`, `example-encoder.embedding_length`
and RMS epsilon (default `1e-5`), consumes feature vectors, and applies this per-layer computation:

```cpp
// Reference computation after resolving the layer's weights (optional biases omitted here).
auto norm = ggml_rms_norm(ctx0, cur, eps);
norm = ggml_mul(ctx0, norm, layer.norm);
auto gate = ggml_silu(ctx0, ggml_mul_mat(ctx0, layer.gate, norm));
auto up = ggml_mul_mat(ctx0, layer.up, norm);
auto down = ggml_mul_mat(ctx0, layer.down, ggml_mul(ctx0, gate, up));
cur = ggml_add(ctx0, cur, down);
```

Its GGUF tensors are `blk.N.norm.weight`, `blk.N.gate.weight`, `blk.N.up.weight`,
`blk.N.down.weight`, optional projection `.bias` tensors, and `output_norm.weight`.
Gate/up project the embedding width into the hidden width; down projects back. The output is the
full feature sequence after final RMS normalization. The port, saved as `new_family.cpp`, is:

```cpp
#include "openvino/core/except.hpp"
#include "openvino/frontend/gguf/builder/graph_context.hpp"
#include "openvino/frontend/gguf/extension/architecture.hpp"

namespace example {
using namespace ov::frontend::gguf;

class FeatureEncoder : public ModelBuilder {
public:
    explicit FeatureEncoder(const BuildContext& context) : m_context(context) {}

    std::shared_ptr<GgufGraph> build() override {
        GgufGraphContext graph(m_context);
        const auto& metadata = graph.metadata();
        const auto count = metadata.get_int("example-encoder.block_count");
        const auto width = metadata.get_int("example-encoder.embedding_length");
        OPENVINO_ASSERT(count && *count > 0, "Missing or invalid block_count");
        OPENVINO_ASSERT(width && *width > 0, "Missing or invalid embedding_length");
        const auto eps = static_cast<float>(
            metadata.get_float("example-encoder.attention.layer_norm_rms_epsilon").value_or(1e-5));
        OPENVINO_ASSERT(eps > 0, "RMS epsilon must be positive");
        auto tensors = graph.tensors();
        auto cur = graph.add_input("features", ov::element::f32, {1, 1, -1, *width});

        const auto linear = [&](const std::string& base, const GgufValue& input) {
            auto out = graph.node("GGML_OP_MUL_MAT", {tensors.require(base + ".weight"), input});
            if (auto bias = tensors(base + ".bias")) {
                out = graph.node("GGML_OP_ADD", {out, bias});
            }
            return out;
        };

        for (int64_t layer = 0; layer < *count; ++layer) {
            const auto prefix = "blk." + std::to_string(layer) + ".";
            const auto residual = cur;
            auto norm = graph.build_norm(cur, tensors.require(prefix + "norm.weight"), eps);
            auto gate = linear(prefix + "gate", norm);
            gate = graph.node("GGML_UNARY_OP_SILU", {gate});
            auto up = linear(prefix + "up", norm);
            auto hidden = graph.node("GGML_OP_MUL", {gate, up});
            auto down = linear(prefix + "down", hidden);
            cur = graph.node("GGML_OP_ADD", {residual, down});
        }

        graph.set_output(graph.build_norm(cur, tensors.require("output_norm.weight"), eps));
        return graph.finish();
    }

private:
    BuildContext m_context;
};

ArchitectureDefinition new_family_architecture() {
    return {"example.feature-encoder", "example-encoder", [](const BuildContext& context) {
                return std::make_shared<FeatureEncoder>(context);
            }};
}
}  // namespace example
```

`BuildContext` is copied here only for the synchronous factory/build lifecycle. The built graph
retains its weights, but the builder must not keep using the borrowed context after that call.
The definition defaults to experimental maturity. Its id identifies the handler; its architecture
string must match the file. Add a metadata predicate if multiple implementations share that string.

For your target, replace this example's contract and layer body with the traced reference graph,
including its embedding/patch frontend and output processing. Do not keep the example's names,
activation, normalization or residual ordering merely because the dimensions match. The existing
`VisionEncoderBuilder` in [`test_architecture_extension.cpp`](../tests/test_architecture_extension.cpp)
shows a larger custom family with Q/K/V projections, non-causal attention, an FFN and a projector.

### 5. Handle missing operations and state explicitly

Inventory the operations in the **expanded** reference helpers against the converter table before
iterating on the whole model. A new architecture built entirely from supported operations needs
no new converters. If an operation is missing, either express its exact semantics using existing
nodes or implement a converter:

- An external plugin can register an `ov::frontend::ConversionExtension` along with its
  `ArchitectureExtension`. The converter receives a frontend `NodeContext` and returns OpenVINO
  outputs. Register it on the same frontend before `convert()`; the builder calls it through `node`.
  `ConverterRegisteredAfterLoadInfersBuilderValues` in the architecture tests is a working example.
- For built-in support, follow [how_to_add_op.md](how_to_add_op.md): register the shared converter
  and add its tests and test-build source entry. Reuse an existing semantic case where applicable.
  If a legacy case requires cgraph output shapes, make its operation parameters explicit in the
  shared converter; do not add intermediate shape calculations to the builder.

This is the only operation-support layer. Do not add a builder-specific enum, switch, shape rule,
or `GgufGraphContext` method for the operation. OpenVINO shape inference happens when its converter
constructs the output nodes. GGUF weight-format support is separate: an unsupported quantization
format requires work in the weight loader, not a graph operation or architecture-name workaround.

For stateful models, first make state flow explicit. Identify the previous-state inputs, the
mathematical update and every output the consumer must retain. `add_recurrent_state` describes an
overwritten state; it is not the declaration for an append-only attention KV cache. Reuse the
shared decoder attention/cache blocks when they fit. A different cache algorithm needs its own
correct graph and consumer contract. Declare RoPE, multimodal positions and sliding windows as
explained in [Declare model contracts](#declare-model-contracts). Do not apply decoder-specific
`AdaptToGenAI` conventions to encoder outputs automatically.

### 6. Build and validate the port in increasing scope

Wrap `new_family_architecture()` in the plugin entry point and build it against installed SDK
headers using [Build and load an external plugin](#build-and-load-an-external-plugin). For this
example, replace `projector.cpp` with `new_family.cpp` in the CMake target and declare the factory
in a header included by the entry point. No llama.cpp objects or internal frontend headers are
needed. If the plugin adds converters, include them in the same extension list.

Validate more than graph construction:

1. **File/interface:** load a small nonzero fixture through the actual plugin. Check required-field
   errors, optional/tied weights, inputs and outputs, then compile the converted model on CPU.
2. **Fragments:** compare nontrivial layouts and new operations against real ggml CPU outputs.
   Use distinct dimensions and nonuniform values to expose swapped axes or misplaced scaling.
   llama.cpp's `cb` labels help locate corresponding intermediates when tracing the first mismatch.
3. **Whole graph:** compare the same preprocessed inputs through llama.cpp and the native frontend.
   For encoders compare features, pooled outputs or projections directly. For decoders compare
   complete logits on identical token histories, then prefill and several cached decode steps.
   Test multiple sequence lengths, nonzero positions, state resets and relevant window boundaries.
4. **Real checkpoint:** repeat with the actual model and quantization. Keep tokenizer, preprocessing,
   masks and precision settings aligned. Coherent text alone does not verify an encoder, pooling
   path or multimodal projector. Document any untested variants rather than marking them verified.
5. **Regression:** retain a small reproducible numerical fixture. For decoder fixtures, reuse
   [`tests/gen_arch_accuracy.py`](../tests/gen_arch_accuracy.py) and the
   [reference-generation instructions](../tests/test_data/arch_accuracy/README.md). A new family's
   fixture needs an oracle for its own inputs/outputs rather than forcing it into a logits harness.
   Run the full frontend and external-library suites; changes to shared converters also need the
   existing architecture fingerprints and numerical regressions.

See [debugging_accuracy.md](debugging_accuracy.md) for comparing intermediates. A static shape
match or successful one-token run is a useful initial check, not evidence of numerical correctness.

### 7. Integrate the verified implementation into the frontend

Keep the architecture definition separate from the plugin entry point throughout development.
Then follow [Promote the same implementation into OpenVINO](#promote-the-same-implementation-into-openvino):
move the source, register the same definition and retain its tests. The `ModelBuilder` and its
`node` calls do not change. If the plugin supplied converters, upstream those into the shared
converter table as well. Update supported-model documentation with the actual validation scope.

## Tensor operations and shapes

`node` is the single operation API. It looks up the operation's registered converter, which
constructs OpenVINO nodes immediately. `GgufValue` holds the resulting `ov::Output`;
`value.shape()`, `value.type()` and `value.ne(i)` read OpenVINO's inferred result. There is no
builder operation registry, per-operation method or separate shape implementation to maintain.

```cpp
auto sum = graph.node("GGML_OP_ADD", {a, b});
auto projected = graph.node("GGML_OP_MUL_MAT", {weight, sum});
auto experts = graph.node("GGML_OP_TOP_K", {scores}, 0, {{"k", int64_t{2}}});
```

Operands follow GGML order (weight before activation for matrix multiplication). Converters
implement ggml's result-type rules: `MUL_MAT` and `MUL_MAT_ID` produce F32; `TOP_K` and `ARGSORT`
produce I32 indices; type-preserving operations use their input type, and state writes use the
destination tensor's type. Architecture code does not supply an output type. OpenVINO outputs
carry the resulting types and shapes, including through custom `ConversionExtension` converters.

`op_case` selects a converter's existing semantic variant. Attributes supply operation parameters,
not inferred output metadata. Explicit types are needed only when the operation itself selects a
destination precision, such as a cast or `IM2COL`:

```cpp
auto half = graph.node("GGML_OP_CPY", {input}, 0, {{"dst_type", ov::element::f16}});
auto copied = graph.node("GGML_OP_CPY", {input, destination});
```

The first call is a cast; the second derives its type from `destination`. `IM2COL` likewise requires
`dst_type` alongside its convolution parameters. Older cgraph decoders may expose these two
operations' destination types through `output_type`; the shared converters accept that encoding
for compatibility. New operation support requires only a converter, without builder changes.
Missing converters report a conversion error.

Value shapes use OpenVINO order; `value.ne(i)` accesses reversed GGML dimensions. Weight values
preserve logical vector and expert axes. Missing optional weights return an empty value;
`tensors.require(name)` reports missing mandatory weights.

Input dimensions and operation-defining parameters remain explicit. For example, a reshape
specifies an OpenVINO-order pattern; `-1` requests an inferred dimension and `special_zero` copies
input dimensions at zero entries. Splitting attention heads while preserving a dynamic token
axis and the leading batch is:

```cpp
auto heads = graph.node("GGML_OP_RESHAPE", {projected}, 6, {{"reshape_target", std::vector<int64_t>{0, -1, n_heads, head_size}},
                         {"special_zero", true}});
auto merged = graph.node("GGML_OP_RESHAPE", {heads}, 2, {{"merge_heads", true}});
```

`PERMUTE` case 1 takes `perm` in OpenVINO axis order. `CONCAT` takes `concat_axis` in GGML order;
`REPEAT` takes integer `repeats`. Converters and OpenVINO validate these parameters. GGML
view/stride semantics need explicit slice/layout parameters. No representative token count or
intermediate output-shape metadata is supplied.

The SDK is not a drop-in implementation of llama.cpp's `llm_graph_context`. Map sublayers onto
shared blocks where applicable, then use `node` for new structure. Model-specific helpers may
name repeated fragments without extending the SDK's operation vocabulary.

`load()` parses the file and selects the architecture. `convert()` invokes the selected builder
with the frontend's current converters, then normalizes the constructed OpenVINO graph. Operation
extensions may therefore be registered between loading and conversion. Each conversion builds a
fresh graph; external libraries implementing the selected builder stay alive with the loaded input.

## Declare model contracts

Graph nodes alone do not describe how consumers should maintain state or form masks:

- `configure_decoder` records the resolved RoPE and sliding-window metadata automatically.
- For custom RoPE, call `configure_rope(config)` before emitting nodes. Set `config.per_op`
  for per-node tables and `config.is_imrope` for the multimodal position contract, then emit
  `node("GGML_OP_ROPE", ...)` with the matching mode and per-node `rope_config` attribute.
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
- For a real architecture, compare its outputs to llama.cpp on the same checkpoint before marking
  it verified: logits/generation for decoders, and features or pooled/projected outputs for encoders.
