# Extending the GGUF frontend

Extensions let a caller add architecture or projector builders, register operation converters, or change
normalization on an individual GGUF frontend. They can be registered as C++ objects or loaded
from a shared library, without rebuilding OpenVINO.

The architecture and builder headers under
[`dev_api/openvino/frontend/gguf`](../dev_api/openvino/frontend/gguf) are installed with the
frontend development package and exposed by `openvino::frontend::gguf`. This developer API does
not promise compatibility across releases: build an extension against the OpenVINO release
that will load it. The examples below use C++17.

## Choose an extension

| What you need | Extension | Register before |
|---|---|---|
| Accept a new GGUF architecture with the existing decoder topology | `ov::frontend::gguf::ArchitectureExtension` with a decoder definition | `load()` |
| Build a different topology or model family | `ArchitectureExtension` with a `ModelBuilder` factory | `load()` |
| Add or override one mmproj encoder/projector branch | `ov::frontend::gguf::ProjectorExtension` (derived from `ArchitectureExtension`) | `load()` |
| Add or override an operation converter | `ov::frontend::ConversionExtension` | `convert()` |
| Change state handling or adapt the model's input/output contract | `ov::frontend::DecoderTransformationExtension` | `convert()` |

Architecture and projector extensions supply computation for the native `.gguf` file path. They do not
change an already-built graph supplied as a `GgufDecoder`. Conversion and transformation
extensions apply to both input paths. Architecture or projector registration does not add weight quantization
formats, tokenization, host preprocessing, or a device execution backend.

The frontend also unwraps shared-library extensions and recursively registers extensions
attached to an `ov::BaseOpExtension`. A bare custom operation definition does not supply a
GGML converter. `TelemetryExtension` is accepted and stored, but the current GGUF implementation
does not invoke its callbacks. Other extension types have no handling in this frontend.

## Registration and conversion lifecycle

Use the same frontend instance for extension registration, loading and conversion:

```cpp
#include "openvino/frontend/gguf/extension/architecture.hpp"
#include "openvino/frontend/manager.hpp"

ov::frontend::FrontEndManager manager;
auto frontend = manager.load_by_framework("gguf");
frontend->add_extension(std::make_shared<ov::frontend::gguf::ArchitectureExtension>(
    "my-decoder", ov::frontend::gguf::RopeMode::Neox));
auto input_model = frontend->load("my-decoder.gguf");
auto model = frontend->convert(input_model);
```

`my-decoder` is an illustrative architecture name, not a supported checkpoint. This example
assumes its tensor names and computation fit the shared decoder builder.

The native file path proceeds as follows:

1. `load()` parses metadata and weights, resolves an architecture handler, and saves its
   definition and a snapshot of the projector registry in the input model. Register architecture
   and projector extensions before this step. Replacing a handler afterward affects subsequent
   loads, not an input model already loaded. Projector selection happens during `convert()`
   against the saved registry, once the mmproj coordinator resolves each declared modality.
2. `convert()` merges the built-in converters with the frontend's current conversion extensions,
   invokes the selected factory and calls `ModelBuilder::build()`. Each conversion builds a
   fresh graph. `GgufGraphContext::node()` invokes converters immediately, so their OpenVINO
   outputs provide inferred shapes and types to subsequent builder calls.
3. Normalization registers extension passes in registration order, after marking compressed
   floating-point constants and before `LowerSetRowsStateless` and the remaining built-in
   cleanup passes. `convert()` returns the normalized OpenVINO model.

Operation converters and transformation extensions can therefore be registered after `load()`
and before `convert()`. Register everything before `load()` when no delayed registration is needed.
Registrations belong to one frontend instance. Adding an extension to another frontend or to a
consumer that does not forward it to this instance does not configure this conversion.

## Reuse the decoder builder

For a new architecture with the existing decoder topology, the short constructor takes the
literal `general.architecture` string and its RoPE mode. `Normal` rotates consecutive pairs,
`Neox` rotates halves, and `Interleaved` selects interleaved multimodal RoPE. Determine the mode
from the reference implementation; registering a name does not establish numerical correctness.

The specified mode is passed to the decoder configuration, including for architecture names
absent from the built-in catalog. It is required when reusing the decoder builder.
Custom builders use `ArchitectureExtension(ArchitectureDefinition)` without a RoPE argument
and implement any positional encoding themselves. `ProjectorExtension` likewise requires
no decoder RoPE mode; its branch builder owns positional encoding.

When tensor names and metadata cannot determine an architectural choice, use
`make_decoder_architecture()` with an options callback:

```cpp
#include "openvino/frontend/gguf/extension/architecture.hpp"

using namespace ov::frontend::gguf;
auto definition = make_decoder_architecture(
    "my-decoder", RopeMode::Neox,
    [](const GgufMetadata&) {
        DecoderOptions options;
        options.geglu = true;
        options.qk_norm_after_rope = true;
        return options;
    });
frontend->add_extension(std::make_shared<ArchitectureExtension>(definition));
```

These options are illustrative; select them only if they reproduce the target architecture.
Unset optional fields retain native detection. The callback runs before dependent decoder
configuration is resolved. It changes architecture facts, not runtime dimensions or execution
plans. Available fields are defined in
[`decoder_options.hpp`](../dev_api/openvino/frontend/gguf/builder/decoder_options.hpp).

The architecture catalog has one supported list, with no maturity categories. Registration
adds builder support; checkpoint validation is documented separately. Follow the
[architecture validation guidance](adding_an_architecture.md#verifying-a-new-architecture)
and record the tested configurations and numerical results.

## Supply a custom topology

An `ArchitectureDefinition` contains:

| Field | Meaning |
|---|---|
| `id` | Unique handler identity used for registration and replacement |
| `architecture` | Required value of the file's `general.architecture` |
| `factory` | Function receiving a `BuildContext` and returning a `ModelBuilder` |
| `match` | Optional additional metadata predicate |

Implement `ModelBuilder::build()` using `GgufGraphContext`. Read the family's own metadata,
declare inputs, load tensors, emit operations, set outputs, and return `finish()`. The
[projector source](../examples/architecture_extension/projector.cpp) is a complete minimal
builder: it reads `projection.weight`, accepts a dynamic number of embeddings, and applies
`GGML_OP_MUL_MAT`. Its definition uses id `example.projector` and architecture `example-projector`.

`BuildContext` borrows metadata and weight tables for the synchronous factory/build call.
A builder may copy it for use in `build()`, as the example does, but must not use it afterward.
The returned graph retains weight storage. Require mandatory metadata and tensors explicitly;
optional scalar metadata getters return `std::optional`, and `tensors.require(name)` reports
missing weights.

A custom decoder can reuse `configure_decoder()`, `decoder_attention()` and `decoder_ffn()`
while owning its layer ordering. A different model family reads its own metadata and uses
generic operations without configuring a decoder. See the
[porting walkthrough](porting_a_llama_cpp_model.md) for complete builders, tensor layout rules,
state contracts, and promotion into the built-in catalog.

### Example: a whole decoder architecture in an extension

The example below supplies the whole model through a custom `ModelBuilder`: token embeddings,
every decoder layer, final normalization, and the output projection. It reuses the frontend's
attention and FFN blocks, but the extension decides how to connect them. It does not use
`make_decoder_architecture()` to select the built-in topology.

This follows the Qwen3 builder tested in
[`test_architecture_extension.cpp`](../tests/test_architecture_extension.cpp). It replaces the
built-in `qwen3` handler so it can load a Qwen3 GGUF file. For a new architecture, use its actual
`general.architecture` string and `RegistrationMode::Add`, and implement its layer order and
options according to the reference model.

Save this as `extension.cpp`:

```cpp
#include "openvino/frontend/gguf/builder/graph_context.hpp"
#include "openvino/frontend/gguf/extension/architecture.hpp"

namespace example {
using namespace ov::frontend::gguf;

class Qwen3Builder : public ModelBuilder {
public:
    explicit Qwen3Builder(const BuildContext& context) : m_context(context) {}

    std::shared_ptr<GgufGraph> build() override {
        GgufGraphContext graph(m_context);
        const auto dimensions = graph.configure_decoder(RopeMode::Neox);
        auto tensors = graph.tensors();
        auto cur = graph.build_inp_embd(tensors.require("token_embd.weight"));
        graph.build_inp_pos();
        graph.build_attn_inp_kv();

        for (int layer = 0; layer < dimensions.layers; ++layer) {
            auto norm = graph.build_norm(
                cur, tensors.layer(layer, "attn_norm.weight"), dimensions.norm_epsilon);
            cur = graph.node("GGML_OP_ADD", {graph.decoder_attention(layer, norm), cur});
            norm = graph.build_norm(
                cur, tensors.layer(layer, "ffn_norm.weight"), dimensions.norm_epsilon);
            cur = graph.node("GGML_OP_ADD", {graph.decoder_ffn(layer, norm), cur});
        }

        cur = graph.build_norm(cur, tensors.require("output_norm.weight"), dimensions.norm_epsilon);
        auto output_weight = tensors("output.weight");
        if (!output_weight) {
            output_weight = tensors.require("token_embd.weight");
        }
        graph.set_primary_output(graph.node("GGML_OP_MUL_MAT", {output_weight, cur}));
        return graph.finish();
    }

private:
    BuildContext m_context;
};

ArchitectureDefinition qwen3_architecture() {
    return {"qwen3", "qwen3", [](const BuildContext& context) {
                return std::make_shared<Qwen3Builder>(context);
            }};
}
}  // namespace example

OPENVINO_CREATE_EXTENSIONS(std::vector<ov::Extension::Ptr>{
    std::make_shared<ov::frontend::gguf::ArchitectureExtension>(
        example::qwen3_architecture(), ov::frontend::gguf::RegistrationMode::Replace)});
```

Build it as a shared library linked to `openvino::frontend::gguf`. Use the
[standalone CMake example](../examples/architecture_extension/CMakeLists.txt), changing its source
list to just `extension.cpp`. Then load the library on the frontend before loading the model,
as shown in [Build and load a shared library](#build-and-load-a-shared-library).
Register `GGUFMakeStateful` and `AdaptToGenAI` separately if the consumer needs them; supplying
the architecture does not select the state or input/output contract.

For a whole model that uses its own operations instead of shared decoder blocks, see the
[complete new-family builder](porting_a_llama_cpp_model.md#4-implement-a-whole-model-builder).
The runnable [extension examples](../examples/architecture_extension/README.md) include decoder,
whole-model projection and mmproj component plugins, generated inputs, and a CPU loader.
The whole-model projection example shows the same packaging
with separate builder and entry-point files.

### Matching and replacement

Matching first checks `general.architecture`, then the optional predicate. Two handlers with
different ids can share an architecture string only when their predicates make their claims
disjoint. For example, different whole-model formats under one architecture can have separate handler
ids. For ordinary mmproj vision/audio components, use `ProjectorExtension`; adding another
whole-model `clip` handler would overlap with the built-in coordinator.
If two handlers claim the same file, loading fails; neither registration order nor a more
specific predicate gives one priority.

`RegistrationMode::Add` is the default and rejects a duplicate id. To replace an existing
handler, preserve its id and explicitly use `RegistrationMode::Replace`:

```cpp
#include "openvino/frontend/gguf/extension/architecture.hpp"

using namespace ov::frontend::gguf;
auto replacement = make_decoder_architecture(
    "my-decoder", RopeMode::Neox,
    [](const GgufMetadata&) {
        DecoderOptions options;
        options.geglu = true;
        return options;
    });
frontend->add_extension(std::make_shared<ArchitectureExtension>(
    replacement, RegistrationMode::Replace));
```

This replaces the illustrative `my-decoder` handler registered above. Built-in handlers can be
replaced the same way; use their existing id and options appropriate to the target model.
Replacement of an unknown id fails. Assigning a new id to another handler for an already-claimed
architecture does not replace the old one and can cause an ambiguous match. Replacement affects
only this frontend's registry.

## Extend mmproj with a projector component

`ProjectorExtension` inherits from `ArchitectureExtension` and uses the same registration
and shared-library packaging. The frontend registers it in a separate projector registry.
It builds one vision or audio encoder/projector branch; the `clip` architecture handler
coordinates all declared branches and finishes the model.

| Definition | Selection | Computation supplied |
|---|---|---|
| `ArchitectureDefinition` | `general.architecture`, then optional metadata predicate | A complete `ModelBuilder` |
| `ProjectorDefinition` | Architecture, modality, resolved projector type, then optional metadata predicate | One branch appended to a shared `GgufGraphContext` |

A `ProjectorDefinition` has `id`, `architecture`, `modality`, `projector_type`, `build`,
and optional `match`. Modality is `vision` or `audio`. Its callback returns `ProjectorResult`
containing an output and optional string configuration under that modality's prefix.
The coordinator names the output `vision.embeddings` or `audio.embeddings`, preserves source
metadata, and combines both branches when the file declares both encoders.

### Reuse an existing encoder/projector topology

This example accepts `clip.vision.projector_type = "my-gemma3"` using the built-in Gemma3
branch. The file must provide the compatible encoder metadata and weights; adding an alias
does not translate a different tensor layout or computation.

```cpp
#include "openvino/frontend/gguf/extension/projector.hpp"

using namespace ov::frontend::gguf;
ProjectorDefinition definition{
    "clip.vision.my-gemma3", "clip", "vision", "my-gemma3",
    [](GgufGraphContext& graph) {
        return build_builtin_projector(graph, "vision", "gemma3");
    },
    {}};
frontend->add_extension(std::make_shared<ProjectorExtension>(definition));
auto model = frontend->convert(frontend->load("mmproj-my-gemma3.gguf"));
```

`build_builtin_projector()` appends a complete built-in encoder/projector branch. A new
topology can instead use the shared graph operations and tensor table directly. The callback
must leave `finish()` and registration of its primary output to the coordinator. Extra
inputs must use unique names. If the output directly aliases an input, the coordinator preserves its input name while
adding the branch output name.

### Build a new branch

This minimal example illustrates the component contract by scaling caller-provided audio
embeddings. It does not implement a checkpoint's audio encoder or preprocessing:

```cpp
#include "openvino/frontend/gguf/extension/projector.hpp"

using namespace ov::frontend::gguf;
ProjectorDefinition definition{
    "clip.audio.example", "clip", "audio", "example-audio",
    [](GgufGraphContext& graph) {
        auto input = graph.add_input("audio.example_input", ov::element::f32, {1, 1, -1, 8});
        auto output = graph.node("GGML_OP_SCALE", {input}, 0, {{"scale", 2.0f}, {"bias", 0.0f}});
        return ProjectorResult{output, {{"audio.merge", "1"}}};
    },
    {}};
frontend->add_extension(std::make_shared<ProjectorExtension>(definition));
```

A file declaring `clip.has_audio_encoder = true` and the corresponding projector type selects
this branch. A built-in vision branch in the same file remains active. A real implementation
reads `graph.metadata()` and `graph.tensors()` and reproduces the encoder/projector computation.

### Selection, replacement and support lists

The mmproj coordinator resolves `clip.projector_type` first, falling back to
`clip.vision.projector_type` or `clip.audio.projector_type` when the global value is empty.
Legacy `qwen2.5o` resolves to `qwen2.5vl_merger` for vision and `qwen2a` for audio before
component lookup. A metadata predicate may distinguish variants of the same resolved type;
two matching handlers are an error, and unknown types fail conversion.

`RegistrationMode::Add` rejects duplicate component ids. To replace only the built-in Gemma3
vision branch, set the definition id to `clip.vision.gemma3`, keep its selectors, and register
with `RegistrationMode::Replace`. Other projector types and the audio branch remain active.
Replacement of an unknown component id fails. Architecture and projector ids belong to
separate registries; replacing a component does not replace the `clip.mmproj` coordinator.

The architecture list is derived by `ArchRegistry::supported_archs()` and the component list
by `ArchRegistry::projectors().supported_projectors()`. Adding a `ProjectorExtension` extends
the component list without adding another architecture entry. Both registries are isolated
to one frontend instance, and loaded input models keep their registration snapshot.

The supported mmproj formats use `general.architecture = "clip"`. This is a format convention,
not a guarantee for future projectors or a claim that every `clip` projector is supported.
An extension for another architecture also needs a whole-model coordinator for that architecture.
See [mmproj support and contracts](mmproj.md) for the supported types, preprocessing boundary,
metadata and GenAI adaptation.

### Package a projector extension

The [mmproj plugin example](../examples/architecture_extension/mmproj_extension.cpp) exports
`ProjectorExtension` using `OPENVINO_CREATE_EXTENSIONS`. The standalone CMake project builds
it as `gguf_mmproj_extension` alongside decoder and whole-model projection examples.
It includes a Gemma3 reuse entry and a custom audio projection branch; the
[example README](../examples/architecture_extension/README.md) provides generated inputs and runner commands:

```sh
cmake --build /tmp/gguf-extension --target gguf_mmproj_extension
```

After configuring the project as described below, load `libgguf_mmproj_extension.so` on the
frontend before `load()`. The plugin accepts `example-gemma3` with Gemma3-compatible metadata and tensors, or
`example-linear` with caller-provided audio embeddings and a projection weight. Compile against the matching GGUF frontend
package; the developer API has no compatibility guarantee across releases.

## Add or override an operation converter

Use the generic `ov::frontend::ConversionExtension` with a converter taking
`const ov::frontend::NodeContext&` and returning an indexed `ov::OutputVector`. The named-output
converter overloads are not used by GGUF. Match the exact operation string emitted by the graph,
such as `GGML_OP_SCALE` or `GGML_UNARY_OP_SILU`.

The following example supplies a custom operation for an external builder:

```cpp
#include "openvino/core/except.hpp"
#include "openvino/frontend/extension/conversion.hpp"
#include "openvino/op/negative.hpp"

frontend->add_extension(std::make_shared<ov::frontend::ConversionExtension>(
    "EXAMPLE_NEGATE",
    [](const ov::frontend::NodeContext& context) -> ov::OutputVector {
        OPENVINO_ASSERT(context.get_input_size() == 1, "EXAMPLE_NEGATE requires one input");
        auto result = std::make_shared<ov::op::v0::Negative>(context.get_input(0));
        result->set_friendly_name(context.get_name());
        return {result};
    }));
```

Its builder can now call `graph.node("EXAMPLE_NEGATE", {value})`. Use the base `NodeContext`
methods for inputs, the operation name and typed attributes, including
`get_attribute<int>("op_case", 0)` when needed. GGUF's concrete `NodeContext` is an internal
header; external extensions should use the installed base interface and input outputs' inferred
shapes/types.

An extension overrides a built-in converter with the same operation string. If several conversion
extensions register that string, the last registration wins. Return outputs in decoder order and
preserve the operation's semantics, element types and layout. For a built-in addition, follow
[how_to_add_op.md](how_to_add_op.md).

## Register normalization passes

Despite its generic name, `DecoderTransformationExtension` runs on the constructed OpenVINO
graph during GGUF normalization. It can wrap an existing pass or a function taking an
`std::shared_ptr<ov::Model>` and returning whether it changed the model.

A plain conversion keeps cache state as explicit inputs and outputs. To expose OpenVINO state
and then the GenAI text input/output contract, register these passes in this order:

```cpp
#include "openvino/frontend/extension/decoder_transformation.hpp"
#include "openvino/frontend/gguf/adapt_to_genai.hpp"
#include "openvino/frontend/gguf/make_stateful.hpp"

frontend->add_extension(std::make_shared<ov::frontend::DecoderTransformationExtension>(
    ov::frontend::gguf::pass::GGUFMakeStateful()));
frontend->add_extension(std::make_shared<ov::frontend::DecoderTransformationExtension>(
    ov::frontend::gguf::pass::AdaptToGenAI()));
auto model = frontend->convert(frontend->load("model.gguf"));
```

`GGUFMakeStateful` consumes cache-write placeholders before the default stateless lowering.
`AdaptToGenAI` requires an already-stateful decoder and adapts its inputs and logits; it does
not create a tokenizer or a pipeline. These passes are caller choices, not properties of an
architecture definition. They do not adapt arbitrary encoder inputs. Recurrent-state restrictions depend on the architecture and adaptation path; see the
[current runtime limitations](supported_models.md#runtime-limitations).

## Build and load a shared library

The [standalone example](../examples/architecture_extension/CMakeLists.txt) links to
`openvino::frontend::gguf` and exports its extension list with `OPENVINO_CREATE_EXTENSIONS` in
[`extension.cpp`](../examples/architecture_extension/extension.cpp). The list may contain
architecture, projector, conversion and transformation extensions together.

From the repository root, with a matching OpenVINO development installation containing the
GGUF frontend:

```sh
cmake -S src/frontends/gguf/examples/architecture_extension -B /tmp/gguf-extension \
    -DOpenVINO_DIR=/path/to/openvino/runtime/cmake
cmake --build /tmp/gguf-extension
```

Load the produced library through the base frontend interface, which exposes the library-path
overload of `add_extension()`:

```cpp
#include "openvino/frontend/manager.hpp"

ov::frontend::FrontEndManager manager;
auto frontend = manager.load_by_framework("gguf");
frontend->add_extension("/tmp/gguf-extension/libgguf_projector_extension.so");
auto model = frontend->convert(frontend->load("projector.gguf"));
```

The `.so` path above is for Linux; use the produced library path for your platform and build
configuration. The example expects the illustrative `example-projector` file contract, not an
arbitrary multimodal checkpoint. Shared-library wrappers retain the library; a natively loaded
input model also retains the libraries needed by its selected builder.

## Validate and troubleshoot

Compile and load the actual extension library, convert a matching file, and compare outputs with
the reference at several input lengths. For decoders, include prefill and cached decode; for
encoders or projectors, compare the relevant features. Registration and successful compilation
alone do not establish accuracy. Existing examples are covered by
[`test_architecture_extension.cpp`](../tests/test_architecture_extension.cpp),
[`test_builder_api.cpp`](../tests/test_builder_api.cpp),
[`test_extensions.cpp`](../tests/test_extensions.cpp), and
[`test_mmproj.cpp`](../tests/test_mmproj.cpp).

| Symptom | Check |
|---|---|
| GGUF is absent from automatic frontend selection | Select `load_by_framework("gguf")` and ensure the frontend was built and installed |
| Unsupported architecture or no builder | Register the architecture before `load()`; check the literal architecture string and metadata predicate |
| Duplicate handler id or unknown replacement | Use `Add` for new ids and `Replace` only for existing ids |
| Two handlers claim a file | Make predicates disjoint or explicitly replace the existing handler |
| Unsupported mmproj projector | Register a `ProjectorExtension` before `load()`; check architecture, modality and resolved projector type |
| Multiple projector handlers claim a type | Make component predicates disjoint or replace the existing component id |
| Missing operation converter | Register the exact emitted operation string on the same frontend before `convert()` |
| Cache remains stateless | Register `GGUFMakeStateful` before conversion, while cache-write placeholders still exist |
| Conversion succeeds but outputs differ | Follow [debugging_accuracy.md](debugging_accuracy.md) and check options, layouts, preprocessing and state against the reference |
