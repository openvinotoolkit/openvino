# Extending the GGUF frontend

Extensions let a caller add architecture builders, register operation converters, or change
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
| Add or override an operation converter | `ov::frontend::ConversionExtension` | `convert()` |
| Change state handling or adapt the model's input/output contract | `ov::frontend::DecoderTransformationExtension` | `convert()` |

An architecture extension supplies computation for the native `.gguf` file path. It does not
change an already-built graph supplied as a `GgufDecoder`. Conversion and transformation
extensions apply to both input paths. Architecture registration does not add weight quantization
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
   definition in the input model. Register architecture extensions before this step. Replacing
   a handler afterward affects subsequent loads, not an input model already loaded.
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

Definitions default to `Maturity::Experimental`, which emits a warning on loading. Setting
`Maturity::Verified` describes validation status; it does not run validation or alter the builder.
Use the [architecture validation guidance](adding_an_architecture.md#verifying-a-new-architecture)
before declaring a real model verified.

## Supply a custom topology

An `ArchitectureDefinition` contains:

| Field | Meaning |
|---|---|
| `id` | Unique handler identity used for registration and replacement |
| `architecture` | Required value of the file's `general.architecture` |
| `factory` | Function receiving a `BuildContext` and returning a `ModelBuilder` |
| `match` | Optional additional metadata predicate |
| `maturity` | Experimental or verified validation status |

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

### Matching and replacement

Matching first checks `general.architecture`, then the optional predicate. Two handlers with
different ids can share an architecture string only when their predicates make their claims
disjoint. For example, vision and audio variants of `clip` can have separate handler ids.
If two handlers claim the same file, loading fails; neither registration order nor a more
specific predicate gives one priority.

`RegistrationMode::Add` is the default and rejects a duplicate id. To replace an existing
handler, preserve its id and explicitly use `RegistrationMode::Replace`:

```cpp
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
architecture definition. They do not adapt arbitrary encoder inputs. Recurrent-state models
retain the batch-one and beam-search restrictions documented in
[`make_stateful.hpp`](../include/openvino/frontend/gguf/make_stateful.hpp).

## Build and load a shared library

The [standalone example](../examples/architecture_extension/CMakeLists.txt) links to
`openvino::frontend::gguf` and exports its extension list with `OPENVINO_CREATE_EXTENSIONS` in
[`extension.cpp`](../examples/architecture_extension/extension.cpp). The list may contain
architecture, conversion and transformation extensions together.

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
[`test_builder_api.cpp`](../tests/test_builder_api.cpp), and
[`test_extensions.cpp`](../tests/test_extensions.cpp).

| Symptom | Check |
|---|---|
| GGUF is absent from automatic frontend selection | Select `load_by_framework("gguf")` and ensure the frontend was built and installed |
| Unsupported architecture or no builder | Register the architecture before `load()`; check the literal architecture string and metadata predicate |
| Duplicate handler id or unknown replacement | Use `Add` for new ids and `Replace` only for existing ids |
| Two handlers claim a file | Make predicates disjoint or explicitly replace the existing handler |
| Missing operation converter | Register the exact emitted operation string on the same frontend before `convert()` |
| Cache remains stateless | Register `GGUFMakeStateful` before conversion, while cache-write placeholders still exist |
| Conversion succeeds but outputs differ | Follow [debugging_accuracy.md](debugging_accuracy.md) and check options, layouts, preprocessing and state against the reference |
