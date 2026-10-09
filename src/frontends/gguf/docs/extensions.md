# Extending the GGUF frontend

Extensions add architecture or projector builders, operation converters or normalization passes to
one GGUF frontend instance, as C++ objects or from a shared library, without rebuilding OpenVINO.
The builder headers in [`dev_api/openvino/frontend/gguf`](../dev_api/openvino/frontend/gguf) ship
with the frontend development package (`openvino::frontend::gguf`) and have no compatibility
guarantee across releases: build against the release that will load the extension (C++17).

## Choose an extension

| What you need | Extension | Register before |
|---|---|---|
| New architecture with the existing decoder topology | `ov::frontend::gguf::ArchitectureExtension` with a decoder definition | `load()` |
| Different topology or model family | `ArchitectureExtension` with a `ModelBuilder` factory | `load()` |
| Add or override one mmproj encoder/projector branch | `ov::frontend::gguf::ProjectorExtension` | `load()` |
| Add or override an operation converter | `ov::frontend::ConversionExtension` | `convert()` |
| Prepare a stateful decoder with GenAI IO | `ov::frontend::gguf::GenAIExtension` | `convert()` |
| Change state handling or the model interface | `ov::frontend::DecoderTransformationExtension` | `convert()` |

Architecture and projector extensions apply only to native `.gguf` files; converters and passes
apply to both input paths. None of them add quantization formats, tokenization, preprocessing or a
device backend. Extensions attached to an `ov::BaseOpExtension` and shared-library wrappers are
unwrapped; a bare custom operation supplies no GGML converter. `TelemetryExtension` is stored but
never invoked; other types are ignored.

## Lifecycle

Use one frontend instance for registration, loading and conversion:

```cpp
#include "openvino/frontend/gguf/extension/architecture.hpp"
#include "openvino/frontend/manager.hpp"

ov::frontend::FrontEndManager manager;
auto frontend = manager.load_by_framework("gguf");
frontend->add_extension(std::make_shared<ov::frontend::gguf::ArchitectureExtension>(
    "my-decoder", ov::frontend::gguf::RopeMode::Neox));
auto model = frontend->convert(frontend->load("my-decoder.gguf"));
```

1. `load()` parses the file, resolves the architecture handler and saves it with a snapshot of the
   projector registry. Later registrations affect only later loads.
2. `convert()` merges built-in and registered converters, builds a fresh graph through the selected
   `ModelBuilder`, and resolves each mmproj modality against the saved projector registry.
   `GgufGraphContext::node()` invokes converters immediately, so builders see inferred shapes.
3. Normalization runs extension passes in registration order, after marking compressed constants and
   before `LowerSetRowsStateless` and the remaining cleanup.

Registrations on another frontend, or on a consumer that does not forward them, have no effect.

## Architecture extensions

The short constructor takes the literal `general.architecture` and its RoPE mode. When tensors and
metadata cannot determine a choice, pass a definition with options:

```cpp
using namespace ov::frontend::gguf;
auto definition = make_decoder_architecture(
    "my-decoder", RopeMode::Neox,
    [](const GgufMetadata&) {
        DecoderOptions options;
        options.geglu = true;
        return options;
    });
frontend->add_extension(std::make_shared<ArchitectureExtension>(definition));
```

`ArchitectureExtension(ArchitectureDefinition)` also accepts a custom `ModelBuilder` factory with no
RoPE mode. See [architectures.md](architectures.md) for options, builders, porting and validation.

### Matching and replacement

A handler matches on `general.architecture`, then its optional predicate. Handlers with different
ids may share an architecture only with disjoint predicates; if two claim one file, loading fails
regardless of order. For mmproj branches use `ProjectorExtension`, not another `clip` handler.

`RegistrationMode::Add` (default) rejects a duplicate id. To replace a handler, including a built-in,
keep its id and register with `RegistrationMode::Replace`:

```cpp
frontend->add_extension(std::make_shared<ArchitectureExtension>(definition, RegistrationMode::Replace));
```

Replacing an unknown id fails; a new id for a claimed architecture causes an ambiguous match.
Replacement affects only this frontend's registry, and `ArchRegistry::supported_archs()` reflects it.

## Extend mmproj with a projector component

`ProjectorExtension` derives from `ArchitectureExtension` and is packaged the same way, but registers
in a separate projector registry. It builds one vision or audio branch; the `clip` coordinator
selects branches as described in [mmproj.md](mmproj.md#files-with-vision-and-audio-encoders), names
outputs `vision.embeddings` / `audio.embeddings` and finishes the model.

A `ProjectorDefinition` has `id`, `architecture`, `modality` (`vision` or `audio`),
`projector_type`, `build` and optional `match`. `build` appends to the shared `GgufGraphContext` and
returns `ProjectorResult`: the output and optional string configuration under the modality prefix.
It must not call `finish()` or register the primary output; extra inputs need unique names. A
predicate may distinguish variants of one resolved type; two matching handlers are an error.

Reuse a built-in topology for a new projector type when the metadata and weights are compatible:

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
```

A new topology instead builds its branch from `graph.metadata()`, `graph.tensors()` and `graph.node()`:

```cpp
ProjectorDefinition audio{
    "clip.audio.example", "clip", "audio", "example-audio",
    [](GgufGraphContext& graph) {
        auto input = graph.add_input("audio.example_input", ov::element::f32, {1, 1, -1, 8});
        auto output = graph.node("GGML_OP_SCALE", {input}, 0, {{"scale", 2.0f}, {"bias", 0.0f}});
        return ProjectorResult{output, {{"audio.merge", "1"}}};
    },
    {}};
```

To replace one built-in branch, reuse its id (for example `clip.vision.gemma3`) with
`RegistrationMode::Replace`; other branches and the `clip.mmproj` coordinator stay active. The
component list comes from `ArchRegistry::projectors().supported_projectors()`. An architecture other
than `clip` would also need its own whole-model coordinator.

## Add or override an operation converter

Register an `ov::frontend::ConversionExtension` whose converter takes `const NodeContext&` and returns
an indexed `ov::OutputVector`; named-output overloads are unused. Match the exact emitted string:

```cpp
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

A builder can then call `graph.node("EXAMPLE_NEGATE", {value})`. Use the base `NodeContext`
interface (`get_attribute<int>("op_case", 0)` for the case); GGUF's concrete one is internal. A
same-named extension overrides the built-in converter and the last registration wins. Preserve the
operation's semantics, types and layout. For a built-in converter see [how_to_add_op.md](how_to_add_op.md).

## Register normalization passes

`DecoderTransformationExtension` wraps a pass or a `std::shared_ptr<ov::Model>` function and runs
during normalization, before built-in lowerings. For a GenAI-ready decoder, register
`GenAIExtension` instead: it owns stateful conversion and adaptation after normalization.
Do not also register `GGUFMakeStateful` or `AdaptToGenAI` on that frontend. Register one
`GenAIExtension` per frontend instance; it applies to every conversion on that instance.
Its embedding lookup getters describe the latest successful conversion.
See [runtime.md](runtime.md#stateful-and-genai-conversion).

## Build and load a shared library

Export extensions with `OPENVINO_CREATE_EXTENSIONS`, as in
[`extension.cpp`](../examples/architecture_extension/extension.cpp); one list may mix all types.
The [example README](../examples/architecture_extension/README.md) builds the decoder, whole-model
projection and mmproj plugins and runs them. Load a library through the base frontend interface:

```cpp
frontend->add_extension("/tmp/gguf-extensions/libgguf_projector_extension.so");
auto model = frontend->convert(frontend->load("projector.gguf"));
```

Loaded input models keep the libraries their builder needs alive.

## Validate and troubleshoot

Load the actual library, convert a matching file and compare with the reference as described in
[architectures.md](architectures.md#validate). Existing coverage lives in
[`test_architecture_extension.cpp`](../tests/test_architecture_extension.cpp),
[`test_builder_api.cpp`](../tests/test_builder_api.cpp),
[`test_extensions.cpp`](../tests/test_extensions.cpp),
[`test_mmproj.cpp`](../tests/test_mmproj.cpp) and `ov_gguf_architecture_library_tests`.

| Symptom | Check |
|---|---|
| GGUF absent from automatic frontend selection | Use `load_by_framework("gguf")`; the frontend is hidden from `read_model()` |
| Unsupported architecture | Register before `load()`; check the literal string and predicate |
| Duplicate id or unknown replacement | `Add` for new ids, `Replace` only for existing ids |
| Two handlers claim a file | Make predicates disjoint or replace the existing handler |
| Unsupported mmproj projector | Register a `ProjectorExtension` before `load()`; check modality and resolved type |
| Missing operation converter | Register the exact string on the same frontend before `convert()` |
| Cache remains stateless | Register `GenAIExtension` for GenAI IO, or `GGUFMakeStateful` for GGUF IO, before `convert()` |
| Outputs differ from the reference | Follow [debugging_accuracy.md](debugging_accuracy.md) |
