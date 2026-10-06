# OpenVINO GGUF Frontend

The GGUF frontend converts native `.gguf` model files and graphs supplied through
the `GgufDecoder` interface into `ov::Model` objects. It supports two input paths:

* **Native GGUF:** parses the file's metadata and weights, selects a model builder
  from the architecture registry, and constructs a graph for conversion.
* **Supplied decoder:** consumes a `GgufDecoder`, such as the graph decoder used by
  the llama.cpp OpenVINO backend.

Both paths use the same operation translators. Their architecture coverage is
validated separately; consult [supported models and limitations](docs/supported_models.md)
for supported configurations and validation coverage.

The native builder has a single [supported architecture list](docs/supported_models.md#supported-architectures).
Projector types have a separate [support list and extension API](docs/mmproj.md).
Checkpoint results and runtime limitations are documented alongside those lists.

## Loading a model

Select this frontend explicitly by name:

```python
from openvino.frontend import FrontEndManager

manager = FrontEndManager()
frontend = manager.load_by_framework("gguf")
input_model = frontend.load("model.gguf")
model = frontend.convert(input_model)
```

The GGUF frontend is currently hidden from automatic frontend discovery, so
`ov.Core.read_model()` does not select it for `.gguf` files. See the
[frontend manager](../common/src/manager.cpp) and
[GGUF frontend implementation](src/frontend.cpp).

Conversion produces a stateless graph by default. Callers that need an OpenVINO
KV cache can register a decoder transformation extension using
[`GGUFMakeStateful`](include/openvino/frontend/gguf/make_stateful.hpp).
[`AdaptToGenAI`](include/openvino/frontend/gguf/adapt_to_genai.hpp) then gives a stateful
language model the OpenVINO GenAI input contract. Multimodal projector files convert to
encoder models; see [native multimodal conversion](docs/mmproj.md).
The [extension guide](docs/extensions.md) explains architecture and projector handlers,
operation converters, normalization passes, registration timing and shared-library loading.
[Internal operation guidance](docs/internal_ops.md) describes lowering and serialization constraints.

## Source layout

| Location | Purpose |
| --- | --- |
| [include](include/openvino/frontend/gguf/) | Frontend, decoder, tokenizer metadata, and conversion-pass interfaces |
| [dev_api](dev_api/openvino/frontend/gguf/) | Model-builder, architecture-extension and projector-extension APIs; compatibility across releases is not guaranteed |
| [src/builder](src/builder/) | Architecture/projector registries, native model builders, and graph construction |
| [src/quant](src/quant/) | GGUF parsing, weight loading, and quantization handling |
| [src/op](src/op/) and [op_table.cpp](src/op_table.cpp) | Operation translators and their registration |
| [src/pass](src/pass/) | Stateful conversion, GenAI adaptation, and graph cleanup |
| [tests](tests/) | Operation, architecture, multimodal, quantization, and extension tests, fixtures, fixture generators and llama.cpp oracles |
| [examples](examples/) | Architecture and projector extension examples |

## Development guides

* [Register and combine frontend extensions](docs/extensions.md).
* [Build and run extension examples](examples/architecture_extension/README.md).
* [Add an operation translator](docs/how_to_add_op.md).
* [Add a built-in architecture](docs/adding_an_architecture.md).
* [Port a llama.cpp model or build an external architecture extension](docs/porting_a_llama_cpp_model.md).
* [Debug accuracy differences](docs/debugging_accuracy.md).
* [Understand weight formats and precision](docs/quantization.md).
* [Convert multimodal projectors](docs/mmproj.md).
* [Build, test, and regenerate references](docs/testing.md).

## Building and testing

The frontend defaults to enabled. C++ tests require a shared-library build and `ENABLE_TESTS=ON`.
See [testing](docs/testing.md) for exact build commands, both test binaries, fixture prerequisites,
model-hub runs, and acceptance criteria. Generated architecture headers must be present for
fingerprint coverage; an otherwise successful test run can skip that suite.
