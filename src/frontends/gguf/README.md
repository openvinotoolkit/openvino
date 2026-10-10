# OpenVINO GGUF Frontend

The GGUF frontend converts models into `ov::Model` from two input paths that share operation
translators:

* **Native GGUF:** parses a `.gguf` file's metadata and weights and builds the graph through the
  architecture registry. llama.cpp is not needed.
* **Supplied decoder:** consumes a `GgufDecoder`, such as the graph decoder of the llama.cpp
  OpenVINO backend.

Architecture coverage is validated separately for each path; see [supported models](docs/supported_models.md)
and, for multimodal projector files, [supported projectors](docs/mmproj.md#supported-projectors).

## Loading a model

The frontend is hidden from automatic discovery, so `ov.Core.read_model()` does not select it for
`.gguf` files. Select it by name:

```python
from openvino.frontend import FrontEndManager

frontend = FrontEndManager().load_by_framework("gguf")
model = frontend.convert(frontend.load("model.gguf"))
```

The result is a stateless graph with llama.cpp-style inputs and explicit cache inputs/outputs.
In C++, register [`GenAIExtension`](include/openvino/frontend/gguf/extension/genai.hpp) to obtain a stateful model with the
OpenVINO GenAI interface; this extension is not available from Python. Tokenizer metadata is attached
to the converted model's rt_info. See [running converted models](docs/runtime.md) for both contracts.
Projector files convert to encoder models; see [multimodal conversion](docs/mmproj.md).

## Source layout

| Location | Purpose |
| --- | --- |
| [include](include/openvino/frontend/gguf/) | Frontend, decoder, tokenizer metadata and conversion-pass interfaces |
| [dev_api](dev_api/openvino/frontend/gguf/) | Builder, architecture and projector extension APIs; no cross-release compatibility |
| [src/builder](src/builder/) | Registries, native model builders and graph construction |
| [src/quant](src/quant/) | GGUF parsing, weight loading and quantization |
| [src/op](src/op/) and [op_table.cpp](src/op_table.cpp) | Operation translators and their registration |
| [src/pass](src/pass/) | Stateful conversion, GenAI adaptation and graph cleanup |
| [tests](tests/) | Unit, accuracy and extension tests, fixtures, generators and llama.cpp oracles |
| [examples](examples/) | Architecture and projector extension examples |

## Guides

* [Run converted models: state, GenAI, tokenizer, internal operations](docs/runtime.md)
* [Supported models](docs/supported_models.md) and [multimodal projectors](docs/mmproj.md)
* [Add or port an architecture](docs/architectures.md)
* [Register extensions and build plugins](docs/extensions.md), with [runnable examples](examples/architecture_extension/README.md)
* [Add an operation translator](docs/how_to_add_op.md)
* [Weight formats and precision](docs/quantization.md)
* [Debug accuracy differences](docs/debugging_accuracy.md)
* [Build, test and regenerate references](docs/testing.md)

The frontend is enabled by default. C++ tests need `ENABLE_TESTS=ON` and a shared-library build;
architecture fingerprint tests skip unless their generated headers are present.
