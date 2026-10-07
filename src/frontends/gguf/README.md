# OpenVINO GGUF Frontend

The GGUF frontend converts native `.gguf` model files and graphs supplied through
the `GgufDecoder` interface into `ov::Model` objects. It supports two input paths:

* **Native GGUF:** parses the file's metadata and weights, selects a model builder
  from the architecture registry, and constructs a graph for conversion.
* **Supplied decoder:** consumes a `GgufDecoder`, such as the graph decoder used by
  the llama.cpp OpenVINO backend.

Both paths use the same operation translators. Their architecture coverage is
validated separately; consult [supported models and limitations](docs/supported_models.md)
for verified configurations and experimental support.

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
The [frontend API](include/openvino/frontend/gguf/frontend.hpp) documents supported
extensions; [internal operation guidance](docs/internal_ops.md) describes lowering
and serialization constraints.

## Source layout

| Location | Purpose |
| --- | --- |
| [include](include/openvino/frontend/gguf/) | Frontend, decoder, tokenizer metadata, and conversion-pass interfaces |
| [dev_api](dev_api/openvino/frontend/gguf/) | Model-builder and architecture-extension APIs; compatibility across releases is not guaranteed |
| [src/builder](src/builder/) | Architecture registry, native model builders, and graph construction |
| [src/quant](src/quant/) | GGUF parsing, weight loading, and quantization handling |
| [src/op](src/op/) and [op_table.cpp](src/op_table.cpp) | Operation translators and their registration |
| [src/pass](src/pass/) | Stateful conversion, GenAI adaptation, and graph cleanup |
| [tests](tests/) | Operation, architecture, quantization, and extension tests and fixtures |
| [examples](examples/) | Architecture-extension example |

## Development guides

* [Add an operation translator](docs/how_to_add_op.md).
* [Add a built-in architecture](docs/adding_an_architecture.md).
* [Port a llama.cpp model or build an external architecture extension](docs/porting_a_llama_cpp_model.md).
* [Debug accuracy differences](docs/debugging_accuracy.md).
* [Generate architecture accuracy fixtures](tests/test_data/arch_accuracy/README.md).

## Building and testing

Follow the repository [build guide](../../../docs/dev/build.md) for prerequisites.
The frontend is controlled by `ENABLE_OV_GGUF_FRONTEND`. From the repository root,
configure a shared-library build with tests enabled:

```sh
cmake -S . -B build -DENABLE_OV_GGUF_FRONTEND=ON -DENABLE_TESTS=ON -DBUILD_SHARED_LIBS=ON
cmake --build build --target openvino_gguf_frontend ov_gguf_frontend_tests
```

The GGUF frontend unit-test target is unavailable in static builds. See
[test configuration](tests/CMakeLists.txt) for fixture requirements and the
[operation testing guide](docs/how_to_add_op.md#test-and-the-coverage-gate) for
reference generation and coverage checks.
