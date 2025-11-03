# OpenVINO™ Paddle Frontend

OpenVINO Paddle Frontend is one of the OpenVINO Frontend libraries created for the Baidu PaddlePaddle™ framework.
The component supports both legacy PaddlePaddle models and the new PP-OCRv5 format.

## Supported Model Formats

### 1. PP-OCRv5 Format (New)
The new format consists of three files in a directory:
- `inference.json` - Model architecture and operators
- `inference.yml` - Model metadata and configuration
- `inference.pdiparams` - Model weights and parameters

### 2. Legacy Format
The legacy format consists of:
- `inference.pdmodel` - Model architecture and operators
- `inference.pdiparams` - Model weights and parameters
- `inference.pdiparams.info` - Optional parameter information file

The component is responsible for:
 * Paddle Reader - reads PaddlePaddle models (both JSON and protobuf formats) and parses them to the frontend InputModel. [Learn more about Paddle Frontend architecture.](./docs/paddle_frontend_architecture.md).
 * Paddle Converter - decodes PaddlePaddle models and operators and maps them semantically to the OpenVINO opset. [Learn more about the operator mapping flow.](./docs/operation_mapping_flow.md).

OpenVINO Paddle Frontend uses [the common coding style rules](../../../docs/dev/coding_style.md).

## Key contacts

People from the [openvino-ie-paddle-maintainers](https://github.com/orgs/openvinotoolkit/teams/openvino-ie-paddle-maintainers) have the rights to approve and merge PRs to the Paddle frontend component. They can assist with any questions about the component.

## Components

OpenVINO Paddle Frontend has the following structure:
 * [docs](./docs) contains developer documentation for the component.
 * [include](./include) contains module API and detailed information about the provided API.
 * [src](./src) folder contains sources of the component.
 * [tests](./tests) contains tests for the component. To get more information, read [How to run and add tests](./docs/tests.md) page.

## Platform-Specific Build Instructions

### macOS Build Requirements
- CMake 3.13 or higher
- Xcode Command Line Tools
- yaml-cpp (via Homebrew: `brew install yaml-cpp`)

#### Build Steps on macOS
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=/opt/homebrew \
      -DENABLE_TESTS=ON \
      ..
cmake --build . --parallel 8
```

#### Code Signing on macOS
Libraries and test binaries need to be signed for execution:
```bash
cd bin/arm64/Release/
# Sign all OpenVINO libraries
find . -name "libopenvino*.dylib" -exec codesign -f -s - {} \;
# Sign test binaries
codesign -f -s - paddle_format_test
```

## Debug capabilities

Developers can use OpenVINO Model debug capabilities that are described in the [OpenVINO Model User Guide](https://docs.openvino.ai/2025/openvino-workflow/running-inference/model-representation.html#model-debug-capabilities).

## Tutorials

 * [How to support a new operator](./docs/operation_mapping_flow.md)
 * [How to run and add tests](./docs/tests.md)

## See also
 * [OpenVINO™ README](../../README.md)
 * [OpenVINO Core Components](../README.md)
 * [Developer Documentation](../../../docs/dev/index.md)
