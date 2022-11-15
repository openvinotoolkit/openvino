# OpenVINO™ Paddle Frontend

OpenVINO Paddle Frontend is one of the OpenVINO Frontend libraries that is created for the Baidu PaddlePaddle™ framework.
The component is responsible for:
 * Paddle Reader - reads PaddlePaddle protobuf model and parses it to the frontend InputModel [paddle frontend architecture](./docs/paddle_frontend_architecture.md).
 * Paddle Converter - decodes the PaddlePaddle model and operators and maps them semantically to the OpenVINO opset [operator mapping flow](./docs/operation_mapping_flow.md).

OpenVINO Paddle Frontend uses [the common coding style rules](../../docs/dev/coding_style.md).

## Key contacts

People from the [openvino-ie-paddle-maintainers](https://github.com/orgs/openvinotoolkit/teams/openvino-ie-paddle-maintainers) have the rights to approve and merge PRs to the Paddle frontend component. They can assist with any questions about the component.

## Components

OpenVINO Paddle Frontend has the next structure:
 * [docs](./docs) contains developer documentation pages for the component.
 * [include](./include) contains module API and detailed information about the provided API.
 * [src](./src) folder contains sources of the component.
 * [tests](./tests) contains tests for the component [here](docs/tests.md).

## Debug capabilities

Developers can use OpenVINO Model debug capabilities that are described in the [OpenVINO Model User Guide](https://docs.openvino.ai/nightly/openvino_docs_OV_UG_Model_Representation.html#model-debug-capabilities).

## Tutorials
  TODO

## See also
 * [OpenVINO™ README](../../README.md)
 * [OpenVINO Core Components](../README.md)
 * [Developer documentation](../../../docs/dev/index.md)
