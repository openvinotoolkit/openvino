# OpenVINO™ Paddle Frontend

OpenVINO Paddle Frontend is one of the OpenVINO Frontend libraries created for the Baidu PaddlePaddle™ framework.
The component is responsible for:
 * Paddle Reader - reads a PaddlePaddle protobuf model and parses it to the frontend InputModel. [Learn more about Paddle Frontend architecture.](./docs/paddle_frontend_architecture.md).
 * Paddle Converter - decodes a PaddlePaddle model and operators and maps them semantically to the OpenVINO opset. [Learn more about the operator mapping flow.](./docs/operation_mapping_flow.md).

OpenVINO Paddle Frontend uses [the common coding style rules](../../../docs/dev/coding_style.md).

## Key contacts

People from the [openvino-ie-paddle-maintainers](https://github.com/orgs/openvinotoolkit/teams/openvino-ie-paddle-maintainers) have the rights to approve and merge PRs to the Paddle frontend component. They can assist with any questions about the component.

## Components

OpenVINO Paddle Frontend has the following structure:
 * [docs](./docs) contains developer documentation for the component.
 * [include](./include) contains module API and detailed information about the provided API.
 * [src](./src) folder contains sources of the component.
 * [tests](./tests) contains tests for the component. To get more information, read [How to run and add tests](./docs/tests.md) page.

## Debug capabilities

Developers can use OpenVINO Model debug capabilities that are described in the [OpenVINO Model User Guide](https://docs.openvino.ai/2025/openvino-workflow/running-inference/model-representation.html#model-debug-capabilities).

## Tutorials

 * [How to support a new operator](./docs/operation_mapping_flow.md)
 * [How to run and add tests](./docs/tests.md)

## See also
 * [OpenVINO™ README](../../README.md)
 * [OpenVINO Core Components](../README.md)
 * [Developer Documentation](../../../docs/dev/index.md)
