# OpenVINO™ Padddle Frontend

OpenVINO Paddle Frontend is one of the OpenVINO Frontend libaries that is dedicated for Baidu PaddlePaddle™ framework. 
The component is responsible for:
 * Paddle Reader - component reads PaddlePaddle protobuf model, and parses it to frontend InputModel [paddle frontend architecture](./docs/paddle_frontend_architecture.md).
 * Paddle Converter - component decodes the PaddlePaddle model and operators, and map them sematically to OpenVINO opset [operator mapping flow](./docs/operation_mapping_flow.md).

The diagram below shows the positions of Paddle Frontend and the main components.![](./docs/img/PaddleFrontendPositioning.PNG)

OpenVINO Paddle Frontend uses [the common coding style rules](../../docs/dev/coding_style.md).

## Key person

People from the [openvino-ie-paddle-maintainers](https://github.com/orgs/openvinotoolkit/teams/openvino-ie-paddle-maintainers) allows to approve and merge PRs to the Paddle frontend component. These guys can help in case of any questions about the component.

## Components

OpenVINO Paddle Frontend has the next structure:
 * [docs](./docs) contains developer documentation pages for the component.
 * [include](./include) contains module API. Detailed information about provided API can be found.
 * [src](./src) folder contains sources of the component.
 * [tests](./tests) contains tests for the component [here](docs/tests.md).

## Debug capabilities

Developers can use OpenVINO Model debug capabilities that are described in the [OpenVINO Model User Guide](https://docs.openvino.ai/nightly/openvino_docs_OV_UG_Model_Representation.html#model-debug-capabilities).

## Tutorials
  TODO

## See also
 * TODO
