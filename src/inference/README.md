# OpenVINO™ Inference

OpenVINO Inference is a part of OpenVINO Runtime library. 
The component is responsible for model inference on hardware device, provides API for OpenVINO Plugin development.

OpenVINO Inference uses [the common coding style rules](../../docs/dev/coding_style.md).

## Key person

People from the [openvino-ie-maintainers](https://github.com/orgs/openvinotoolkit/teams/openvino-ie-maintainers) allows to approve and merge PRs to the inference component. These guys can help in case of any questions about the component.

## Components

OpenVINO Inference has the next structure:
 * [dev_api](./dev_api) contains developer API which is needed to develop OpenVINO Plugins. In order to use this API, you need to link your component against `openvino::runtime::dev`.
 * [include](./include) contains public API. Detailed information about provided API can be found [here](./docs/api_details.md).
 * [src](./src) folder contains sources of the component.

OpenVINO Inference has unit and functional tests. Unit tests are located in [src/tests/unit/inference_engine](../tests/unit/inference_engine/), functional tests locates [src/tests/functional/inference_engine](../tests/functional/inference_engine/).

## See also
 * [OpenVINO™ README](../../README.md)
 * [OpenVINO Core Components](../README.md)
 * [Developer documentation](../../docs/dev/index.md)

