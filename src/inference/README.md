# OpenVINO™ Inference

OpenVINO Inference is a part of the OpenVINO Runtime library. 
The component is responsible for model inference on hardware devices and provides API for OpenVINO Plugin development.

OpenVINO Inference uses [the common coding style rules](../../docs/dev/coding_style.md).

## Key contacts

People from the [openvino-ie-maintainers](https://github.com/orgs/openvinotoolkit/teams/openvino-ie-maintainers) group have the rights to approve and merge PRs to the inference component. They can assist with any questions about the component.

## Components

OpenVINO Inference has the following structure:
 * [dev_api](./dev_api) contains developer API required to develop OpenVINO Plugins. To use this API, link your component against `openvino::runtime::dev`.
 * [include](./include) contains public API. Find more information in the [OpenVINO Inference API](./docs/api_details.md) document.
 * [src](./src) contains sources of the component.

OpenVINO Inference has unit and functional tests. Unit tests are located in [src/inference/tests/unit](tests/unit/), functional tests are located in [src/inference/tests/functional](tests/unit/).

## See also

 * [OpenVINO README](../../README.md)
 * [OpenVINO Core Components](../README.md)
 * [Developer Documentation](../../docs/dev/index.md)

