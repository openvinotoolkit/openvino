# Roadmap to OpenVINO contribution

This document lists all the necessary steps required to setup your environment, build OpenVINO locally and run tests for specific components. It's a perfect place to start when you have just picked a Good First Issue and are wondering where to start working on it.

Keep in mind that we are here to help - **do not hesitate to ask the development team if something is not clear**. Such questions allow up to keep improving our documentation.

## 1. Prerequisites 

You can start with the following links:
- [What is OpenVINO?](https://github.com/openvinotoolkit/openvino#what-is-openvino-toolkit)
- [OpenVINO architecture](https://github.com/openvinotoolkit/openvino/blob/master/src/docs/architecture.md)
- [Contribution guide](https://github.com/openvinotoolkit/openvino/blob/master/CONTRIBUTING.md)
- [User documentation](https://docs.openvino.ai/)
- [Pick up a Good First Issue](https://github.com/orgs/openvinotoolkit/projects/3)

## 2. Building the project

In order to build the project, please follow the [build instructions for your specific OS](https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build.md).

## 3. Familiarize yourself with component you'll be working with

Choose the component your Good First Issue is related to. You can run tests to make sure it is working correctly.

#### APIs
- [C API](https://github.com/openvinotoolkit/openvino/tree/master/src/bindings/c)
- [Core](https://github.com/openvinotoolkit/openvino/tree/master/src/core)
- [Python API](https://github.com/openvinotoolkit/openvino/tree/master/src/bindings/python)

#### Frontends
- [IR Frontend](https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/ir)
- [ONNX Frontend](https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/onnx)
- [PaddlePaddle Frontend](https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/paddle)
- [PyTorch Frontend](https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/pytorch)
- [TensorFlow Frontend](https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/tensorflow)

#### Plugins
- [Auto plugin](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/auto)
- [CPU plugin](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_cpu)
- [GPU plugin](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu)
- [Hetero plugin](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/hetero)
- [Template plugin](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/README.md)

#### Tools
- [Benchmark Tool](https://github.com/openvinotoolkit/openvino/tree/master/tools/benchmark_tool)
- [Model Optimizer](https://github.com/openvinotoolkit/openvino/tree/master/tools/mo)

#### Others
- [Documentation](https://github.com/openvinotoolkit/openvino/blob/master/CONTRIBUTING_DOCS.md)

## 3. Start working on your Good First Issue

Use the issue description and locally built OpenVINO to complete the task. Remember that you can always ask users tagged in the "Contact points" section for help!

## 4. Submit a PR with your changes

Follow our [Good Pull Request guidelines](https://github.com/openvinotoolkit/openvino/blob/master/CONTRIBUTING_PR.md).

## 5. Wait for review

We'll make sure to review your Pull Request as soon as possible and provide you with our feedback. You can expect a merge once your changes are validated with automatic tests and approved by maintainers.
