<div align="center">
<img src="docs/sphinx_setup/_static/images/img/openvino-logo-purple-black.png" width="400px">

[![PyPI Status](https://badge.fury.io/py/openvino.svg)](https://badge.fury.io/py/openvino)
[![Anaconda Status](https://anaconda.org/conda-forge/openvino/badges/version.svg)](https://anaconda.org/conda-forge/openvino)
[![brew Status](https://img.shields.io/homebrew/v/openvino)](https://formulae.brew.sh/formula/openvino)

[![PyPI Downloads](https://static.pepy.tech/badge/openvino)](https://pepy.tech/project/openvino)
[![Anaconda Downloads](https://anaconda.org/conda-forge/libopenvino/badges/downloads.svg)](https://anaconda.org/conda-forge/openvino/files)
[![brew Downloads](https://img.shields.io/homebrew/installs/dy/openvino)](https://formulae.brew.sh/formula/openvino)
 </div>

## Contents:

 - [What is OpenVINO?](#what-is-openvino-toolkit)
    - [Components](#components)
 - [Supported Hardware matrix](#supported-hardware-matrix)
 - [License](#license)
 - [Documentation](#documentation)
 - [Tutorials](#tutorials)
 - [Products which use OpenVINO](#products-which-use-openvino)
 - [System requirements](#system-requirements)
 - [How to build](#how-to-build)
 - [How to contribute](#how-to-contribute)
 - [Get a support](#get-a-support)
 - [See also](#see-also)

## What is OpenVINO toolkit?

OpenVINO™ is an open-source toolkit for optimizing and deploying AI inference.
 - Boost deep learning performance in computer vision, automatic speech recognition, natural language processing and other common tasks
 - Use models trained with popular frameworks like TensorFlow, PyTorch and more
 - Reduce resource demands and efficiently deploy on a range of Intel® platforms from edge to cloud


This open-source version includes several components: namely [OpenVINO Model Converter (OVC)], [OpenVINO™ Runtime], as well as CPU, GPU, NPU, multi device and heterogeneous plugins to accelerate deep learning inference on Intel® CPUs and Intel® Processor Graphics.
It supports pre-trained models from [Open Model Zoo], along with 100+ open
source and public models in popular formats such as TensorFlow, ONNX, PaddlePaddle, MXNet, Caffe, Kaldi.

### Components
* [OpenVINO™ Runtime] - is a set of C++ libraries with C and Python bindings providing a common API to deliver inference solutions on the platform of your choice.
    * [core](./src/core) - provides the base API for model representation and modification.
    * [inference](./src/inference) - provides an API to infer models on the device.
    * [transformations](./src/common/transformations) - contains the set of common transformations which are used in OpenVINO plugins.
    * [low precision transformations](./src/common/low_precision_transformations) - contains the set of transformations that are used in low precision models
    * [bindings](./src/bindings) - contains all available OpenVINO bindings which are maintained by the OpenVINO team.
        * [c](./src/bindings/c) - C API for OpenVINO™ Runtime
        * [python](./src/bindings/python) - Python API for OpenVINO™ Runtime
* [Plugins](./src/plugins) - contains OpenVINO plugins which are maintained in open-source by the OpenVINO team. For more information, take a look at the [list of supported devices](#supported-hardware-matrix).
* [Frontends](./src/frontends) - contains available OpenVINO frontends that allow reading models from the native framework format.
* [OpenVINO Model Converter (OVC)] - is a cross-platform command-line tool that facilitates the transition between training and deployment environments, and adjusts deep learning models for optimal execution on end-point target devices.
* [Samples] - applications in C, C++ and Python languages that show basic OpenVINO use cases.

## Supported Hardware matrix

The OpenVINO™ Runtime can infer models on different hardware devices. This section provides the list of supported devices.

<table>
    <thead>
        <tr>
            <th>Device</th>
            <th>Plugin</th>
            <th>Library</th>
            <th>Short Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=2>CPU</td>
            <td> <a href="https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html">Intel CPU</a></tb>
            <td><b><i><a href="./src/plugins/intel_cpu">openvino_intel_cpu_plugin</a></i></b></td>
            <td>Intel Xeon with Intel® Advanced Vector Extensions 2 (Intel® AVX2), Intel® Advanced Vector Extensions 512 (Intel® AVX-512), and AVX512_BF16, Intel Core Processors with Intel AVX2, Intel Atom Processors with Intel® Streaming SIMD Extensions (Intel® SSE), Intel® Advanced Matrix Extensions (Intel® AMX)</td>
        </tr>
        <tr>
            <td> <a href="https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html">ARM CPU</a></tb>
            <td><b><i><a href="./src/plugins/intel_cpu">openvino_arm_cpu_plugin</a></i></b></td>
            <td>Raspberry Pi™ 4 Model B, Apple® Mac mini with Apple silicon
        </tr>
        <tr>
            <td>GPU</td>
            <td><a href="https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html">Intel GPU</a></td>
            <td><b><i><a href="./src/plugins/intel_gpu">openvino_intel_gpu_plugin</a></i></b></td>
            <td>Intel Processor Graphics, including Intel HD Graphics and Intel Iris Graphics</td>
        </tr>
        <tr>
            <td>NPU</td>
            <td><a href="https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html">Intel NPU</a></td>
            <td><b><i><a href="./src/plugins/intel_npu">openvino_intel_npu_plugin</a></i></b></td>
            <td>Intel® Core™ Ultra Processors</td>
        </tr>
    </tbody>
</table>

OpenVINO™ Toolkit also contains several plugins which simplify loading models on several hardware devices:
<table>
    <thead>
        <tr>
            <th>Plugin</th>
            <th>Library</th>
            <th>Short Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><a href="https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/auto-device-selection.html">Auto</a></td>
            <td><b><i><a href="./src/plugins/auto">openvino_auto_plugin</a></i></b></td>
            <td>Auto plugin enables selecting Intel device for inference automatically</td>
        </tr>
        <tr>
            <td><a href="https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/automatic-batching.html">Auto Batch</a></td>
            <td><b><i><a href="./src/plugins/auto_batch">openvino_auto_batch_plugin</a></i></b></td>
            <td>Auto batch plugin performs on-the-fly automatic batching (i.e. grouping inference requests together) to improve device utilization, with no programming effort from the user</td>
        </tr>
        <tr>
            <td><a href="https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/hetero-execution.html">Hetero</a></td>
            <td><b><i><a href="./src/plugins/hetero">openvino_hetero_plugin</a></i></b></td>
            <td>Heterogeneous execution enables automatic inference splitting between several devices</td>
        </tr>
        <tr>
            <td><a href="https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/multi-device.html">Multi</a></td>
            <td><b><i><a href="./src/plugins/auto">openvino_auto_plugin</a></i></b></td>
            <td>Multi plugin enables simultaneous inference of the same model on several devices in parallel</td>
        </tr>
    </tbody>
</table>

## License
OpenVINO™ Toolkit is licensed under [Apache License Version 2.0](LICENSE).
By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

## Telemetry
OpenVINO™ collects software performance and usage data for the purpose of improving OpenVINO™ tools. This data is collected directly by OpenVINO™ or through the use of Google Analytics 4.
You can opt-out at any time by running the command:

``` bash
opt_in_out --opt_out
```

More Information is available at https://docs.openvino.ai/latest/openvino_docs_telemetry_information.html.

## Documentation

### User documentation

The latest documentation for OpenVINO™ Toolkit is available [here](https://docs.openvino.ai/). This documentation contains detailed information about all OpenVINO components and provides all the important information you may need to create an application based on binary OpenVINO distribution or own OpenVINO version without source code modification.

### Developer documentation

[Developer documentation](./docs/dev/index.md) contains information about architectural decisions which are applied inside the OpenVINO components. This documentation has all necessary information which could be needed in order to contribute to OpenVINO.

## Tutorials

The list of OpenVINO tutorials:

- [Jupyter notebooks](https://github.com/openvinotoolkit/openvino_notebooks)

## Products which use OpenVINO

- [OpenCV](https://opencv.org/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [OpenVINO™ Integration with TensorFlow](https://www.intel.com/content/www/us/en/developer/tools/devcloud/edge/build/ovtfoverview.html)
- [TNN](https://github.com/Tencent/TNN/tree/master)

You can also check out [Awesome OpenVINO](https://github.com/openvinotoolkit/awesome-openvino) to see all the community-made projects using OpenVINO!

## System requirements

The system requirements vary depending on platform and are available on dedicated pages:
- [Linux](https://docs.openvino.ai/2024/get-started/install-openvino/install-openvino-linux.html)
- [Windows](https://docs.openvino.ai/2024/get-started/install-openvino/install-openvino-windows.html)
- [macOS](https://docs.openvino.ai/2024/get-started/install-openvino/install-openvino-macos.html)

## How to build

See [How to build OpenVINO](./docs/dev/build.md) to get more information about the OpenVINO build process.

## How to contribute

See [Contributions Welcome](https://github.com/openvinotoolkit/openvino/issues/17502) for good first issues.

See [CONTRIBUTING](./CONTRIBUTING.md) for contribution details. Thank you!

Visit [Intel DevHub Discord server](https://discord.gg/7pVRxUwdWG) if you need help or wish to talk to OpenVINO developers. You can go to the channel dedicated to Good First Issue support if you are working on a task.

## Take the issue
If you wish to be assigned to an issue please add a comment with `.take` command.

## Get support

Report questions, issues and suggestions, using:

* [GitHub* Issues](https://github.com/openvinotoolkit/openvino/issues)
* The [`openvino`](https://stackoverflow.com/questions/tagged/openvino) tag on StackOverflow\*
* [Forum](https://software.intel.com/en-us/forums/computer-vision)
* OpenVINO channels on the [Intel DevHub Discord server](https://discord.gg/7pVRxUwdWG)

## Additional Resources

* [OpenVINO Wiki](https://github.com/openvinotoolkit/openvino/wiki)
* [OpenVINO Storage](https://storage.openvinotoolkit.org/)
* Additional OpenVINO™ toolkit modules:
    * [openvino_contrib](https://github.com/openvinotoolkit/openvino_contrib)
* [Intel® Distribution of OpenVINO™ toolkit Product Page](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html)
* [Intel® Distribution of OpenVINO™ toolkit Release Notes](https://software.intel.com/en-us/articles/OpenVINO-RelNotes)
* [Neural Network Compression Framework (NNCF)](https://github.com/openvinotoolkit/nncf) - a suite of advanced algorithms for model inference optimization including quantization, filter pruning, binarization and sparsity
* [OpenVINO™ Training Extensions (OTE)](https://github.com/openvinotoolkit/training_extensions) - convenient environment to train Deep Learning models and convert them using OpenVINO for optimized inference.
* [OpenVINO™ Model Server (OVMS)](https://github.com/openvinotoolkit/model_server) - a scalable, high-performance solution for serving deep learning models optimized for Intel architectures
* [Computer Vision Annotation Tool (CVAT)](https://github.com/opencv/cvat) - an online, interactive video and image annotation tool for computer vision purposes.
* [Dataset Management Framework (Datumaro)](https://github.com/openvinotoolkit/datumaro) - a framework and CLI tool to build, transform, and analyze datasets.

---
\* Other names and brands may be claimed as the property of others.

[Open Model Zoo]:https://github.com/openvinotoolkit/open_model_zoo
[OpenVINO™ Runtime]:https://docs.openvino.ai/2024/openvino-workflow/running-inference.html
[OpenVINO Model Converter (OVC)]:https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html#convert-a-model-in-cli-ovc
[Samples]:https://github.com/openvinotoolkit/openvino/tree/master/samples
