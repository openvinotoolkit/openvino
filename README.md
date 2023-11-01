<div align="center">
<img src="docs/img/openvino-logo-purple-black.png" width="400px">

[![PyPI Status](https://badge.fury.io/py/openvino.svg)](https://badge.fury.io/py/openvino)
[![Anaconda Status](https://anaconda.org/conda-forge/openvino/badges/version.svg)](https://anaconda.org/conda-forge/openvino)
[![brew Status](https://img.shields.io/homebrew/v/openvino)](https://formulae.brew.sh/formula/openvino)

[![PyPI Downloads](https://static.pepy.tech/badge/openvino)](https://pepy.tech/project/openvino)
[![Anaconda Downloads](https://anaconda.org/conda-forge/libopenvino/badges/downloads.svg)](https://anaconda.org/conda-forge/openvino/files)
[![brew Downloads](https://img.shields.io/homebrew/installs/dy/openvino)](https://formulae.brew.sh/formula/openvino)
 </div>

## Contents:
 - [What is OpenVINO?](#what-is-openvino-toolkit)
 - [Getting Started](#getting-started)
    - [Tutorials](#tutorials)
 - [Installation](#installation)
 - [Products Powered by OpenVINO](#products-povered-by-openvino)
 - [Documentation](#documentation)
 - [Supported Hardware](#system-requirements)
 - [Building OpenVINO](#building-openvino)
 - [Contributing](#contributing)
 - [Getting Support](#getting-support)
 - [License](#license)
 - [Telemetry](#telemetry)

## What is OpenVINO toolkit?

OpenVINO™ is an open-source toolkit for simple and efficient deployment of various deep-learning models.

- Boost deep learning performance in computer vision, automatic speech recognition, natural language processing, and other common tasks.
- Use models trained with popular frameworks such as TensorFlow, Pytorch, ONNX, Keras, and PaddlePaddle.
- Reduce resource demands and efficiently deploy on a range of Intel® platforms from edge to cloud.

OpenVINO includes several components: [OpenVINO Model Converter (OVC)], [OpenVINO™ Runtime], as well as CPU, GPU, NPU, auto batch and heterogeneous plugins to accelerate deep learning inference on Intel® CPUs and Intel® Processor Graphics.

<details>
  <summary>List of Components</summary>

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

</details>


### Getting Started  

#### Documentation
Read detailed information on OpenVINO's features, components, and usage in the [OpenVINO Documentation](https://docs.openvino.ai).

#### Tutorials
Explore a variety of tutorials in the [OpenVINO Notebooks Repository](https://github.com/openvinotoolkit/openvino_notebooks)📚.

Check out these notebooks that show how to optimize and deploy popular models (no installation required):
- [Create an LLM-powered Chatbot using OpenVINO](https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/254-llm-chatbot/254-llm-chatbot.ipynb)
- [YOLOv8 Optimization](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/230-yolov8-optimization)
- [Text-to-Image Generation](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/235-controlnet-stable-diffusion)
- [Quick Start Inference Example](https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/201-vision-monodepth)

### Installation

Get your OpenVINO installation command with just a few clicks. [Go to the installation page](https://docs.openvino.ai/2023.1/openvino_docs_install_guides_overview.html).

### Products Powered by OpenVINO

#### OpenVINO Ecosystem

-	[🤗Optimum Intel](https://github.com/huggingface/optimum-intel) -  a simple interface to optimize your Transformers and Diffusers models.
-   [Neural Network Compression Framework (NNCF)](https://github.com/openvinotoolkit/nncf) - a suite of advanced algorithms for model inference optimization including quantization, filter pruning, binarization and sparsity.
-   [OpenVINO™ Model Server (OVMS)](https://github.com/openvinotoolkit/model_server) - a scalable, high-performance solution for serving models optimized for Intel architectures.
-   [Software Development Kit (SDK) for the Intel® Geti™ platform for Computer Vision AI model training](https://github.com/openvinotoolkit/geti-sdk) - an online, interactive video and image annotation tool for computer vision use cases.
-	[RapidOCR](https://github.com/RapidAI/RapidOCR) - a cross platform OCR Library based on PaddleOCR & OnnxRuntime & OpenVINO.

#### External OpenVINO-Powered Products

-	[GIMP AI plugins](https://github.com/intel/openvino-ai-plugins-gimp) - GIMP AI plugins with OpenVINO Backend.
-	[Frigate](https://github.com/blakeblackshear/frigate) – NVR with real-time local object detection for IP cameras.
-	[whisper.cpp](https://github.com/ggerganov/whisper.cpp) - high-performance inference of OpenAI's Whisper automatic speech recognition (ASR) model.
-	[langchain](https://github.com/langchain-ai/langchain) -  a framework designed to simplify the creation of applications using large language models.
-	[stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) - a browser interface based on Gradio library for Stable Diffusion.

<details>
  <summary>More products</summary>

   **Ecosystem** 

  -   [OpenVINO™ Training Extensions (OTE)](https://github.com/openvinotoolkit/training_extensions) - an environment to train models and convert them using OpenVINO for optimized inference.
  -   [Dataset Management Framework (Datumaro)](https://github.com/openvinotoolkit/datumaro) - a framework and CLI tool to build, transform, and analyze datasets.
  -	  [openvino-rs](https://github.com/intel/openvino-rs) - bindings for accessing OpenVINO functionality in Rust.
  -	  [OpenVINO.NET](https://github.com/sdcb/OpenVINO.NET) - a high-quality .NET wrapper for OpenVINO™ toolkit.
  -   [openvino_contrib](https://github.com/openvinotoolkit/openvino_contrib) - a repository for the development of additional OpenVINO modules 

   **External** 
   -   [OpenCV](https://opencv.org/) - a library of programming functions mainly for real-time computer vision. 
   -   [ONNX Runtime](https://onnxruntime.ai/) - a cross-platform inference and training machine-learning accelerator. 
   -   [TNN](https://github.com/Tencent/TNN/tree/master) - a high-performance, lightweight neural network inference framework.

</details>

### Documentation

#### User documentation

The latest documentation for OpenVINO™ Toolkit is available [here](https://docs.openvino.ai/). This documentation contains detailed information about all OpenVINO components, providing you with essential details to create applications using binary OpenVINO distribution or your own  OpenVINO version without requiring source code modifications.

For additional resources and information, you can explore:

* [Product Page](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html)
* [Release Notes](https://www.intel.com/content/www/us/en/developer/articles/release-notes/openvino/2023-1.html)
* [Success Stories](https://www.intel.com/content/www/us/en/internet-of-things/ai-in-production/success-stories.html)
* [OpenVINO Blog](https://blog.openvino.ai/)
* [OpenVINO™ toolkit on Medium](https://medium.com/@openvino)

#### Developer documentation

[Developer documentation](./docs/dev/index.md) contains information about architectural decisions implemented within OpenVINO components. It offers all the essential information required for contributing to OpenVINO.

### Supported Hardware

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
            <td> <a href="https://docs.openvino.ai/2023.1/openvino_docs_OV_UG_supported_plugins_CPU.html#doxid-openvino-docs-o-v-u-g-supported-plugins-c-p-u">Intel CPU</a></tb>
            <td><b><i><a href="./src/plugins/intel_cpu">openvino_intel_cpu_plugin</a></i></b></td>
            <td>Intel® Xeon® with Intel® Advanced Vector Extensions 2 (Intel® AVX2), Intel® Advanced Vector Extensions 512 (Intel® AVX-512), Intel® Advanced Matrix Extensions (Intel® AMX), Intel® Core™ Processors with Intel® AVX2, Intel® Atom® Processors with Intel® Streaming SIMD Extensions (Intel® SSE)</td>
        </tr>
        <tr>
            <td> <a href="https://docs.openvino.ai/2023.1/openvino_docs_OV_UG_supported_plugins_CPU.html#doxid-openvino-docs-o-v-u-g-supported-plugins-c-p-u">ARM CPU</a></tb>
            <td><b><i><a href="./src/plugins/intel_cpu">openvino_arm_cpu_plugin</a></i></b></td>
            <td>Raspberry Pi™ 4 Model B, Apple® Mac mini with Apple silicon
        </tr>
        <tr>
            <td>GPU</td>
            <td><a href="https://docs.openvino.ai/2023.1/openvino_docs_OV_UG_supported_plugins_GPU.html#doxid-openvino-docs-o-v-u-g-supported-plugins-g-p-u">Intel GPU</a></td>
            <td><b><i><a href="./src/plugins/intel_gpu">openvino_intel_gpu_plugin</a></i></b></td>
            <td>Intel® Processor Graphics including Intel® HD Graphics and Intel® Iris® Graphics, Intel® Arc™ A-Series Graphics, Intel® Data Center GPU Flex Series, Intel® Data Center GPU Max Series</td>
        </tr>
        <tr>
            <td>GNA</td>
            <td><a href="https://docs.openvino.ai/2023.1/openvino_docs_OV_UG_supported_plugins_GNA.html#doxid-openvino-docs-o-v-u-g-supported-plugins-g-n-a">Intel GNA</a></td>
            <td><b><i><a href="./src/plugins/intel_gna">openvino_intel_gna_plugin</a></i></b></td>
            <td>Intel® Speech Enabling Developer Kit, Amazon Alexa* Premium Far-Field Developer Kit, Intel® Pentium® Silver J5005 Processor, Intel® Pentium® Silver N5000 Processor, Intel® Celeron® J4005 Processor, Intel® Celeron® J4105 Processor, Intel® Celeron® Processor N4100, Intel® Celeron® Processor N4000, Intel® Core™ i3-8121U Processor, Intel® Core™ i7-1065G7 Processor, Intel® Core™ i7-1060G7 Processor, Intel® Core™ i5-1035G4 Processor, Intel® Core™ i5-1035G7 Processor, Intel® Core™ i5-1035G1 Processor, Intel® Core™ i5-1030G7 Processor, Intel® Core™ i5-1030G4 Processor, Intel® Core™ i3-1005G1 Processor, Intel® Core™ i3-1000G1 Processor, Intel® Core™ i3-1000G4 Processor</td>
        </tr>
        <tr>
            <td></td>
            <td><a href="https://docs.openvino.ai/2023.1/openvino_docs_OV_UG_Automatic_Batching.html">Auto Batch</a></td>
            <td><b><i><a href="./src/plugins/auto_batch">openvino_auto_batch_plugin</a></i></b></td>
            <td>Auto batch plugin performs on-the-fly automatic batching (i.e. grouping inference requests together) to improve device utilization, with no programming effort from the user</td>
        </tr>
        <tr>
            <td></td>
            <td><a href="https://docs.openvino.ai/2023.1/openvino_docs_OV_UG_Hetero_execution.html#doxid-openvino-docs-o-v-u-g-hetero-execution">Hetero</a></td>
            <td><b><i><a href="./src/plugins/hetero">openvino_hetero_plugin</a></i></b></td>
            <td>Heterogeneous execution enables automatic inference splitting between several devices</td>
        </tr>
        <tr>
            <td>NPU</td>
            <td><a href="https://docs.openvino.ai/2023.1/openvino_docs_OV_UG_Working_with_devices.html">Intel NPU</a></td>
            <td><b><i></i></b></td>
            <td></td>
        </tr>
    </tbody>
</table>

Check [System Requirements](https://docs.openvino.ai/2023.1/system_requirements.html) and [compatibility details](https://docs.openvino.ai/2023.1/compatibility_and_support.html) for more information.


### Building OpenVINO

See [How to build OpenVINO](./docs/dev/build.md) to get more information about the OpenVINO build process.

### Contributing

Explore the list of [Good First Issues](https://github.com/openvinotoolkit/openvino/issues/17502), if you're looking for a place to start contributing.

For detailed guidelines, see [CONTRIBUTING](./CONTRIBUTING.md). Your contributions are greatly appreciated!

If you'd like to be assigned to an issue, simply leave a comment with the `.take` command in the selected issue.


### Getting Support

To report questions, issues, and suggestions, use:

* [GitHub* Issues](https://github.com/openvinotoolkit/openvino/issues)
* The [`openvino`](https://stackoverflow.com/questions/tagged/openvino) tag on StackOverflow\*


### License
OpenVINO™ Toolkit is licensed under [Apache License Version 2.0](LICENSE).
By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

### Telemetry
OpenVINO™ collects software performance and usage data to improve OpenVINO™ tools. This data is collected directly by OpenVINO™ or through the use of Google Analytics 4.
You can opt-out at any time by running the command:

``` bash
opt_in_out --opt_out
```
More Information is available at https://docs.openvino.ai/nightly/openvino_docs_telemetry_information.html.

---
\* Other names and brands may be claimed as the property of others.

[OpenVINO™ Runtime]:https://docs.openvino.ai/2023.1/openvino_docs_OV_UG_OV_Runtime_User_Guide.html
[OpenVINO Model Converter (OVC)]:https://docs.openvino.ai/2023.1/openvino_docs_model_processing_introduction.html#convert-a-model-in-cli-ovc
[Samples]:https://github.com/openvinotoolkit/openvino/tree/master/samples
