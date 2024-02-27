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
 - [Getting Started](#getting-started)
 - [Installation](#installation)
 - [Tutorials](#tutorials)
 - [Products Powered by OpenVINO](#products-povered-by-openvino)
 - [Documentation](#documentation)
 - [Contribution and Support](#contribution-and-support)
 - [License](#license)


Welcome to OpenVINO™, an open-source software toolkit for optimizing and deploying deep learning models.

- **Inference Optimization**: Boost deep learning performance in computer vision, automatic speech recognition, natural language processing, and many other common tasks.
- **Flexible Model Support**: Use models trained with popular frameworks such as TensorFlow, Pytorch, ONNX, Keras, and PaddlePaddle.
- **Broad Platform Compatibility**: Reduce resource demands and efficiently deploy on a range of platforms from edge to cloud.
- **Community and Ecosystem**: Join an active community contributing to the enhancement of deep learning performance across various domains.


<details>
  <summary>List of Distributed OpenVINO Components</summary>

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

[OpenVINO Quickstart example](https://docs.openvino.ai/2024/notebooks/201-vision-monodepth-with-output.html) will walk you through the basics to deploy your first model.

### Installation

Get your OpenVINO installation command with just a few clicks. [Go to the installation page](https://docs.openvino.ai/2024/openvino_docs_install_guides_overview.html).

Check [system requirements](https://docs.openvino.ai/2024/system_requirements.html) and [supported devices](https://docs.openvino.ai/2024/openvino_docs_OV_UG_supported_plugins_Supported_Devices.html) for detailed information.

### Tutorials

Explore a variety of tutorials in the [OpenVINO Notebooks Repository](https://github.com/openvinotoolkit/openvino_notebooks)📚.

Check out these notebooks that show how to optimize and deploy popular models (no installation required):
- [Create an LLM-powered Chatbot using OpenVINO](https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/254-llm-chatbot/254-llm-chatbot.ipynb)
- [YOLOv8 Optimization](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/230-yolov8-optimization)
- [Text-to-Image Generation](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/235-controlnet-stable-diffusion)

### Products Powered by OpenVINO

#### OpenVINO Ecosystem

-	 [🤗Optimum Intel](https://github.com/huggingface/optimum-intel) -  a simple interface to optimize Transformers and Diffusers models.
-   [Neural Network Compression Framework (NNCF)](https://github.com/openvinotoolkit/nncf) - advanced model optimization techniques including quantization, filter pruning, binarization and sparsity.
-   [GenAI Repository](https://github.com/openvinotoolkit/openvino.genai) and [OpenVINO Tokenizers](https://github.com/openvinotoolkit/openvino_tokenizers) - resources and tools for developing and optimizing Generative AI applications.
-   [OpenVINO™ Model Server (OVMS)](https://github.com/openvinotoolkit/model_server) - a scalable, high-performance solution for serving models optimized for Intel architectures.
-   [Software Development Kit (SDK) for the Intel® Geti™](https://github.com/openvinotoolkit/geti-sdk) - an interactive video and image annotation tool for computer vision use cases.

#### External OpenVINO-Powered Products

[GIMP AI plugins](https://github.com/intel/openvino-ai-plugins-gimp), [Frigate](https://github.com/blakeblackshear/frigate), [whisper.cpp](https://github.com/ggerganov/whisper.cpp) and many other products showing OpenVINO's flexibility across different applications.

Check out [Awesome OpenVINO](https://github.com/openvinotoolkit/awesome-openvino) repository to see all the community-made projects using OpenVINO!

<details>
  <summary>More products ### this will be removed when we decide on the products to highlight </summary>

   **Ecosystem**

  -   [OpenVINO™ Training Extensions (OTE)](https://github.com/openvinotoolkit/training_extensions) - an environment to train models and convert them using OpenVINO for optimized inference.
  -   [Dataset Management Framework (Datumaro)](https://github.com/openvinotoolkit/datumaro) - a framework and CLI tool to build, transform, and analyze datasets.
  - [RapidOCR](https://github.com/RapidAI/RapidOCR) - a cross platform OCR Library based on PaddleOCR & OnnxRuntime & OpenVINO.
  -	  [openvino-rs](https://github.com/intel/openvino-rs) - bindings for accessing OpenVINO functionality in Rust.
  -	  [OpenVINO.NET](https://github.com/sdcb/OpenVINO.NET) - a high-quality .NET wrapper for OpenVINO™ toolkit.
  -   [openvino_contrib](https://github.com/openvinotoolkit/openvino_contrib) - a repository for the development of additional OpenVINO modules

   **External**
   -   [OpenCV](https://opencv.org/) - a library of programming functions mainly for real-time computer vision.
   -   [ONNX Runtime](https://onnxruntime.ai/) - a cross-platform inference and training machine-learning accelerator.
   -   [TNN](https://github.com/Tencent/TNN/tree/master) - a high-performance, lightweight neural network inference framework.
   -	[langchain](https://github.com/langchain-ai/langchain) -  a framework designed to simplify the creation of applications using large language models.
   -	[stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) - a browser interface based on Gradio library for Stable Diffusion.

</details>

### Documentation

**User documentation**

Find latest documentation, from getting started guides to advanced topics, at [docs.openvino.ai](https://docs.openvino.ai/). User documentation contains detailed information about all OpenVINO components, providing you with essential details to create applications using OpenVINO.

**Developer documentation**

[Developer documentation](./docs/dev/index.md) on GitHub contains information about architectural decisions implemented within OpenVINO components. It offers all the essential information required for [building](./docs/dev/build.md)  and [contributing](./CONTRIBUTING.md) to OpenVINO.

For additional resources, read:

* [Product Page](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html)
* [Release Notes](https://docs.openvino.ai/2024/openvino_release_notes.html)
* [OpenVINO Blog](https://blog.openvino.ai/) and [OpenVINO™ toolkit on Medium](https://medium.com/@openvino)

### Contribution and Support

We welcome contributions of all kinds!

Explore the list of [Good First Issues](https://github.com/openvinotoolkit/openvino/issues/17502), if you're looking for a place to start contributing.
To be assigned to an issue, simply leave a comment with the `.take` command in the selected issue.
Check out [Contribution Guidelines](./CONTRIBUTING.md) for more details.

Ask questions and seek support on:

* [GitHub Issues](https://github.com/openvinotoolkit/openvino/issues)
* OpenVINO channels on the [Intel DevHub Discord server](https://discord.gg/7pVRxUwdWG)
* the [`openvino`](https://stackoverflow.com/questions/tagged/openvino) tag on Stack Overflow\*.

### License

OpenVINO™ Toolkit is licensed under [Apache License Version 2.0](LICENSE).
By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

**Telemetry**

To help us improve, OpenVINO™ collects software performance and usage data to improve OpenVINO™ tools. This data is collected directly by OpenVINO™ or through the use of Google Analytics 4. Visit our [telemetry page](https://docs.openvino.ai/nightly/openvino_docs_telemetry_information.html) to learn more.
You can opt-out at any time by running the command:

``` bash
opt_in_out --opt_out
```

---
\* Other names and brands may be claimed as the property of others.


[OpenVINO™ Runtime]:https://docs.openvino.ai/2023.1/openvino_docs_OV_UG_OV_Runtime_User_Guide.html
[OpenVINO Model Converter (OVC)]:https://docs.openvino.ai/2023.1/openvino_docs_model_processing_introduction.html#convert-a-model-in-cli-ovc
[Samples]:https://github.com/openvinotoolkit/openvino/tree/master/samples
