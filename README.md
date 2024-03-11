<div align="center">
<img src="docs/sphinx_setup/_static/images/img/openvino-logo-purple-black.png" width="400px">

[![PyPI Status](https://badge.fury.io/py/openvino.svg)](https://badge.fury.io/py/openvino)
[![Anaconda Status](https://anaconda.org/conda-forge/openvino/badges/version.svg)](https://anaconda.org/conda-forge/openvino)
[![brew Status](https://img.shields.io/homebrew/v/openvino)](https://formulae.brew.sh/formula/openvino)

[![PyPI Downloads](https://static.pepy.tech/badge/openvino)](https://pepy.tech/project/openvino)
[![Anaconda Downloads](https://anaconda.org/conda-forge/libopenvino/badges/downloads.svg)](https://anaconda.org/conda-forge/openvino/files)
[![brew Downloads](https://img.shields.io/homebrew/installs/dy/openvino)](https://formulae.brew.sh/formula/openvino)
 </div>

Welcome to OpenVINO™, an open-source software toolkit for optimizing and deploying deep learning models.

- **Inference Optimization**: Boost deep learning performance in computer vision, automatic speech recognition, natural language processing, and many other common tasks.
- **Flexible Model Support**: Use models trained with popular frameworks such as TensorFlow, Pytorch, ONNX, Keras, and PaddlePaddle.
- **Broad Platform Compatibility**: Reduce resource demands and efficiently deploy on a range of platforms from edge to cloud.
- **Community and Ecosystem**: Join an active community contributing to the enhancement of deep learning performance across various domains.

### Installation

Get your OpenVINO installation command with just a few clicks. [Go to the installation page](https://docs.openvino.ai/2024/openvino_docs_install_guides_overview.html) to see all distribution channels or use this command for quick installation:

```
pip install openvino
```

Check [system requirements](https://docs.openvino.ai/2024/system_requirements.html) and [supported devices](https://docs.openvino.ai/2024/openvino_docs_OV_UG_supported_plugins_Supported_Devices.html) for detailed information.

### Getting Started

[OpenVINO Quickstart example](https://docs.openvino.ai/2024/notebooks/201-vision-monodepth-with-output.html) will walk you through the basics to deploy your first model.


### Tutorials

Explore a variety of tutorials in the [OpenVINO Notebooks Repository](https://github.com/openvinotoolkit/openvino_notebooks)📚.

Check out these notebooks that show how to optimize and deploy popular models (no installation required):
- [Create an LLM-powered Chatbot using OpenVINO](https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/254-llm-chatbot/254-llm-chatbot.ipynb)
- [YOLOv8 Optimization](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/230-yolov8-optimization)
- [Text-to-Image Generation](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/235-controlnet-stable-diffusion)

### Products Powered by OpenVINO

-	 [🤗Optimum Intel](https://github.com/huggingface/optimum-intel) -  a simple interface to optimize Transformers and Diffusers models.
-   [Neural Network Compression Framework (NNCF)](https://github.com/openvinotoolkit/nncf) - advanced model optimization techniques including quantization, filter pruning, binarization and sparsity.
-   [GenAI Repository](https://github.com/openvinotoolkit/openvino.genai) and [OpenVINO Tokenizers](https://github.com/openvinotoolkit/openvino_tokenizers) - resources and tools for developing and optimizing Generative AI applications.
-   [OpenVINO™ Model Server (OVMS)](https://github.com/openvinotoolkit/model_server) - a scalable, high-performance solution for serving models optimized for Intel architectures.
-   [Software Development Kit (SDK) for the Intel® Geti™](https://github.com/openvinotoolkit/geti-sdk) - an interactive video and image annotation tool for computer vision use cases.

Check out [Awesome OpenVINO](https://github.com/openvinotoolkit/awesome-openvino) repository to see all the community-made  AI projects based on OpenVINO!

### Documentation

**User documentation**

Find the latest documentation, from getting started guides to advanced topics, at [docs.openvino.ai](https://docs.openvino.ai/). User documentation contains detailed information about all OpenVINO components, providing you with essential details to create applications using OpenVINO.

**Developer documentation**

[Developer documentation](./docs/dev/index.md) on GitHub contains information about architectural decisions implemented within OpenVINO [components](./docs/dev/index.md#openvino-components). It offers all the essential information required for [building](./docs/dev/build.md)  and [contributing](./CONTRIBUTING.md) to OpenVINO.

To learn more, refer to the additional resources:

* [Product Page](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html)
* [Release Notes](https://docs.openvino.ai/2024/openvino_release_notes.html)
* [OpenVINO Blog](https://blog.openvino.ai/) and [OpenVINO™ toolkit on Medium](https://medium.com/@openvino)

### Contribution and Support

We welcome contributions of all kinds!

Explore the list of [Good First Issues](https://github.com/openvinotoolkit/openvino/issues/17502), if you're looking for a place to start contributing.
To be assigned to an issue, simply leave a comment with the `.take` command in the selected issue.
Check out [Contribution Guidelines](./CONTRIBUTING.md) for more details.

You can ask questions and get support on:

* [GitHub Issues](https://github.com/openvinotoolkit/openvino/issues)
* OpenVINO channels on the [Intel DevHub Discord server](https://discord.gg/7pVRxUwdWG)
* the [`openvino`](https://stackoverflow.com/questions/tagged/openvino) tag on Stack Overflow\*.

### License

OpenVINO™ Toolkit is licensed under [Apache License Version 2.0](LICENSE).
By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

**Telemetry**

OpenVINO™ collects software performance and usage data to improve OpenVINO™ tools. This data is collected directly by OpenVINO™ or through the use of Google Analytics 4. Visit our [telemetry page](https://docs.openvino.ai/nightly/openvino_docs_telemetry_information.html) to learn more.
You can opt out at any time by running the command:

``` bash
opt_in_out --opt_out
```

---
\* Other names and brands may be claimed as the property of others.

[OpenVINO™ Runtime]:https://docs.openvino.ai/2023.1/openvino_docs_OV_UG_OV_Runtime_User_Guide.html
[OpenVINO Model Converter (OVC)]:https://docs.openvino.ai/2023.1/openvino_docs_model_processing_introduction.html#convert-a-model-in-cli-ovc
[Samples]:https://github.com/openvinotoolkit/openvino/tree/master/samples
