<div align="center">
<img src="docs/dev/assets/openvino-logo-purple-black.svg" width="400px">

[![PyPI Status](https://badge.fury.io/py/openvino.svg)](https://badge.fury.io/py/openvino)
[![Anaconda Status](https://anaconda.org/conda-forge/openvino/badges/version.svg)](https://anaconda.org/conda-forge/openvino)
[![brew Status](https://img.shields.io/homebrew/v/openvino)](https://formulae.brew.sh/formula/openvino)

[![PyPI Downloads](https://static.pepy.tech/badge/openvino)](https://pepy.tech/project/openvino)
[![Anaconda Downloads](https://anaconda.org/conda-forge/libopenvino/badges/downloads.svg)](https://anaconda.org/conda-forge/openvino/files)
[![brew Downloads](https://img.shields.io/homebrew/installs/dy/openvino)](https://formulae.brew.sh/formula/openvino)
 </div>

Welcome to OpenVINO™, an open-source software toolkit for optimizing and deploying deep learning models.

- **Inference Optimization**: Boost deep learning performance in computer vision, automatic speech recognition, natural language processing, and many other common tasks.
- **Flexible Model Support**: Use models trained with popular frameworks such as TensorFlow, PyTorch, ONNX, Keras, and PaddlePaddle.
- **Broad Platform Compatibility**: Reduce resource demands and efficiently deploy on a range of platforms from edge to cloud.
- **Community and Ecosystem**: Join an active community contributing to the enhancement of deep learning performance across various domains.

## Installation

[Get your preferred distribution of OpenVINO](https://docs.openvino.ai/2024/get-started/install-openvino.html) or use this command for quick installation:

```
pip install openvino
```

Check [system requirements](https://docs.openvino.ai/2024/about-openvino/system-requirements.html) and [supported devices](https://docs.openvino.ai/2024/about-openvino/compatibility-and-support/supported-devices.html) for detailed information.

## Tutorials

[OpenVINO Quickstart example](https://docs.openvino.ai/2024/notebooks/201-vision-monodepth-with-output.html) will walk you through the basics of deploying your first model.

Learn how to optimize and deploy popular models with the [OpenVINO Notebooks](https://github.com/openvinotoolkit/openvino_notebooks)📚:
- [Create an LLM-powered Chatbot using OpenVINO](https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/254-llm-chatbot/254-llm-chatbot.ipynb)
- [YOLOv8 Optimization](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/230-yolov8-optimization)
- [Text-to-Image Generation](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/235-controlnet-stable-diffusion)

## OpenVINO Ecosystem

-	 [🤗Optimum Intel](https://github.com/huggingface/optimum-intel) -  a simple interface to optimize Transformers and Diffusers models.
-   [Neural Network Compression Framework (NNCF)](https://github.com/openvinotoolkit/nncf) - advanced model optimization techniques including quantization, filter pruning, binarization, and sparsity.
-   [GenAI Repository](https://github.com/openvinotoolkit/openvino.genai) and [OpenVINO Tokenizers](https://github.com/openvinotoolkit/openvino_tokenizers) - resources and tools for developing and optimizing Generative AI applications.
-   [OpenVINO™ Model Server (OVMS)](https://github.com/openvinotoolkit/model_server) - a scalable, high-performance solution for serving models optimized for Intel architectures.
-   [Intel® Geti™](https://geti.intel.com/) - an interactive video and image annotation tool for computer vision use cases.

Check out the [Awesome OpenVINO](https://github.com/openvinotoolkit/awesome-openvino) repository to discover a collection of community-made AI projects based on OpenVINO!

## Documentation

[**User documentation**](https://docs.openvino.ai/)

Contains detailed information about OpenVINO and guides you from installation through optimizing and deploying models for your AI applications.

[**Developer documentation**](./docs/dev/index.md)

Focuses on how OpenVINO [components](./docs/dev/index.md#openvino-components) work and describes [building](./docs/dev/build.md)  and [contributing](./CONTRIBUTING.md) processes.

## Contribution and Support

Check out [Contribution Guidelines](./CONTRIBUTING.md) for more details.
Read the [Good First Issues section](./CONTRIBUTING.md#3-start-working-on-your-good-first-issue), if you're looking for a place to start contributing. We welcome contributions of all kinds!

You can ask questions and get support on:

* [GitHub Issues](https://github.com/openvinotoolkit/openvino/issues).
* OpenVINO channels on the [Intel DevHub Discord server](https://discord.gg/7pVRxUwdWG).
* The [`openvino`](https://stackoverflow.com/questions/tagged/openvino) tag on Stack Overflow\*.

## Additional Resources

* [Product Page](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html)
* [Release Notes](https://docs.openvino.ai/2024/about-openvino/release-notes-openvino.html)
* [OpenVINO Blog](https://blog.openvino.ai/)
* [OpenVINO™ toolkit on Medium](https://medium.com/@openvino)
* [Telemetry](https://docs.openvino.ai/2024/about-openvino/additional-resources/telemetry.html)

## License

OpenVINO™ Toolkit is licensed under [Apache License Version 2.0](LICENSE).
By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

---
\* Other names and brands may be claimed as the property of others.

