<div align="center">
<img src="docs/dev/assets/openvino-logo-purple-black.svg" width="400px">

<h3 align="center">
Open-source software toolkit for optimizing and deploying deep learning models.
</h3>

<p align="center">
 <a href="https://docs.openvino.ai/2025/index.html"><b>Documentation</b></a> • <a href="https://blog.openvino.ai"><b>Blog</b></a> • <a href="https://docs.openvino.ai/2025/about-openvino/key-features.html"><b>Key Features</b></a> • <a href="https://docs.openvino.ai/2025/get-started/learn-openvino.html"><b>Tutorials</b></a> • <a href="https://docs.openvino.ai/2025/documentation/openvino-ecosystem.html"><b>Integrations</b></a> • <a href="https://docs.openvino.ai/2025/about-openvino/performance-benchmarks.html"><b>Benchmarks</b></a> • <a href="https://github.com/openvinotoolkit/openvino.genai"><b>Generative AI</b></a>
</p>

[![PyPI Status](https://badge.fury.io/py/openvino.svg)](https://badge.fury.io/py/openvino)
[![Anaconda Status](https://anaconda.org/conda-forge/openvino/badges/version.svg)](https://anaconda.org/conda-forge/openvino)
[![brew Status](https://img.shields.io/homebrew/v/openvino)](https://formulae.brew.sh/formula/openvino)

[![PyPI Downloads](https://static.pepy.tech/badge/openvino)](https://pepy.tech/project/openvino)
[![Anaconda Downloads](https://anaconda.org/conda-forge/libopenvino/badges/downloads.svg)](https://anaconda.org/conda-forge/openvino/files)
[![brew Downloads](https://img.shields.io/homebrew/installs/dy/openvino)](https://formulae.brew.sh/formula/openvino)
 </div>


- **Inference Optimization**: Boost deep learning performance in computer vision, automatic speech recognition, generative AI, natural language processing with large and small language models, and many other common tasks.
- **Flexible Model Support**: Use models trained with popular frameworks such as PyTorch, TensorFlow, ONNX, Keras, PaddlePaddle, and JAX/Flax. Directly integrate models built with transformers and diffusers from the Hugging Face Hub using Optimum Intel. Convert and deploy models without original frameworks.
- **Broad Platform Compatibility**: Reduce resource demands and efficiently deploy on a range of platforms from edge to cloud. OpenVINO™ supports inference on CPU (x86, ARM), GPU (Intel integrated & discrete GPU) and AI accelerators (Intel NPU).
- **Community and Ecosystem**: Join an active community contributing to the enhancement of deep learning performance across various domains.

Check out the [OpenVINO Cheat Sheet](https://docs.openvino.ai/2025/_static/download/OpenVINO_Quick_Start_Guide.pdf) and [Key Features](https://docs.openvino.ai/2025/about-openvino/key-features.html) for a quick reference.


## Installation

[Get your preferred distribution of OpenVINO](https://docs.openvino.ai/2025/get-started/install-openvino.html) or use this command for quick installation:

```sh
pip install -U openvino
```

Check [system requirements](https://docs.openvino.ai/2025/about-openvino/release-notes-openvino/system-requirements.html) and [supported devices](https://docs.openvino.ai/2025/documentation/compatibility-and-support/supported-devices.html) for detailed information.

## Tutorials and Examples

[OpenVINO Quickstart example](https://docs.openvino.ai/2025/get-started.html) will walk you through the basics of deploying your first model.

Learn how to optimize and deploy popular models with the [OpenVINO Notebooks](https://github.com/openvinotoolkit/openvino_notebooks)📚:
- [Create an LLM-powered Chatbot using OpenVINO](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/llm-chatbot/llm-chatbot-generate-api.ipynb)
- [YOLOv11 Optimization](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/yolov11-optimization/yolov11-object-detection.ipynb)
- [Text-to-Image Generation](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/text-to-image-genai/text-to-image-genai.ipynb)
- [Multimodal assistant with LLaVa and OpenVINO](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/llava-multimodal-chatbot/llava-multimodal-chatbot-genai.ipynb)
- [Automatic speech recognition using Whisper and OpenVINO](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/whisper-asr-genai/whisper-asr-genai.ipynb)

Discover more examples in the [OpenVINO Samples (Python & C++)](https://docs.openvino.ai/2025/get-started/learn-openvino/openvino-samples.html) and [Notebooks (Python)](https://docs.openvino.ai/2025/get-started/learn-openvino/interactive-tutorials-python.html).

Here are easy-to-follow code examples demonstrating how to run PyTorch and TensorFlow model inference using OpenVINO:

**PyTorch Model**

```python
import openvino as ov
import torch
import torchvision

# load PyTorch model into memory
model = torch.hub.load("pytorch/vision", "shufflenet_v2_x1_0", weights="DEFAULT")

# convert the model into OpenVINO model
example = torch.randn(1, 3, 224, 224)
ov_model = ov.convert_model(model, example_input=(example,))

# compile the model for CPU device
core = ov.Core()
compiled_model = core.compile_model(ov_model, 'CPU')

# infer the model on random data
output = compiled_model({0: example.numpy()})
```

**TensorFlow Model**

```python
import numpy as np
import openvino as ov
import tensorflow as tf

# load TensorFlow model into memory
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# convert the model into OpenVINO model
ov_model = ov.convert_model(model)

# compile the model for CPU device
core = ov.Core()
compiled_model = core.compile_model(ov_model, 'CPU')

# infer the model on random data
data = np.random.rand(1, 224, 224, 3)
output = compiled_model({0: data})
```

OpenVINO supports the CPU, GPU, and NPU [devices](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes.html) and works with models from PyTorch, TensorFlow, ONNX, TensorFlow Lite, PaddlePaddle, and JAX/Flax [frameworks](https://docs.openvino.ai/2025/openvino-workflow/model-preparation.html). It includes [APIs](https://docs.openvino.ai/2025/api/api_reference.html) in C++, Python, C, NodeJS, and offers the GenAI API for optimized model pipelines and performance.

## Generative AI with OpenVINO

Get started with the OpenVINO GenAI [installation](https://docs.openvino.ai/2025/get-started/install-openvino/install-openvino-genai.html) and refer to the [detailed guide](https://docs.openvino.ai/2025/openvino-workflow-generative/generative-inference.html) to explore the capabilities of Generative AI using OpenVINO.

Learn how to run LLMs and GenAI with [Samples](https://github.com/openvinotoolkit/openvino.genai/tree/master/samples) in the [OpenVINO™ GenAI repo](https://github.com/openvinotoolkit/openvino.genai). See GenAI in action with Jupyter notebooks: [LLM-powered Chatbot](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/llm-chatbot) and [LLM Instruction-following pipeline](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/llm-question-answering).

## Documentation

[User documentation](https://docs.openvino.ai/) contains detailed information about OpenVINO and guides you from installation through optimizing and deploying models for your AI applications.

[Developer documentation](./docs/dev/index.md) focuses on the OpenVINO architecture and describes [building](./docs/dev/build.md)  and [contributing](./CONTRIBUTING.md) processes.

## OpenVINO Ecosystem

### OpenVINO Tools

-   [Neural Network Compression Framework (NNCF)](https://github.com/openvinotoolkit/nncf) - advanced model optimization techniques including quantization, filter pruning, binarization, and sparsity.
-   [GenAI Repository](https://github.com/openvinotoolkit/openvino.genai) and [OpenVINO Tokenizers](https://github.com/openvinotoolkit/openvino_tokenizers) - resources and tools for developing and optimizing Generative AI applications.
-   [OpenVINO™ Model Server (OVMS)](https://github.com/openvinotoolkit/model_server) - a scalable, high-performance solution for serving models optimized for Intel architectures.
-   [Intel® Geti™](https://geti.intel.com/) - an interactive video and image annotation tool for computer vision use cases.

### Integrations

-   [🤗Optimum Intel](https://github.com/huggingface/optimum-intel) - grab and use models leveraging OpenVINO within the Hugging Face API.
-   [Torch.compile](https://docs.openvino.ai/2025/openvino-workflow/torch-compile.html) - use OpenVINO for Python-native applications by JIT-compiling code into optimized kernels.
-   [OpenVINO LLMs inference and serving with vLLM​](https://docs.vllm.ai/en/stable/getting_started/openvino-installation.html) - enhance vLLM's fast and easy model serving with the OpenVINO backend.
-   [OpenVINO Execution Provider for ONNX Runtime](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html) - use OpenVINO as a backend with your existing ONNX Runtime code.
-   [LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/llm/openvino/) - build context-augmented GenAI applications with the LlamaIndex framework and enhance runtime performance with OpenVINO.
-   [LangChain](https://python.langchain.com/docs/integrations/llms/openvino/) - integrate OpenVINO with the LangChain framework to enhance runtime performance for GenAI applications.
-   [Keras 3](https://github.com/keras-team/keras) - Keras 3 is a multi-backend deep learning framework. Users can switch model inference to the OpenVINO backend using the Keras API.

Check out the [Awesome OpenVINO](https://github.com/openvinotoolkit/awesome-openvino) repository to discover a collection of community-made AI projects based on OpenVINO!

## Performance

Explore [OpenVINO Performance Benchmarks](https://docs.openvino.ai/2025/about-openvino/performance-benchmarks.html) to discover the optimal hardware configurations and plan your AI deployment based on verified data.

## Contribution and Support

Check out [Contribution Guidelines](./CONTRIBUTING.md) for more details.
Read the [Good First Issues section](./CONTRIBUTING.md#3-start-working-on-your-good-first-issue), if you're looking for a place to start contributing. We welcome contributions of all kinds!

You can ask questions and get support on:

* [GitHub Issues](https://github.com/openvinotoolkit/openvino/issues).
* OpenVINO channels on the [Intel DevHub Discord server](https://discord.gg/7pVRxUwdWG).
* The [`openvino`](https://stackoverflow.com/questions/tagged/openvino) tag on Stack Overflow\*.


## Resources

* [Release Notes](https://docs.openvino.ai/2025/about-openvino/release-notes-openvino.html)
* [OpenVINO Blog](https://blog.openvino.ai/)
* [OpenVINO™ toolkit on Medium](https://medium.com/@openvino)


## Telemetry

OpenVINO™ collects software performance and usage data for the purpose of improving OpenVINO™ tools.
This data is collected directly by OpenVINO™ or through the use of Google Analytics 4.
You can opt-out at any time by running the command:

``` bash
opt_in_out --opt_out
```

More Information is available at [OpenVINO™ Telemetry](https://docs.openvino.ai/2025/about-openvino/additional-resources/telemetry.html).

## License

OpenVINO™ Toolkit is licensed under [Apache License Version 2.0](LICENSE).
By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

---
\* Other names and brands may be claimed as the property of others.
