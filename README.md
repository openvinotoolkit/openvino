<div align="center">
<img src="docs/img/openvino-logo-purple-black.png" width="400px">

[![PyPI Status](https://badge.fury.io/py/openvino.svg)](https://badge.fury.io/py/openvino)
[![Anaconda Status](https://anaconda.org/conda-forge/openvino/badges/version.svg)](https://anaconda.org/conda-forge/openvino)
[![brew Status](https://img.shields.io/homebrew/v/openvino)](https://formulae.brew.sh/formula/openvino)

[![PyPI Downloads](https://static.pepy.tech/badge/openvino)](https://pepy.tech/project/openvino)
[![Anaconda Downloads](https://anaconda.org/conda-forge/libopenvino/badges/downloads.svg)](https://anaconda.org/conda-forge/openvino/files)
[![brew Downloads](https://img.shields.io/homebrew/installs/dy/openvino)](https://formulae.brew.sh/formula/openvino)
 </div>

## What is OpenVINO toolkit?

OpenVINO™ is an open-source toolkit for simple and efficient deployment of various deep learning models.

- Boost deep learning performance in computer vision, automatic speech recognition, natural language processing and other common tasks.
- Use models trained with popular frameworks lsuch as TensorFlow, Pytorch, ONNX, Keras, and PaddlePaddle.
- Reduce resource demands and efficiently deploy on a range of Intel® platforms from edge to cloud.

OpenVINO includes several components: [OpenVINO Model Converter (OVC)], [OpenVINO™ Runtime], as well as CPU, GPU, NPU, auto batch and heterogeneous plugins to accelerate deep learning inference on Intel® CPUs and Intel® Processor Graphics.

### How to Use  

**Documentation**: Detailed information on OpenVINO's features, components, and usage can be found in the [OpenVINO Documentation](https://docs.openvino.ai).

**Tutorials**: Explore a variety of tutorials in the [OpenVINO Notebooks Repository](https://github.com/openvinotoolkit/openvino_notebooks).

Check out these notebooks that show how to optimize and deploy popular models:
- [Create an LLM-powered Chatbot using OpenVINO](https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/254-llm-chatbot/254-llm-chatbot.ipynb)
- [YOLOv8 Optimization](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/230-yolov8-optimization)
- [A Text-to-Image Generation with ControlNet Conditioning and OpenVINO™](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/235-controlnet-stable-diffusion)

### Quick Start

To quickly install and start using OpenVINO, follow these steps:

**Installation**: [Configure your OpenVINO installation command](https://docs.openvino.ai/2023.1/openvino_docs_install_guides_overview.html) in a couple of clicks.

**Quick Start Example**: Try out a simple [inference example with OpenVINO](https://docs.openvino.ai/2023.1/get_started.html#quick-start-example-no-installation-required) by following the steps from a Jupyter Notebook (no installation required).

 
### Products which use OpenVINO

**продукты из эко-системы (всякие биндинги, контриб) и в будущем экстеншны аля токенайзеры**

-	[Optimum Intel](https://github.com/huggingface/optimum-intel) - 🤗 Optimum Intel: Accelerate inference with Intel optimization tools.
-	[RapidOCR](https://github.com/RapidAI/RapidOCR) - A cross platform OCR Library based on PaddleOCR & OnnxRuntime & OpenVINO.
-   [OpenCV](https://opencv.org/)
-   [ONNX Runtime](https://onnxruntime.ai/)
-   [TNN](https://github.com/Tencent/TNN/tree/master)
-   [Neural Network Compression Framework (NNCF)](https://github.com/openvinotoolkit/nncf) - a suite of advanced algorithms for model inference optimization including quantization, filter pruning, binarization and sparsity
-   [OpenVINO™ Training Extensions (OTE)](https://github.com/openvinotoolkit/training_extensions) - convenient environment to train Deep Learning models and convert them using OpenVINO for optimized inference.
-   [OpenVINO™ Model Server (OVMS)](https://github.com/openvinotoolkit/model_server) - a scalable, high-performance solution for serving deep learning models optimized for Intel architectures
-   [Software Development Kit (SDK) for the Intel® Geti™ platform for Computer Vision AI model training](https://github.com/openvinotoolkit/geti-sdk) - an online, interactive video and image annotation tool for computer vision purposes.
-   [Dataset Management Framework (Datumaro)](https://github.com/openvinotoolkit/datumaro) - a framework and CLI tool to build, transform, and analyze datasets.
-	[openvino-rs](https://github.com/intel/openvino-rs) New OpenVINO APIs
-	[OpenVINO.NET](https://github.com/sdcb/OpenVINO.NET)
-   [openvino_contrib](https://github.com/openvinotoolkit/openvino_contrib) - Additional OpenVINO™ toolkit modules

**хайповые репо, куда опенвино проинтегрирован**

-	[GIMP AI plugins](https://github.com/intel/openvino-ai-plugins-gimp) - GIMP AI plugins with OpenVINO Backend
-	[Frigate](https://github.com/blakeblackshear/frigate) – NVR with realtime local object detection for IP cameras
-	[whisper.cpp](https://github.com/ggerganov/whisper.cpp)
-	[langchain](https://github.com/langchain-ai/langchain)
-	[stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

### Documentation

#### User documentation

The latest documentation for OpenVINO™ Toolkit is available [here](https://docs.openvino.ai/). This documentation contains detailed information about all OpenVINO components and provides all the important information you may need to create an application based on binary OpenVINO distribution or own OpenVINO version without source code modification.

* [Intel® Distribution of OpenVINO™ toolkit Product Page](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html)
* [Intel® Distribution of OpenVINO™ toolkit Release Notes](https://www.intel.com/content/www/us/en/developer/articles/release-notes/openvino/2023-1.html)
* [OpenVINO Success Stories](https://www.intel.com/content/www/us/en/internet-of-things/ai-in-production/success-stories.html)
* [OpenVINO Blog](https://blog.openvino.ai/)
* [OpenVINO™ toolkit on Medium](https://medium.com/@openvino)

#### Developer documentation

[Developer documentation](./docs/dev/index.md) contains information about architectural decisions which are applied inside the OpenVINO components. This documentation has all necessary information which could be needed in order to contribute to OpenVINO.




Комментарии: 

Дальше уже можно добавить (или вынести отдельно) - где используется, как контрибутить, саппорт и т.д.

В целом коммент: 
Мб вставить в начало какое-то мини-видео, где демонстрируются хайповые юзкейсы, поддерживаемые в опенвино?
Лично я замечал, где больше «интерактива», там больше звездочек и люди больше понимают крутость продукта и его возможности.
 
Возможно надо определиться что мы хотим тут рассказывать. Если давать всю вводную про OV то тогда да, надо вставлять какие-то дифференциаторы, к примеру Run LLMs on iGPU или типа того.
Я вот подумал может быть можно было дать 2 предложения и сослаться куда-то, беда в том что такой страницы похоже нет.




## Supported Hardware 

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


More Information is available at https://docs.openvino.ai/nightly/openvino_docs_telemetry_information.html.


### Tutorials

The list of OpenVINO tutorials:

- [Jupyter notebooks](https://github.com/openvinotoolkit/openvino_notebooks)



## System requirements

The system requirements vary depending on platform and are available on dedicated pages:
- [Linux](https://docs.openvino.ai/2023.1/openvino_docs_install_guides_installing_openvino_linux_header.html)
- [Windows](https://docs.openvino.ai/2023.1/openvino_docs_install_guides_installing_openvino_windows_header.html)
- [macOS](https://docs.openvino.ai/2023.1/openvino_docs_install_guides_installing_openvino_macos_header.html)

## How to build

See [How to build OpenVINO](./docs/dev/build.md) to get more information about the OpenVINO build process.

## How to contribute

See [Contributions Welcome](https://github.com/openvinotoolkit/openvino/issues/17502) for good first issues.

See [CONTRIBUTING](./CONTRIBUTING.md) for contribution details. Thank you!

[Good First Issues](https://github.com/openvinotoolkit/openvino/issues/17502)
[Good First Issues Board](https://github.com/orgs/openvinotoolkit/projects/3)

Комментарии: 
•	Сюда следует добавить информацию про нашу активность с  Good First Issues. https://github.com/openvinotoolkit/openvino/issues/17502
 
Ilya: поддерживаю 


## Take the issue
If you wish to be assigned to an issue please add a comment with `.take` command.  

## Get a support

Report questions, issues and suggestions, using:

* [GitHub* Issues](https://github.com/openvinotoolkit/openvino/issues)
* The [`openvino`](https://stackoverflow.com/questions/tagged/openvino) tag on StackOverflow\*

* [OpenVINO Storage](https://storage.openvinotoolkit.org/)


### License
OpenVINO™ Toolkit is licensed under [Apache License Version 2.0](LICENSE).
By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

### Telemetry
OpenVINO™ collects software performance and usage data for the purpose of improving OpenVINO™ tools. This data is collected directly by OpenVINO™ or through the use of Google Analytics 4.
You can opt-out at any time by running the command:

``` bash
opt_in_out --opt_out
```


---
\* Other names and brands may be claimed as the property of others.

[OpenVINO™ Runtime]:https://docs.openvino.ai/2023.1/openvino_docs_OV_UG_OV_Runtime_User_Guide.html
[OpenVINO Model Converter (OVC)]:https://docs.openvino.ai/2023.1/openvino_docs_model_processing_introduction.html#convert-a-model-in-cli-ovc
[Samples]:https://github.com/openvinotoolkit/openvino/tree/master/samples
