# OpenVINO™ Toolkit
[![Stable release](https://img.shields.io/badge/version-2022.1-green.svg)](https://github.com/openvinotoolkit/openvino/releases/tag/2022.1)
[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](LICENSE)
![GitHub branch checks state](https://img.shields.io/github/checks-status/openvinotoolkit/openvino/master?label=GitHub%20checks)
![Azure DevOps builds (branch)](https://img.shields.io/azure-devops/build/openvinoci/b2bab62f-ab2f-4871-a538-86ea1be7d20f/13?label=Public%20CI)
[![PyPI Downloads](https://pepy.tech/badge/openvino)](https://pepy.tech/project/openvino)

## Contents:

 - [What is OpenVINO?](#what-is-openvino-toolkit)
    - [Components](#components)
 - [Supported Hardware matrix](#supported-hardware-matrix)
 - [License](#license)
 - Documentation
 - Tutorials
 - Products which uses OpenVINO
 - System requirements
 - How to build
 - How to contribute
 - See also

## What is OpenVINO toolkit?

OpenVINO™ toolkit allows developers to deploy pre-trained deep learning models through a high-level OpenVINO™ Runtime C++ and Python APIs integrated with application logic.

This open source version includes several components: namely [Model Optimizer], [OpenVINO™ Runtime], [Post-Training Optimization Tool], as well as CPU, GPU, MYRIAD, multi device and heterogeneous plugins to accelerate deep learning inferencing on Intel® CPUs and Intel® Processor Graphics.
It supports pre-trained models from the [Open Model Zoo], along with 100+ open
source and public models in popular formats such as TensorFlow, ONNX, PaddlePaddle, MXNet, Caffe, Kaldi.

### Components
* [OpenVINO™ Runtime]
* [Model Optimizer]
* [Post-Training Optimization Tool]
* [Samples]

## Supported Hardware matrix

The OpenVINO™ Runtime can infer models on different hardware devices. This section provides the list of supported devices.

<table>
    <thead>
        <tr>
            <th>Device</th>
            <th>Plugin</th>
            <th>Library</th>
            <th>Location</th>
            <th>ShortDescription</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=2>CPU</td>
            <td> <a href="https://docs.openvino.ai/nightly/openvino_docs_OV_UG_supported_plugins_CPU.html#doxid-openvino-docs-o-v-u-g-supported-plugins-c-p-u">Intel CPU</a></tb>
            <td><b><i>openvino_intel_cpu_plugin</i></b></td>
            <td><a href="https://github.com/openvinotoolkit/openvino/tree/master/src/plugins/intel_cpu">openvino/src/plugins/intel_cpu</a></td>
            <td>Intel Xeon with Intel® Advanced Vector Extensions 2 (Intel® AVX2), Intel® Advanced Vector Extensions 512 (Intel® AVX-512), and AVX512_BF16, Intel Core Processors with Intel AVX2, Intel Atom Processors with Intel® Streaming SIMD Extensions (Intel® SSE)</td>
        </tr>
        <tr>
            <td> <a href="https://docs.openvino.ai/nightly/openvino_docs_OV_UG_supported_plugins_ARM_CPU.html">ARM CPU</a></tb>
            <td><b><i>openvino_arm_cpu_plugin</i></b></td>
            <td> <a href="https://github.com/openvinotoolkit/openvino_contrib/tree/master/modules/arm_plugin">openvino_contrib/modules/arm_plugin</a></td>
            <td>Raspberry Pi™ 4 Model B, Apple® Mac mini with M1 chip, NVIDIA® Jetson Nano™, Android™ devices
        </tr>
        <tr>
            <td>GPU</td>
            <td><a href="https://docs.openvino.ai/nightly/openvino_docs_OV_UG_supported_plugins_GPU.html#doxid-openvino-docs-o-v-u-g-supported-plugins-g-p-u">Intel GPU</a></td>
            <td><b><i>openvino_intel_gpu_plugin</i></b></td>
            <td><a href="https://github.com/openvinotoolkit/openvino/tree/master/src/plugins/intel_gpu">openvino/src/plugins/intel_gpu</a></td>
            <td>Intel Processor Graphics, including Intel HD Graphics and Intel Iris Graphics</td>
        </tr>
        <tr>
            <td>GNA</td>
            <td><a href="https://docs.openvino.ai/nightly/openvino_docs_OV_UG_supported_plugins_GNA.html#doxid-openvino-docs-o-v-u-g-supported-plugins-g-n-a">Intel GNA</a></td>
            <td><b><i>openvino_intel_gna_plugin</i></b></td>
            <td><a href="https://github.com/openvinotoolkit/openvino/tree/master/src/plugins/intel_gna">openvino/src/plugins/intel_gna</a></td>
            <td>Intel Speech Enabling Developer Kit, Amazon Alexa* Premium Far-Field Developer Kit, Intel Pentium Silver J5005 Processor, Intel Pentium Silver N5000 Processor, Intel Celeron J4005 Processor, Intel Celeron J4105 Processor, Intel Celeron Processor N4100, Intel Celeron Processor N4000, Intel Core i3-8121U Processor, Intel Core i7-1065G7 Processor, Intel Core i7-1060G7 Processor, Intel Core i5-1035G4 Processor, Intel Core i5-1035G7 Processor, Intel Core i5-1035G1 Processor, Intel Core i5-1030G7 Processor, Intel Core i5-1030G4 Processor, Intel Core i3-1005G1 Processor, Intel Core i3-1000G1 Processor, Intel Core i3-1000G4 Processor</td>
        </tr>
        <tr>
            <td>VPU</td>
            <td><a href="https://docs.openvino.ai/nightly/openvino_docs_IE_DG_supported_plugins_VPU.html#doxid-openvino-docs-i-e-d-g-supported-plugins-v-p-u">Myriad plugin</a></td>
            <td><b><i>openvino_intel_myriad_plugin</i></b></td>
            <td><a href="https://github.com/openvinotoolkit/openvino/tree/master/src/plugins/intel_myriad">openvino/src/plugins/intel_myriad</a></td>
            <td>Intel® Neural Compute Stick 2 powered by the Intel® Movidius™ Myriad™ X</td>
        </tr>
    </tbody>
</table>

Also OpenVINO™ Toolkit contains several plugins which should simplify to load model on several hardware devices:
<table>
    <thead>
        <tr>
            <th>Plugin</th>
            <th>Library</th>
            <th>Location</th>
            <th>ShortDescription</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><a href="https://docs.openvino.ai/nightly/openvino_docs_IE_DG_supported_plugins_AUTO.html#doxid-openvino-docs-i-e-d-g-supported-plugins-a-u-t-o">Auto</a></td>
            <td><b><i>openvino_auto_plugin</i></b></td>
            <td><a href="https://github.com/openvinotoolkit/openvino/tree/master/src/plugins/auto">openvino/src/plugins/auto</a></td>
            <td>Auto plugin enables selecting Intel device for inference automatically</td>
        </tr>
        <tr>
            <td><a href="https://docs.openvino.ai/nightly/openvino_docs_OV_UG_Hetero_execution.html#doxid-openvino-docs-o-v-u-g-hetero-execution">Hetero</a></td>
            <td><b><i>openvino_hetero_plugin</i></b></td>
            <td><a href="https://github.com/openvinotoolkit/openvino/tree/master/src/plugins/hetero">openvino/src/plugins/hetero</a></td>
            <td>Heterogeneous execution enables automatic inference splitting between several devices</td>
        </tr>
        <tr>
            <td><a href="https://docs.openvino.ai/nightly/openvino_docs_OV_UG_Running_on_multiple_devices.html#doxid-openvino-docs-o-v-u-g-running-on-multiple-devices">Multi</a></td>
            <td><b><i>openvino_auto_plugin</i></b></td>
            <td><a href="https://github.com/openvinotoolkit/openvino/tree/master/src/plugins/auto">openvino/src/plugins/auto</a></td>
            <td>Multi plugin enables simultaneous inference of the same model on several devices in parallel</td>
        </tr>
    </tbody>
</table>

## License
OpenVINO™ Toolkit is licensed under [Apache License Version 2.0](LICENSE).
By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

## Resources
* Docs: https://docs.openvino.ai/
* Wiki: https://github.com/openvinotoolkit/openvino/wiki
* Issue tracking: https://github.com/openvinotoolkit/openvino/issues
* Storage: https://storage.openvinotoolkit.org/
* Additional OpenVINO™ toolkit modules: https://github.com/openvinotoolkit/openvino_contrib
* [Intel® Distribution of OpenVINO™ toolkit Product Page](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html)
* [Intel® Distribution of OpenVINO™ toolkit Release Notes](https://software.intel.com/en-us/articles/OpenVINO-RelNotes)

## Support
Please report questions, issues and suggestions using:

* The [`openvino`](https://stackoverflow.com/questions/tagged/openvino) tag on StackOverflow\*
* [GitHub* Issues](https://github.com/openvinotoolkit/openvino/issues)
* [Forum](https://software.intel.com/en-us/forums/computer-vision)

---
\* Other names and brands may be claimed as the property of others.

[Open Model Zoo]:https://github.com/openvinotoolkit/open_model_zoo
[OpenVINO™ Runtime]:https://docs.openvino.ai/latest/openvino_docs_OV_Runtime_User_Guide.html
[Model Optimizer]:https://docs.openvino.ai/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html
[Post-Training Optimization Tool]:https://docs.openvino.ai/latest/pot_README.html
[Samples]:https://github.com/openvinotoolkit/openvino/tree/master/samples
[tag on StackOverflow]:https://stackoverflow.com/search?q=%23openvino

