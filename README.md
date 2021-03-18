# OpenVINO™ Toolkit
[![Stable release](https://img.shields.io/badge/version-2021.3-green.svg)](https://github.com/openvinotoolkit/openvino/releases/tag/2021.3)
[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](LICENSE)
![GitHub branch checks state](https://img.shields.io/github/checks-status/openvinotoolkit/openvino/master?label=GitHub%20checks)
![Azure DevOps builds (branch)](https://img.shields.io/azure-devops/build/openvinoci/b2bab62f-ab2f-4871-a538-86ea1be7d20f/13?label=Public%20CI)

This toolkit allows developers to deploy pre-trained deep learning models
through a high-level C++ Inference Engine API integrated with application logic.

This open source version includes several components: namely [Model Optimizer], [nGraph] and
[Inference Engine], as well as CPU, GPU, MYRIAD, multi device and heterogeneous plugins to accelerate deep learning inferencing on Intel® CPUs and Intel® Processor Graphics.
It supports pre-trained models from the [Open Model Zoo], along with 100+ open
source and public models in popular formats such as Caffe\*, TensorFlow\*,
MXNet\* and ONNX\*.

## HEADS UP: OneTBB transition is on the way
Currently, the default threading model of the OpenVINO is Intel® Threading Building Blocks (Intel® TBB).
Recently the TBB was revamped into the "oneTBB" (https://github.com/oneapi-src/oneTBB/)
While OpenVINO is still compiled with the pre-oneTBB versions (also moved to the https://github.com/oneapi-src/oneTBB/releases),
a transition to the oneTBB is imminent and we are just waiting for a critical mass of our customers to become ready for that.

So, if your private or third-party components are using the TBB directly, it is strongly advised to test them for compatibility with the oneTBB in advance.
As discussed in the official oneTBB migration guide, this may include code changes, performance implications and so on: 
https://software.intel.com/content/www/us/en/develop/documentation/onetbb-documentation/top/onetbb-developer-guide/migrating-from-threading-building-blocks-tbb.html

## Repository components:
* [Inference Engine]
* [nGraph]
* [Model Optimizer]

## License
Deep Learning Deployment Toolkit is licensed under [Apache License Version 2.0](LICENSE).
By contributing to the project, you agree to the license and copyright terms therein
and release your contribution under these terms.

## Resources:
* Docs: https://docs.openvinotoolkit.org/
* Wiki: https://github.com/openvinotoolkit/openvino/wiki
* Issue tracking: https://github.com/openvinotoolkit/openvino/issues
* Storage: https://storage.openvinotoolkit.org/
* Additional OpenVINO™ modules: https://github.com/openvinotoolkit/openvino_contrib
* [Intel® Distribution of OpenVINO™ toolkit Product Page](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html)
* [Intel® Distribution of OpenVINO™ toolkit Release Notes](https://software.intel.com/en-us/articles/OpenVINO-RelNotes)

## Support
Please report questions, issues and suggestions using:

* The [`openvino`](https://stackoverflow.com/questions/tagged/openvino) tag on StackOverflow\*
* [GitHub* Issues](https://github.com/openvinotoolkit/openvino/issues)
* [Forum](https://software.intel.com/en-us/forums/computer-vision)

---
\* Other names and brands may be claimed as the property of others.

[Open Model Zoo]:https://github.com/opencv/open_model_zoo
[Inference Engine]:https://software.intel.com/en-us/articles/OpenVINO-InferEngine
[Model Optimizer]:https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer
[nGraph]:https://docs.openvinotoolkit.org/latest/openvino_docs_nGraph_DG_DevGuide.html
[tag on StackOverflow]:https://stackoverflow.com/search?q=%23openvino

