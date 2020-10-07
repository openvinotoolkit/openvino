# [OpenVINO™ Toolkit](https://01.org/openvinotoolkit) - Deep Learning Deployment Toolkit repository
[![Stable release](https://img.shields.io/badge/version-2021.1-green.svg)](https://github.com/openvinotoolkit/openvino/releases/tag/2021.1)
[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](LICENSE)
![Azure DevOps builds (branch)](https://img.shields.io/azure-devops/build/openvinoci/b2bab62f-ab2f-4871-a538-86ea1be7d20f/9/master?label=Public%20CI)

This toolkit allows developers to deploy pre-trained deep learning models
through a high-level C++ Inference Engine API integrated with application logic.

This open source version includes two components: namely [Model Optimizer] and
[Inference Engine], as well as CPU, GPU and heterogeneous plugins to accelerate
deep learning inferencing on Intel® CPUs and Intel® Processor Graphics.
It supports pre-trained models from the [Open Model Zoo], along with 100+ open
source and public models in popular formats such as Caffe\*, TensorFlow\*,
MXNet\* and ONNX\*.

## Repository components:
* [Inference Engine]
* [Model Optimizer]

## License
Deep Learning Deployment Toolkit is licensed under [Apache License Version 2.0](LICENSE).
By contributing to the project, you agree to the license and copyright terms therein
and release your contribution under these terms.

## Documentation
* [OpenVINO™ Release Notes](https://software.intel.com/en-us/articles/OpenVINO-RelNotes)
* [OpenVINO™ Inference Engine Build Instructions](build-instruction.md)
* [Get Started with Deep Learning Deployment Toolkit on Linux](get-started-linux.md)\*
* [Introduction to Deep Learning Deployment Toolkit](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_Introduction.html)
* [Inference Engine Developer Guide](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_Deep_Learning_Inference_Engine_DevGuide.html)
* [Model Optimizer Developer Guide](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Get Started with DockerHub CI for OpenVINO™ toolkit](https://github.com/openvinotoolkit/docker_ci/blob/master/README.md)

## How to Contribute
See [CONTRIBUTING](./CONTRIBUTING.md) for contribution to the code.
See [CONTRIBUTING_DOCS](./CONTRIBUTING_DOCS.md) for contribution to the documentation.
Thank you!

## Support
Please report questions, issues and suggestions using:

* The `openvino` [tag on StackOverflow]\*
* [GitHub* Issues](https://github.com/openvinotoolkit/openvino/issues)
* [Forum](https://software.intel.com/en-us/forums/computer-vision)

---
\* Other names and brands may be claimed as the property of others.

[Open Model Zoo]:https://github.com/opencv/open_model_zoo
[Inference Engine]:https://software.intel.com/en-us/articles/OpenVINO-InferEngine
[Model Optimizer]:https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer
[tag on StackOverflow]:https://stackoverflow.com/search?q=%23openvino
