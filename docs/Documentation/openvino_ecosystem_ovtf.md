# OpenVINO™ integration with TensorFlow {#ovtf_integration}

**OpenVINO™ integration with TensorFlow** is a solution for TensorFlow developers who want to get started with OpenVINO™ in their inferencing applications. By adding just two lines of code you can now take advantage of OpenVINO™ toolkit optimizations with TensorFlow inference applications across a range of Intel® computation devices.

This is all you need:
```bash
import openvino_tensorflow
openvino_tensorflow.set_backend('<backend_name>')
```

**OpenVINO™ integration with TensorFlow** accelerates inference across many AI models on a variety of Intel® technologies, such as:
- Intel® CPUs
- Intel® integrated GPUs
- Intel® Movidius™ Vision Processing Units - referred to as VPU
- Intel® Vision Accelerator Design with 8 Intel Movidius™ MyriadX VPUs - referred to as VAD-M or HDDL

> **NOTE**: For maximum performance, efficiency, tooling customization, and hardware control, we recommend developers to adopt native OpenVINO™ solutions.
To find out more about the product itself, as well as learn how to use it in your project, check its dedicated [GitHub repository](https://github.com/openvinotoolkit/openvino_tensorflow/tree/master/docs). 


To see what you can do with **OpenVINO™ integration with TensorFlow**, explore the demos located in the [examples folder](https://github.com/openvinotoolkit/openvino_tensorflow/tree/master/examples) in our GitHub repository.  

Sample tutorials are also hosted on [Intel® DevCloud](https://www.intel.com/content/www/us/en/developer/tools/devcloud/edge/build/ovtfoverview.html). The demo applications are implemented using Jupyter Notebooks. You can interactively execute them on Intel® DevCloud nodes, compare the results of **OpenVINO™ integration with TensorFlow**, native TensorFlow, and OpenVINO™. 

## License
**OpenVINO™ integration with TensorFlow** is licensed under [Apache License Version 2.0](https://github.com/openvinotoolkit/openvino_tensorflow/blob/master/LICENSE).
By contributing to the project, you agree to the license and copyright terms therein
and release your contribution under these terms.

## Support

Submit your questions, feature requests and bug reports via [GitHub issues](https://github.com/openvinotoolkit/openvino_tensorflow/issues).

## How to Contribute

We welcome community contributions to **OpenVINO™ integration with TensorFlow**. If you have an idea for improvement:

* Share your proposal via [GitHub issues](https://github.com/openvinotoolkit/openvino_tensorflow/issues).
* Submit a [pull request](https://github.com/openvinotoolkit/openvino_tensorflow/pulls).

We will review your contribution as soon as possible. If any additional fixes or modifications are necessary, we will guide you and provide feedback. Before you make your contribution, make sure you can build **OpenVINO™ integration with TensorFlow** and run all the examples with your fix/patch. If you want to introduce a large feature, create test cases for your feature. Upon our verification of your pull request, we will merge it to the repository provided that the pull request has met the above mentioned requirements and proved acceptable.

---
\* Other names and brands may be claimed as the property of others.