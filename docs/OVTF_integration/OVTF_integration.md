# OpenVINO™ integration with TensorFlow {#ovtf_integration}

@sphinxdirective

.. toctree:: 
   :maxdepth: 1 
   :hidden:

   ovtf_install
   ovtf_build
   ovtf_usage
   ovtf_models
   ovtf_architecture
   ovtf_troubleshooting

@endsphinxdirective


**OpenVINO™ integration with TensorFlow** has been designed for TensorFlow developers who want to get started with OpenVINO™ in their inferencing applications. They can now take advantage of [OpenVINO™ toolkit](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html) optimizations with TensorFlow inference applications across a wide range of Intel® computation devices by adding just two lines of code.

```bash
import openvino_tensorflow
openvino_tensorflow.set_backend('<backend_name>')
```

This product delivers OpenVINO™ inline optimizations which enhance inferencing performance with minimal code modifications. **OpenVINO™ integration with TensorFlow** accelerates inference across many [AI models](ovms_models.md) on a variety of Intel® technologies, such as:
- Intel® CPUs
- Intel® integrated GPUs
- Intel® Movidius™ Vision Processing Units - referred to as VPU
- Intel® Vision Accelerator Design with 8 Intel Movidius™ MyriadX VPUs - referred to as VAD-M or HDDL

**Note:** For maximum performance, efficiency, tooling customization, and hardware control, we recommend developers to adopt native OpenVINO™ APIs and its runtime.


### Installation

**Installation requirements:**
- Ubuntu 18.04, 20.04, macOS 11.2.3, or Windows 10 - 64 bit
- Python* 3.7, 3.8 or 3.9
- TensorFlow* v2.7.0

Note that the Windows package is released in a Beta preview mode and currently supports only Python3.9 


Check our [Interactive Installation Table](https://openvinotoolkit.github.io/openvino_tensorflow/) for a menu of installation options. The table will help you configure the installation process.

The **OpenVINO™ integration with TensorFlow** package comes with pre-built libraries of OpenVINO™ version 2021.4.2. The users do not have to install OpenVINO™ separately. This package supports:
- Intel® CPUs
- Intel® integrated GPUs
- Intel® Movidius™ Vision Processing Units (VPUs)

    pip3 install -U pip
    pip3 install tensorflow==2.7.0
    pip3 install -U openvino-tensorflow

For installation instructions on Windows please refer to [OpenVINO™ integration with TensorFlow for Windows](docs/INSTALL.md#InstallOpenVINOintegrationwithTensorFlowalongsideTensorFlow)

To use Intel® integrated GPUs for inference, make sure to install the [Intel® Graphics Compute Runtime for OpenCL™ drivers](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_linux.html#install-gpu)

To leverage Intel® Vision Accelerator Design with Movidius™ (VAD-M) for inference, install [OpenVINO™ integration with TensorFlow alongside the Intel® Distribution of OpenVINO™ Toolkit](docs/INSTALL.md#12-install-openvino-integration-with-tensorflow-alongside-the-intel-distribution-of-openvino-toolkit).

For more details, please refer to the [installation](ovtf_install.md) and [build from source](ovtf_build.md) guides.

### Configuration

Once you've installed **OpenVINO™ integration with TensorFlow**, you can use TensorFlow to run inference using a trained model.

For the best results, it is advised to enable [oneDNN Deep Neural Network Library (oneDNN)](https://github.com/oneapi-src/oneDNN) by setting the environment variable of `TF_ENABLE_ONEDNN_OPTS=1`.

To see if **OpenVINO™ integration with TensorFlow** is properly installed, run:

    python3 -c "import tensorflow as tf; print('TensorFlow version: ',tf.__version__);\
                import openvino_tensorflow; print(openvino_tensorflow.__version__)"

This should produce an output like:

        TensorFlow version:  2.7.0
        OpenVINO integration with TensorFlow version: b'1.1.0'
        OpenVINO version used for this build: b'2021.4.2'
        TensorFlow version used for this build: v2.7.0
        CXX11_ABI flag used for this build: 0

By default, Intel® CPU is used to run inference. However, you can change the default option to either Intel® integrated GPU or Intel® VPU for AI inferencing. Invoke the following function to change the hardware on which inferencing is done.

    openvino_tensorflow.set_backend('<backend_name>')

Supported backends include 'CPU', 'GPU', 'GPU_FP16', 'MYRIAD', and 'VAD-M'.

To determine what processing units are available on your system for inference, use the following function:

    openvino_tensorflow.list_backends()
    
For more API calls and environment variables, see [USAGE.md](docs/USAGE.md).

**Note:** If a CUDA-capable device is present in the system then set the environment variable CUDA_VISIBLE_DEVICES to -1. 

## Examples

To see what you can do with **OpenVINO™ integration with TensorFlow**, explore the demos located in the [examples folder](https://github.com/openvinotoolkit/openvino_tensorflow/tree/master/examples) in our GitHub repository.  

## Try it on Intel® DevCloud
Sample tutorials are also hosted on [Intel<sup>®</sup> DevCloud](https://www.intel.com/content/www/us/en/developer/tools/devcloud/edge/build/ovtfoverview.html). The demo applications are implemented using Jupyter Notebooks. You can interactively execute them on Intel® DevCloud nodes, compare the results of **OpenVINO™ integration with TensorFlow**, native TensorFlow and OpenVINO™. 

## License
**OpenVINO™ integration with TensorFlow** is licensed under [Apache License Version 2.0](LICENSE).
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
