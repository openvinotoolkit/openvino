# OpenVINO™ Development Tools 

Intel® Distribution of OpenVINO™ toolkit is an open-source toolkit for optimizing and deploying AI inference. It can be used to develop applications and solutions based on deep learning tasks, such as: emulation of human vision, automatic speech recognition, natural language processing, recommendation systems, etc. It provides high-performance and rich deployment options, from edge to cloud.

OpenVINO™ Development Tools enables you to download models from Open Model Zoo, convert your own models to OpenVINO IR, as well as optimize and tune pre-trained deep learning models. See [What's in the Package](#whats-in-the-package) for more information.

## System Requirements

Before you start the installation, check the supported operating systems and required Python* versions. The complete list of supported hardware is available in the [Release Notes](https://www.intel.com/content/www/us/en/developer/articles/release-notes/openvino-relnotes.html).

| Supported Operating System                                   | [Python* Version (64-bit)](https://www.python.org/) |
| :------------------------------------------------------------| :---------------------------------------------------|
|   Ubuntu* 18.04 long-term support (LTS), 64-bit              | 3.6, 3.7, 3.8                                       |
|   Ubuntu* 20.04 long-term support (LTS), 64-bit              | 3.6, 3.7, 3.8, 3.9                                  |
|   Red Hat* Enterprise Linux* 8, 64-bit                       | 3.6, 3.8                                            |
|   macOS* 10.15.x                                             | 3.6, 3.7, 3.8, 3.9                                  |
|   Windows 10*, 64-bit                                        | 3.6, 3.7, 3.8, 3.9                                  |

> **NOTE**: This package can be installed on other versions of macOS, Linux and Windows, but only the specific versions above are fully validated.

> **NOTE**: The current version of the OpenVINO™ Runtime for macOS* supports inference on Intel® CPUs only.

## Install the OpenVINO™ Development Tools Package

### Step 1. Set Up Python Virtual Environment

Use a virtual environment to avoid dependency conflicts. 

To create a virtual environment, use the following commands:

On Windows:
```sh
python -m venv openvino_env
```

On Linux and macOS:
```sh
python3 -m venv openvino_env
```

> **NOTE**: On Linux and macOS, you may need to [install pip](https://pip.pypa.io/en/stable/installing/). For example, on Ubuntu execute the following command to get pip installed: `sudo apt install python3-venv python3-pip`.

### Step 2. Activate Virtual Environment

On Linux and macOS:
```sh
source openvino_env/bin/activate
```
On Windows:
```sh
openvino_env\Scripts\activate
```

### Step 3. Set Up and Update PIP to the Highest Version

Run the command below:
```sh
python -m pip install --upgrade pip
```

### Step 4. Install the Package

There are two options to install OpenVINO Development Tools:

#### Installing Default Components

To install [default components in the package](#whats-in-the-package), use the following command:
```
pip install openvino-dev
```

#### Installing Components for Specific Frameworks

To install and configure the components of the package for working with specific frameworks, use the following command:
```sh
pip install openvino-dev[extras]
```
 where `extras` has the following values: 

| Extras Value                    | DL Framework                                                                     |
| :-------------------------------| :------------------------------------------------------------------------------- |
| caffe                           |   [Caffe*](https://caffe.berkeleyvision.org/)                                    |
| kaldi                           |   [Kaldi*](https://github.com/kaldi-asr/kaldi)                                   |
| mxnet                           |   [Apache MXNet*](https://mxnet.apache.org/)                                            |
| onnx                            |   [ONNX*](https://github.com/microsoft/onnxruntime/)                             |
| pytorch                         |   [PyTorch*](https://pytorch.org/)                                               |
| tensorflow                      |   [TensorFlow* 1.x](https://www.tensorflow.org/versions#tensorflow_1)            |
| tensorflow2                     |   [TensorFlow* 2.x](https://www.tensorflow.org/versions#tensorflow_2)            |

For example, to install and configure the components for working with TensorFlow 2.x, MXNet and Caffe, use the following command:  
   ```sh
   pip install openvino-dev[tensorflow2,mxnet,caffe]
   ```
> **NOTE**: Model Optimizer support for TensorFlow 1.x environment has been deprecated. Use TensorFlow 2.x environment to convert both TensorFlow 1.x and 2.x models.

### Step 5. Verify that the Package Is Installed

- To verify that the **developer package** is properly installed, run the command below (this may take a few seconds):
   ```sh
   mo -h
   ```
   You will see the help message for Model Optimizer if installation finished successfully.

- To verify that OpenVINO Runtime from the **runtime package** is available, run the command below:
   ```sh
   python -c "from openvino.runtime import Core"
   ```
   If installation was successful, you will not see any error messages (no console output).

## What's in the Package?

> **NOTE**: The openvino-dev package installs [OpenVINO™ Runtime](https://pypi.org/project/openvino) as a dependency, which is the engine that runs the deep learning model and includes a set of libraries for an easy inference integration into your applications.  

**In addition, the openvino-dev package installs the following components by default:**

| Component        | Console Script                                                                   | Description                                                                                                                                                                                                                                                                                                   |  
|------------------|---------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Model Optimizer](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md) | `mo` |**Model Optimizer** imports, converts, and optimizes models that were trained in popular frameworks to a format usable by OpenVINO components. <br>Supported frameworks include Caffe\*, TensorFlow\*, MXNet\*, PaddlePaddle\*, and ONNX\*.                                               |
| [Benchmark Tool](../../tools/benchmark_tool/README.md)| `benchmark_app` | **Benchmark Application** allows you to estimate deep learning inference performance on supported devices for synchronous and asynchronous modes.                                              |
| [Accuracy Checker](@ref omz_tools_accuracy_checker) and <br> [Annotation Converter](@ref omz_tools_accuracy_checker_annotation_converters) | `accuracy_check` <br> `convert_annotation` |**Accuracy Checker**  is a deep learning accuracy validation tool that allows you to collect accuracy metrics against popular datasets. The main advantages of the tool are the flexibility of configuration and a set of supported datasets, preprocessing, postprocessing, and metrics. <br> **Annotation Converter** is a utility that prepares datasets for evaluation with Accuracy Checker.                                             |
| [Post-Training Optimization Tool](../../tools/pot/docs/pot_introduction.md)| `pot` |**Post-Training Optimization Tool** allows you to optimize trained models with advanced capabilities, such as quantization and low-precision optimizations, without the need to retrain or fine-tune models.                                            |
| [Model Downloader and other Open Model Zoo tools](@ref omz_tools_downloader)| `omz_downloader` <br> `omz_converter` <br> `omz_quantizer` <br> `omz_info_dumper`| **Model Downloader** is a tool for getting access to the collection of high-quality and extremely fast pre-trained deep learning [public](@ref omz_models_group_public) and [Intel](@ref omz_models_group_intel)-trained models. These free pre-trained models can be used to speed up the development and production deployment process without training your own models. The tool downloads model files from online sources and, if necessary, patches them to make them more usable with Model Optimizer. A number of additional tools are also provided to automate the process of working with downloaded models:<br> **Model Converter** is a tool for converting Open Model Zoo models that are stored in an original deep learning framework format into the OpenVINO Intermediate Representation (IR) using Model Optimizer. <br> **Model Quantizer** is a tool for automatic quantization of full-precision models in the IR format into low-precision versions using the Post-Training Optimization Tool. <br> **Model Information Dumper** is a helper utility for dumping information about the models to a stable, machine-readable format.                                          |      

## Troubleshooting

For general troubleshooting steps and issues, see [Troubleshooting Guide for OpenVINO Installation](./troubleshooting.md). The following sections also provide explanations to several error messages. 

### zsh: no matches found : openvino-dev[...]

If you use zsh (Z shell) interpreter, that is the default shell for macOS starting with version 10.15 (Catalina), you may encounter the following error while installing `openvino-dev` package with extras:

```sh
pip install openvino-dev[tensorflow2,mxnet,caffe]
zsh: no matches found: openvino-dev[tensorflow2,mxnet,caffe]
```

By default zsh interprets square brackets as an expression for pattern matching. To resolve this issue, you need to escape the command with quotes: 

```sh
pip install 'openvino-dev[tensorflow2,mxnet,caffe]'
```

To avoid such issues you can also disable globbing for PIP commands by defining an alias in `~/.zshrc` file:

```sh
alias pip='noglob pip'
```

### Error: Microsoft Visual C++ 14.0 is required. Get it with "Build Tools for Visual Studio"

On Windows* some dependencies may require compilation from source when installing. To resolve this issue, you need to install [Build Tools for Visual Studio* 2019](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019) and repeat package installation.

### ImportError: libpython3.7m.so.1.0: cannot open shared object file: No such file or directory

To resolve missing external dependency on Ubuntu* 18.04, execute the following command:
```sh
sudo apt-get install libpython3.7
```

## Additional Resources

- [Intel® Distribution of OpenVINO™ toolkit](https://software.intel.com/en-us/openvino-toolkit)
- [OpenVINO™ Notebooks](https://github.com/openvinotoolkit/openvino_notebooks)

Copyright © 2018-2022 Intel Corporation
> **LEGAL NOTICE**: Your use of this software and any required dependent software (the
“Software Package”) is subject to the terms and conditions of the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0.html) for the Software Package, which may also include notices, disclaimers, or
license terms for third party or open source software included in or with the Software Package, and your use indicates your acceptance of all such terms. Please refer to the “third-party-programs.txt” or other similarly-named text file included with the Software Package for additional details.

>Intel is committed to the respect of human rights and avoiding complicity in human rights abuses, a policy reflected in the [Intel Global Human Rights Principles](https://www.intel.com/content/www/us/en/policy/policy-human-rights.html). Accordingly, by accessing the Intel material on this platform you agree that you will not use the material in a product or application that causes or contributes to a violation of an internationally recognized human right.
