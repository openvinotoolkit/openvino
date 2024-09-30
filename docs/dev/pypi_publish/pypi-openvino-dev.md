# OpenVINO™ Development Tools

<!--- The note below is intended for master branch only for pre-release purpose. Remove it for official releases. --->
> **NOTE**: This version is pre-release software and has not undergone full release validation or qualification. No support is offered on pre-release software and APIs/behavior are subject to change. It should NOT be incorporated into any production software/solution and instead should be used only for early testing and integration while awaiting a final release version of this software.

> **NOTE**: OpenVINO™ Development Tools package has been deprecated and will be discontinued with 2025.0 release. To learn more, refer to the [OpenVINO Legacy Features and Components page](https://docs.openvino.ai/2024/documentation/legacy-features.html).

Intel® Distribution of OpenVINO™ toolkit is an open-source toolkit for optimizing and deploying AI inference. It can be used to develop applications and solutions based on deep learning tasks, such as: emulation of human vision, automatic speech recognition, natural language processing, recommendation systems, etc. It provides high-performance and rich deployment options, from edge to cloud.

OpenVINO™ Development Tools enables you to download models from Open Model Zoo, convert your own models to OpenVINO IR, as well as optimize and tune pre-trained deep learning models. See [What's in the Package](#whats-in-the-package) for more information.

## System Requirements

Before you start the installation, check the supported operating systems and required Python* versions. The complete list of supported hardware is available in the [System Requirements](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/system-requirements.html).

**C++ libraries** are also required for the installation on Windows*. To install that, you can [download the Visual Studio Redistributable file (.exe)](https://aka.ms/vs/17/release/vc_redist.x64.exe).

> **NOTE**: This package can be installed on other versions of macOS, Linux and Windows, but only the specific versions above are fully validated.

## Install the OpenVINO™ Development Tools Package

There are two options to install OpenVINO Development Tools: installation into an existing environment with a deep learning framework used for model training or creation;
or installation in a new environment.

### Installation into an Existing Environment with the Source Deep Learning Framework

To install OpenVINO Development Tools (see the [What's in the Package](#whats-in-the-package) section of this article) into an existing environment
with the source deep learning framework used for model training or creation, run the following command:
```
pip install openvino-dev
```

### Installation in a New Environment

If you do not have an environment with the source deep learning framework for the input model or you encounter any compatibility issues between OpenVINO and your version of deep learning framework,
you may install OpenVINO Development Tools with validated versions of frameworks into a new environment.

#### Step 1. Set Up Python Virtual Environment

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

> **NOTE**: On Linux and macOS, you may need to [install pip](https://pip.pypa.io/en/stable/installation/). For example, on Ubuntu execute the following command to get pip installed: `sudo apt install python3-venv python3-pip`.

#### Step 2. Activate Virtual Environment

On Linux and macOS:
```sh
source openvino_env/bin/activate
```
On Windows:
```sh
openvino_env\Scripts\activate
```

#### Step 3. Set Up and Update PIP to the Highest Version

Run the command below:
```sh
python -m pip install --upgrade pip
```

#### Step 4. Install the Package

Use the following command:
```sh
pip install openvino-dev[extras]
```
 where `extras` is the source deep learning framework for the input model and is one or more of the following values separated with "," :

| Extras Value                    | DL Framework                                                                     |
| :-------------------------------| :------------------------------------------------------------------------------- |
| caffe                           |   [Caffe*](https://caffe.berkeleyvision.org/)                                    |
| kaldi                           |   [Kaldi*](https://github.com/kaldi-asr/kaldi)                                   |
| onnx                            |   [ONNX*](https://github.com/microsoft/onnxruntime/)                             |
| pytorch                         |   [PyTorch*](https://pytorch.org/)                                               |
| tensorflow                      |   [TensorFlow* 1.x](https://www.tensorflow.org/versions#tensorflow_1)            |
| tensorflow2                     |   [TensorFlow* 2.x](https://www.tensorflow.org/versions#tensorflow_2)            |

For example, to install and configure the components for working with TensorFlow 2.x and ONNX models, use the following command:
   ```sh
   pip install openvino-dev[tensorflow2,onnx]
   ```
> **NOTE**: Model conversion API support for TensorFlow 1.x environment has been deprecated. Use TensorFlow 2.x environment to convert both TensorFlow 1.x and 2.x models.

> **NOTE**: On macOS, you may need to enclose the package name in quotes: `pip install "openvino-dev[extras]"`.

## How to Verify that the Package Is Installed

- To verify that the **developer package** is properly installed, run the command below (this may take a few seconds):
   ```sh
   mo -h
   ```
   You will see the help message for ``mo`` if installation finished successfully.

- To verify that OpenVINO Runtime from the **runtime package** is available, run the command below:
   ```sh
   python -c "from openvino import Core; print(Core().available_devices)"
   ```
   If installation was successful, you will see a list of available devices.

<a id="whats-in-the-package"></a>

## What's in the Package?

> **NOTE**: The openvino-dev package installs [OpenVINO™ Runtime](https://pypi.org/project/openvino) as a dependency, which is the engine that runs the deep learning model and includes a set of libraries for an easy inference integration into your applications.

**In addition, the openvino-dev package installs the following components by default:**

| Component        | Console Script                                                                  | Description                                                                                                                                                                                                                                                                                                   |
|------------------|---------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Legacy Model conversion API](https://docs.openvino.ai/2024/documentation/legacy-features/transition-legacy-conversion-api/legacy-conversion-api.html) | `mo` |**Model conversion API** imports, converts, and optimizes models that were trained in popular frameworks to a format usable by OpenVINO components. <br>Supported frameworks include Caffe\*, TensorFlow\*, MXNet\*, PaddlePaddle\*, and ONNX\*.                                               |                                         |
| [Model Downloader and other Open Model Zoo tools](https://docs.openvino.ai/2024/omz_tools_downloader.html)| `omz_downloader` <br> `omz_converter` <br> `omz_quantizer` <br> `omz_info_dumper`| **Model Downloader** is a tool for getting access to the collection of high-quality and extremely fast pre-trained deep learning [public](@ref omz_models_group_public) and [Intel](@ref omz_models_group_intel)-trained models. These free pre-trained models can be used to speed up the development and production deployment process without training your own models. The tool downloads model files from online sources and, if necessary, patches them to make them more usable with model conversion API. A number of additional tools are also provided to automate the process of working with downloaded models:<br> **Model Converter** is a tool for converting Open Model Zoo models that are stored in an original deep learning framework format into the OpenVINO Intermediate Representation (IR) using model conversion API. <br> **Model Quantizer** is a tool for automatic quantization of full-precision models in the IR format into low-precision versions using the Post-Training Optimization Tool. <br> **Model Information Dumper** is a helper utility for dumping information about the models to a stable, machine-readable format.                                          |

## Troubleshooting

For general troubleshooting steps and issues, see [Troubleshooting Guide for OpenVINO Installation](https://docs.openvino.ai/2024/get-started/troubleshooting-install-config.html). The following sections also provide explanations to several error messages.

### Errors with Installing via PIP for Users in China

Users in China might encounter errors while downloading sources via PIP during OpenVINO™ installation. To resolve the issues, try the following solution:

* Add the download source using the ``-i`` parameter with the Python ``pip`` command. For example:

   ``` sh
   pip install openvino-dev -i https://mirrors.aliyun.com/pypi/simple/
   ```
   Use the ``--trusted-host`` parameter if the URL above is ``http`` instead of ``https``.
   You can also run the following command to install openvino-dev with specific frameworks. For example:

   ```
   pip install openvino-dev[tensorflow2] -i https://mirrors.aliyun.com/pypi/simple/
   ```

### zsh: no matches found : openvino-dev[...]

If you use zsh (Z shell) interpreter, that is the default shell for macOS starting with version 10.15 (Catalina), you may encounter the following error while installing `openvino-dev` package with extras:

```sh
pip install openvino-dev[tensorflow2,caffe]
zsh: no matches found: openvino-dev[tensorflow2,caffe]
```

By default zsh interprets square brackets as an expression for pattern matching. To resolve this issue, you need to escape the command with quotes:

```sh
pip install 'openvino-dev[tensorflow2,caffe]'
```

To avoid such issues you can also disable globbing for PIP commands by defining an alias in `~/.zshrc` file:

```sh
alias pip='noglob pip'
```

### ERROR:root:Could not find OpenVINO Python API.

On Windows*, some libraries are necessary to run OpenVINO. To resolve this issue, install the [C++ redistributable (.exe)](https://aka.ms/vs/17/release/vc_redist.x64.exe). You can also view a full download list on the [official support page](https://docs.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).

### ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory

To resolve missing external dependency on Ubuntu* 18.04, execute the following command:
```sh
sudo apt-get install libpython3.8
```

## Additional Resources

- [Intel® Distribution of OpenVINO™ toolkit](https://software.intel.com/en-us/openvino-toolkit)
- [OpenVINO™ Documentation](https://docs.openvino.ai/)
- [OpenVINO™ Notebooks](https://github.com/openvinotoolkit/openvino_notebooks)
- [OpenVINO Installation Selector Tool](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html)

Copyright © 2018-2024 Intel Corporation
> **LEGAL NOTICE**: Your use of this software and any required dependent software (the
“Software Package”) is subject to the terms and conditions of the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0.html) for the Software Package, which may also include notices, disclaimers, or
license terms for third party or open source software included in or with the Software Package, and your use indicates your acceptance of all such terms. Please refer to the “third-party-programs.txt” or other similarly-named text file included with the Software Package for additional details.

>Intel is committed to the respect of human rights and avoiding complicity in human rights abuses, a policy reflected in the [Intel Global Human Rights Principles](https://www.intel.com/content/www/us/en/policy/policy-human-rights.html). Accordingly, by accessing the Intel material on this platform you agree that you will not use the material in a product or application that causes or contributes to a violation of an internationally recognized human right.
