# OpenVINO™ 

Intel® Distribution of OpenVINO™ toolkit is an open-source toolkit for optimizing and deploying AI inference. It can be used to develop applications and solutions based on deep learning tasks, such as: emulation of human vision, automatic speech recognition, natural language processing, recommendation systems, etc. It provides high-performance and rich deployment options, from edge to cloud.

If you have already finished developing your models and converting them to the OpenVINO model format, you can install OpenVINO Runtime to deploy your applications on various devices. The [OpenVINO™](https://docs.openvino.ai/2023.2/openvino_docs_OV_UG_OV_Runtime_User_Guide.html) Python package includes a set of libraries for an easy inference integration with your products.

## System Requirements

Before you start the installation, check the supported operating systems and required Python* versions. The complete list of supported hardware is available in the [System Requirements](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/system-requirements.html).

**C++ libraries** are also required for the installation on Windows*. To install that, you can [download the Visual Studio Redistributable file (.exe)](https://aka.ms/vs/17/release/vc_redist.x64.exe).

> **NOTE**: This package can be installed on other versions of Linux and Windows OSes, but only the specific versions above are fully validated.

## Install OpenVINO™ 

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

> **NOTE**: On Linux and macOS, you may need to [install pip](https://pip.pypa.io/en/stable/installation/). For example, on Ubuntu execute the following command to get pip installed: `sudo apt install python3-venv python3-pip`.

### Step 2. Activate Virtual Environment

On Windows:
```sh
openvino_env\Scripts\activate
```

On Linux and macOS:
```sh
source openvino_env/bin/activate
```

### Step 3. Set Up and Update PIP to the Highest Version

Run the command below:
```sh
python -m pip install --upgrade pip
```

### Step 4. Install the Package

Run the command below: <br>

   ```sh
   pip install openvino
   ```

### Step 5. Verify that the Package Is Installed

Run the command below:
```sh
python -c "from openvino.runtime import Core; print(Core().available_devices)"
```

If installation was successful, you will see the list of available devices.

## What's in the Package

| Component        | Content                                                                  | Description                                                                                                                                                                                                                                                                                                   |
|------------------|---------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [OpenVINO Runtime](https://docs.openvino.ai/2023.2/openvino_docs_OV_UG_OV_Runtime_User_Guide.html) | `openvino package` |**OpenVINO Runtime**  is a set of C++ libraries with C and Python bindings providing a common API to deliver inference solutions on the platform of your choice. Use the OpenVINO Runtime API to read PyTorch\*, TensorFlow\*, TensorFlow Lite\*, ONNX\*, and PaddlePaddle\* models and execute them on preferred devices. OpenVINO Runtime uses a plugin architecture and includes the following plugins: [CPU](https://docs.openvino.ai/2023.2/openvino_docs_OV_UG_supported_plugins_CPU.html), [GPU](https://docs.openvino.ai/2023.2/openvino_docs_OV_UG_supported_plugins_GPU.html), [Auto Batch](https://docs.openvino.ai/2023.2/openvino_docs_OV_UG_Automatic_Batching.html), [Auto](https://docs.openvino.ai/2023.2/openvino_docs_OV_UG_supported_plugins_AUTO.html), [Hetero](https://docs.openvino.ai/2023.2/openvino_docs_OV_UG_Hetero_execution.html).       
| [OpenVINO Model Converter (OVC)](https://docs.openvino.ai/2023.2/openvino_docs_model_processing_introduction.html#convert-a-model-in-cli-ovc) | `ovc` |**OpenVINO Model Converter**  converts models that were trained in popular frameworks to a format usable by OpenVINO components. <br>Supported frameworks include ONNX\*, TensorFlow\*, TensorFlow Lite\*, and PaddlePaddle\*.                                    |
| [Benchmark Tool](https://docs.openvino.ai/2023.2/openvino_inference_engine_tools_benchmark_tool_README.html)| `benchmark_app` | **Benchmark Application** allows you to estimate deep learning inference performance on supported devices for synchronous and asynchronous modes.                                              |

## Troubleshooting

For general troubleshooting steps and issues, see [Troubleshooting Guide for OpenVINO Installation](https://docs.openvino.ai/2023.2/openvino_docs_get_started_guide_troubleshooting.html). The following sections also provide explanations to several error messages. 

### Errors with Installing via PIP for Users in China

Users in China might encounter errors while downloading sources via PIP during OpenVINO™ installation. To resolve the issues, try the following solution:
   
* Add the download source using the ``-i`` parameter with the Python ``pip`` command. For example: 

   ``` sh
   pip install openvino -i https://mirrors.aliyun.com/pypi/simple/
   ```
   Use the ``--trusted-host`` parameter if the URL above is ``http`` instead of ``https``.

### ERROR:root:Could not find the Inference Engine or nGraph Python API.

On Windows*, some libraries are necessary to run OpenVINO. To resolve this issue, install the [C++ redistributable (.exe)](https://aka.ms/vs/17/release/vc_redist.x64.exe). You can also view a full download list on the [official support page](https://docs.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).

### ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory

To resolve missing external dependency on Ubuntu*, execute the following command:
```sh
sudo apt-get install libpython3.8
```

## Additional Resources

- [Intel® Distribution of OpenVINO™ toolkit](https://software.intel.com/en-us/openvino-toolkit)
- [OpenVINO™ Documentation](https://docs.openvino.ai/)
- [OpenVINO™ Notebooks](https://github.com/openvinotoolkit/openvino_notebooks)
- [OpenVINO Installation Selector Tool](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html)

Copyright © 2018-2023 Intel Corporation
> **LEGAL NOTICE**: Your use of this software and any required dependent software (the
“Software Package”) is subject to the terms and conditions of the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0.html) for the Software Package, which may also include notices, disclaimers, or
license terms for third party or open source software included in or with the Software Package, and your use indicates your acceptance of all such terms. Please refer to the “third-party-programs.txt” or other similarly-named text file included with the Software Package for additional details.

>Intel is committed to the respect of human rights and avoiding complicity in human rights abuses, a policy reflected in the [Intel Global Human Rights Principles](https://www.intel.com/content/www/us/en/policy/policy-human-rights.html). Accordingly, by accessing the Intel material on this platform you agree that you will not use the material in a product or application that causes or contributes to a violation of an internationally recognized human right.
