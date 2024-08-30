# OpenVINO™

<!--- The note below is intended for master branch only for pre-release purpose. Remove it for official releases. --->
> **NOTE**: This version is pre-release software and has not undergone full release validation or qualification. No support is offered on pre-release software and APIs/behavior are subject to change. It should NOT be incorporated into any production software/solution and instead should be used only for early testing and integration while awaiting a final release version of this software.

Intel® Distribution of OpenVINO™ toolkit is an open-source toolkit for optimizing and deploying
AI inference. It can be used to develop applications and solutions based on deep learning tasks,
such as: emulation of human vision, automatic speech recognition, natural language processing,
recommendation systems, etc. It provides high-performance and rich deployment options, from
edge to cloud.

If you have chosen a model, you can integrate it with your application through OpenVINO™ and
deploy it on various devices. The OpenVINO™ Python package includes a set of libraries for easy
inference integration with your products.

## System Requirements

Before you start the installation, check the supported operating systems and required Python*
versions. The complete list of supported hardware is available on the
[System Requirements page](https://docs.openvino.ai/system_requirements).

**C++ libraries** are also required for the installation on Windows*. To install that, you can
[download the Visual Studio Redistributable file (.exe)](https://aka.ms/vs/17/release/vc_redist.x64.exe).

> **NOTE**: This package may work on other Linux and Windows versions but only the versions specified in system requirements are fully validated.

## Install OpenVINO™

### Step 1. Set Up Python Virtual Environment

Use a virtual environment to avoid dependency conflicts. To create a virtual environment, use
the following commands:

On Windows:
```sh
python -m venv openvino_env
```

On Linux and macOS:
```sh
python3 -m venv openvino_env
```

> **NOTE**: On Linux and macOS, you may need to [install pip](https://pip.pypa.io/en/stable/installation/).

### Step 2. Activate the Virtual Environment

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
python -c "from openvino import Core; print(Core().available_devices)"
```

If installation was successful, you will see the list of available devices.

## What's in the Package

<table>
  <tr>
    <th>Component</th>
    <th>Content</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><a href="https://docs.openvino.ai/2024/openvino-workflow/running-inference.html">OpenVINO Runtime</a></td>
    <td>`openvino package`</td>
    <td>OpenVINO Runtime is a set of C++ libraries with C and Python bindings providing a common
        API to deliver inference solutions on the platform of your choice. Use the OpenVINO
        Runtime API to read PyTorch, TensorFlow, TensorFlow Lite, ONNX, and PaddlePaddle models
        and execute them on preferred devices. OpenVINO Runtime uses a plugin architecture and
        includes the following plugins:
        <a href="https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html">CPU</a>,
        <a href="https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html">GPU</a>,
        <a href="https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/automatic-batching.html">Auto Batch</a>,
        <a href="https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/auto-device-selection.html">Auto</a>,
        <a href="https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/hetero-execution.html">Hetero</a>,
    </td>
  </tr>
  <tr>
    <td><a href="https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html#convert-a-model-in-cli-ovc">OpenVINO Model Converter (OVC)</a></td>
    <td>`ovc`</td>
    <td>OpenVINO Model Converter converts models that were trained in popular frameworks to a
        format usable by OpenVINO components. </br>Supported frameworks include ONNX, TensorFlow,
        TensorFlow Lite, and PaddlePaddle.
    </td>
  </tr>
  <tr>
    <td><a href="https://docs.openvino.ai/2024/learn-openvino/openvino-samples/benchmark-tool.html">Benchmark Tool</a></td>
    <td>`benchmark_app`</td>
    <td>Benchmark Application** allows you to estimate deep learning inference performance on
        supported devices for synchronous and asynchronous modes.
    </td>
</table>



## Troubleshooting

For general troubleshooting steps and issues, see
[Troubleshooting Guide for OpenVINO Installation](https://docs.openvino.ai/2024/get-started/troubleshooting-install-config.html).
The following sections also provide explanations to several error messages.

### Errors with Installing via PIP for Users in China

Users in China may encounter errors while downloading sources via PIP during OpenVINO™ installation.
To resolve the issues, try the following solution:

* Add the download source using the ``-i`` parameter with the Python ``pip`` command. For example:

   ``` sh
   pip install openvino -i https://mirrors.aliyun.com/pypi/simple/
   ```
   Use the ``--trusted-host`` parameter if the URL above is ``http`` instead of ``https``.

### ERROR:root:Could not find OpenVINO Python API.

On Windows, additional libraries may be necessary to run OpenVINO. To resolve this issue, install
the [C++ redistributable (.exe)](https://aka.ms/vs/17/release/vc_redist.x64.exe).
You can also view a full download list on the
[official support page](https://docs.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).

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

Copyright © 2018-2024 Intel Corporation
> **LEGAL NOTICE**: Your use of this software and any required dependent software (the
“Software Package”) is subject to the terms and conditions of the
[Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0.html) for the Software Package,
which may also include notices, disclaimers, or license terms for third party or open source
software included in or with the Software Package, and your use indicates your acceptance of all
such terms. Please refer to the “third-party-programs.txt” or other similarly-named text file
included with the Software Package for additional details.

>Intel is committed to the respect of human rights and avoiding complicity in human rights abuses,
a policy reflected in the [Intel Global Human Rights Principles](https://www.intel.com/content/www/us/en/policy/policy-human-rights.html).
Accordingly, by accessing the Intel material on this platform you agree that you will not use the
material in a product or application that causes or contributes to a violation of an
internationally recognized human right.

Intel, the Intel logo, and other Intel marks are trademarks of Intel Corporation or its
subsidiaries. Other names and brands may be claimed as the property of others.
