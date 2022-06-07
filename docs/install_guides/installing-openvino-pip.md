# Install Intel® Distribution of OpenVINO™ Toolkit from PyPI Repository {#openvino_docs_install_guides_installing_openvino_pip}

You can install both OpenVINO™ Runtime and OpenVINO Development Tools through the PyPI repository. This page provides the main steps for installing OpenVINO Runtime.

> **NOTE**: From the 2022.1 release, the OpenVINO™ Development Tools can only be installed via PyPI. See [Install OpenVINO Development Tools](installing-model-dev-tools.md) for detailed steps.

## Installing OpenVINO Runtime

### System Requirements

Below are all the supported operating systems and required Python versions. 

|Operating System (64-bit)                         | [Python Version](https://www.python.org/downloads/) (64-bit)                   |
|--------------------------------------------------|--------------------------------------------------------------------------------|
|macOS 10.15.x versions                            | 3.6, 3.7, 3.8                                                                  |
|Red Hat Enterprise Linux 8                        | 3.6, 3.8                                                                       |
|Ubuntu 18.04 long-term support (LTS)              | 3.6, 3.7, 3.8                                                                  |
|Ubuntu 20.04 long-term support (LTS)              | 3.6, 3.7, 3.8                                                                  |
|Windows 10                                        | 3.6, 3.7, 3.8                                                                  |

For system requirements and troubleshooting, see the [OpenVINO™ Runtime](https://pypi.org/project/openvino/).

### Step 1. Set Up Python Virtual Environment

To avoid dependency conflicts, use a virtual environment. Skip this step only if you do want to install all dependencies globally.

Use the following command to create a virtual environment:

@sphinxdirective

.. tab:: Linux and macOS

   .. code-block:: sh
   
      python3 -m venv openvino_env
   
.. tab:: Windows

   .. code-block:: sh
   
      python -m venv openvino_env
     
     
@endsphinxdirective

### Step 2. Activate Virtual Environment

@sphinxdirective

.. tab:: On Linux and macOS

   .. code-block:: sh
   
      source openvino_env/bin/activate
   
.. tab:: On Windows

   .. code-block:: sh
   
      openvino_env\Scripts\activate
     
     
@endsphinxdirective

### Step 3. Set Up and Update PIP to the Highest Version

Use the following command:
```sh
python -m pip install --upgrade pip
```

### Step 4. Install the Package

Use the following command:
```
pip install openvino
```

### Step 5. Verify that the Package Is Installed

Run the command below:
```sh
python -c "from openvino.runtime import Core"
```

If installation was successful, you will not see any error messages (no console output).


@sphinxdirective

.. note::
   While installing Python, make sure to add it to system `PATH`.

@endsphinxdirective

### Step 1. Set Up Python Virtual Environment

To avoid dependency conflicts, use a virtual environment. Skip this step only if you do want to install all dependencies globally.

Use the following command to create a virtual environment:

@sphinxdirective

.. tab:: Linux and macOS

   .. code-block:: sh

      python3 -m venv openvino_env

.. tab:: Windows

   .. code-block:: sh

      python -m venv openvino_env


@endsphinxdirective

### Step 2. Activate Virtual Environment

@sphinxdirective

.. tab:: On Linux and macOS

   .. code-block:: sh

      source openvino_env/bin/activate

.. tab:: On Windows

   .. code-block:: sh

      openvino_env\Scripts\activate


@endsphinxdirective

### Step 3. Set Up and Update PIP to the Highest Version

Use the following command:
```sh
python -m pip install --upgrade pip
```

### Step 4. Install the Package

Use the following command:
```
pip install openvino
```

### Step 5. Verify that the Package Is Installed

Run the command below:
```sh
python -c "from openvino.runtime import Core"
```

If installation was successful, you will not see any error messages (no console output).

### Troubleshooting

Here you may find solutions to some common issues.

@sphinxdirective

.. tab:: Linux and macOS

   *ImportError: libpython3.7m.so.1.0: cannot open shared object file: No such file or directory*
   
   A possible cause is missing the external dependency on Ubuntu 18.04. To resolve this, run the following command:

   .. code-block:: sh

      sudo apt-get install libpython3.7

.. tab:: Windows

   *Microsoft Visual C++ 14.0 is required. Get it with "Build Tools for Visual Studio"*

   To resolve this issue, you need to install [Build Tools for Visual Studio 2019](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019) and then retry the installation steps.


@endsphinxdirective

## Installing OpenVINO Development Tools

OpenVINO Development Tools include Model Optimizer, Benchmark Tool, Accuracy Checker, Post-Training Optimization Tool and Open Model Zoo tools including Model Downloader. If you want to install OpenVINO Development Tools, OpenVINO Runtime will also be installed as a dependency, so you don't need to install OpenVINO Runtime separately.

See [Install OpenVINO™ Development Tools](installing-model-dev-tools.md) for detailed steps.

## What's Next?

Now you may continue with the following tasks:

* To convert models for use with OpenVINO, see [Model Optimizer Developer Guide](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
* See pre-trained deep learning models in our [Open Model Zoo](../model_zoo.md).
* Try out OpenVINO via [OpenVINO Notebooks](https://docs.openvino.ai/latest/notebooks/notebooks.html).
* To write your own OpenVINO™ applications, see [OpenVINO Runtime User Guide](../OV_Runtime_UG/openvino_intro.md).
* See sample applications in [OpenVINO™ Toolkit Samples Overview](../OV_Runtime_UG/Samples_Overview.md).


## Additional Resources

- Intel® Distribution of OpenVINO™ toolkit home page: <https://software.intel.com/en-us/openvino-toolkit>
- For IoT Libraries & Code Samples, see [Intel® IoT Developer Kit](https://github.com/intel-iot-devkit).