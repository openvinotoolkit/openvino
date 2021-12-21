# Install and Configure Intel® Distribution of OpenVINO™ toolkit for Linux* {#openvino_docs_install_guides_installing_openvino_linux}

## Introduction

The [OpenVINO™ Toolkit](https://docs.openvinotoolkit.org/latest/index.html) installation on this page installs the following components:

* [Inference Engine](../IE_DG/Deep_Learning_Inference_Engine_DevGuide.md) - This is the engine that runs the deep learning model. It includes a set of libraries for easy inference integration into your applications.
* [Inference Engine Code Samples](../IE_DG/Samples_Overview.md) - A set of simple command-line applications demonstrating how to utilize OpenVINO capabilities.

> **NOTE**: Since the OpenVINO™ 2022.1 release, the following development tools: Model Optimizer, Post-Training Optimization Tool, Model Downloader and other Open Model Zoo tools, Accuracy Checker, and Annotation Converter are not part of the installer. These tools are now only available on [pypi.org](https://pypi.org/project/openvino-dev/).

## System Requirements

@sphinxdirective
.. tab:: Operating Systems

  * Ubuntu 18.04.x long-term support (LTS), 64-bit
  * Ubuntu 20.04.0 long-term support (LTS), 64-bit
  * Red Hat* Enterprise Linux* 8.2, 64-bit

  .. note::
     Since the OpenVINO™ 2022.1 release, CentOS 7.6, 64-bit is not longer supported.

.. tab:: Hardware

  Optimized for these processors:

  * 6th to 12th generation Intel® Core™ processors and Intel® Xeon® processors 
  * 3rd generation Intel® Xeon® Scalable processor (formerly code named Cooper Lake)
  * Intel® Xeon® Scalable processor (formerly Skylake and Cascade Lake)
  * Intel Atom® processor with support for Intel® Streaming SIMD Extensions 4.1 (Intel® SSE4.1)
  * Intel Pentium® processor N4200/5, N3350/5, or N3450/5 with Intel® HD Graphics
  * Intel® Iris® Xe MAX Graphics
  * Intel® Neural Compute Stick 2
  * Intel® Vision Accelerator Design with Intel® Movidius™ VPUs

.. tab:: Processor Notes

  Processor graphics are not included in all processors. 
  See `Product Specifications`_ for information about your processor.
  
  .. _Product Specifications: https://ark.intel.com/

@endsphinxdirective

## Overview

This guide provides step-by-step instructions on how to install the Intel® Distribution of OpenVINO™ toolkit. Links are provided for each type of compatible hardware including downloads, initialization and configuration steps. The following steps will be covered:

1. <a href="#install-openvino">Install the Intel® Distribution of OpenVINO™ Toolkit</a>
2. <a href="#install-external-dependencies">Install External Software Dependencies</a>
3. <a href="#set-the-environment-variables">Configure the Environment</a>
4. <a href="#model-optimizer">Download additional components (Optional)</a>
5. <a href="#optional-steps">Configure Inference on non-CPU Devices (Optional)</a>
6. <a href="#get-started">What's next?</a>

@sphinxdirective

.. important::
   Before you start your journey with installation of the OpenVINO, we encourage you to check up our :ref:`code samples <code samples>` in C, C++, and Python and :ref:`notebook tutorials <notebook tutorials>` that we prepared for you, so you could see all amazing things that you can achieve with our tool.

@endsphinxdirective

## <a name="install-openvino"></a>Step 1: Install the Intel® Distribution of OpenVINO™ Toolkit

1. Select and download the Intel® Distribution of OpenVINO™ toolkit installer file from [Intel® Distribution of OpenVINO™ toolkit for Linux*](https://software.intel.com/en-us/openvino-toolkit/choose-download).
2. Open a command prompt terminal window. You can use the keyboard shortcut: Ctrl+Alt+T
3. Change directories to where you downloaded the Intel Distribution of OpenVINO toolkit for Linux\* file.<br>
   If you downloaded the starter script to the current user's `Downloads` directory:
   ```sh
   cd ~/Downloads/
   ```
   You should find there a bootstrapper script `l_openvino_toolkit_p_<version>.sh`.
4. Add executable rights for the current user:
   ```sh
   chmod +x l_openvino_toolkit_p_<version>.sh
   ```
5. If you want to use graphical user interface (GUI) installation wizard, run the script without any parameters:
   ```sh
   ./l_openvino_toolkit_p_<version>.sh
   ```
   <br>You should see the following dialog open up:

   @sphinxdirective
   
   .. image:: _static/images/openvino-install-linux-installer-1.png
      :width: 400px
      :align: center
   
   @endsphinxdirective

   Otherwise, you can add parameters `-a` for additional arguments and `--cli` to run installation in command line (CLI):
   ```sh
   ./l_openvino_toolkit_p_<version>.sh -a --cli
   ```

   @sphinxdirective
   
   .. note::
      To get additional information on all parameters that can be used, check up the help option: `--help`. Among others, you can find there `-s` option which offers silent mode, which together with `--eula approve` allows you to run whole installation with default values without any user inference.
   
   @endsphinxdirective

6. Follow the instructions on your screen. During the installation you will be asked to accept the license agreement. The acceptance is required to continue. Check out the installation process on the image below:<br>

   ![](../img/openvino-install-linux-run-boostrapper-script.gif)
   Click on the image to see the details.
   <br>
   <br>By default, the Intel® Distribution of OpenVINO™ is installed to the following directory:
   * For root or administrator: `/opt/intel/openvino_<version>/`
   * For regular users: `/home/<USER>/intel/openvino_<version>/`

   <br>For simplicity, a symbolic link to the latest installation is also created: `/opt/intel/openvino_2022/` or `/home/<USER>/intel/openvino_2022/`

To check **Release Notes** please visit: [Release Notes](https://software.intel.com/en-us/articles/OpenVINO-RelNotes)

The core components are now installed. Continue to the next section to install additional dependencies.

## <a name="install-external-dependencies"></a>Step 2: Install External Software Dependencies

These dependencies are required for:

- Intel-optimized build of OpenCV library
- Deep Learning Inference Engine
- Deep Learning Model Optimizer tools

1. Go to the `install_dependencies` directory:
   ```sh
   cd <INSTALL_DIR>/intel/openvino_2022/install_dependencies
   ```
2. Run a script to download and install the external software dependencies:
   ```sh
   sudo -E ./install_openvino_dependencies.sh
   ```
   
   Once the dependencies are installed, continue to the next section to set your environment variables.

## <a name="set-the-environment-variables"></a>Step 3: Configure the Environment

You must update several environment variables before you can compile and run OpenVINO™ applications. Set environment variables as follows:

```sh
source <INSTALL_DIR>/intel/openvino_2022/bin/setupvars.sh
```  

If you have more than one OpenVINO version on your machine, you can easily switch its version by sourcing `setupvars.sh` of your choice.

> **NOTE**: You can also run this script every time when you start new terminal session. Open `~/.bashrc` in your favorite editor, and add `source <INSTALL_DIR>/intel/openvino_2022/bin/setupvars.sh`. Next time when you open a terminal, you will see `[setupvars.sh] OpenVINO environment initialized`. Changing `.bashrc` is not recommended when you have many OpenVINO versions on your machine and want to switch among them, as each may require different setup.

The environment variables are set. Next, you can download some additional tools.

## <a name="model-optimizer">Step 4 (Optional): Download additional components

> **NOTE**: Since the OpenVINO™ 2022.1 release, the following development tools: Model Optimizer, Post-Training Optimization Tool, Model Downloader and other Open Model Zoo tools, Accuracy Checker, and Annotation Converter are not part of the installer. These tools are now only available on [pypi.org](https://pypi.org/project/openvino-dev/).

Among the developer tools you can find e.g. Model Optimizer which imports, converts, and optimizes models, and Model Downloader which gives you access to the collection of pre-trained deep learning public and intel-trained models.  

For details, visit [pypi.org](https://pypi.org/project/openvino-dev/) site.

## <a name="optional-steps"></a>Step 5 (Optional): Configure Inference on non-CPU Devices

@sphinxdirective
.. tab:: GPU

   Only if you want to enable the toolkit components to use processor graphics (GPU) on your system, follow the steps in :ref:`GPU Setup Guide <gpu guide>`.

.. tab:: NCS 2

   Only if you want to perform inference on Intel® Movidius™ NCS powered by the Intel® Movidius™ Myriad™ 2 VPU or Intel® Neural Compute Stick 2 powered by the Intel® Movidius™ Myriad™ X VPU, follow the steps on :ref:`NCS2 Setup Guide <ncs guide>`.
   For more details, see the `Get Started page for Intel® Neural Compute Stick 2 <https://software.intel.com/en-us/neural-compute-stick/get-started>`_.

.. tab:: VPU

   To install and configure your Intel® Vision Accelerator Design with Intel® Movidius™ VPUs, see the :ref:`VPUs Configuration Guide <vpu guide>`.
   After configuration is done, you are ready to run the verification scripts with the HDDL Plugin for your Intel® Vision Accelerator Design with Intel® Movidius™ VPUs. Check up our :ref:`Movidius VPU demos <vpu demos>`.

   .. warning::
      While working with either HDDL or NCS, choose one of them as they cannot run simultaneously on the same machine.

@endsphinxdirective

## <a name="get-started"></a>Step 6: What's next?

Now you are ready to try out the toolkit.

Developing in Python:
   * [Start with tensorflow models with OpenVINO](https://docs.openvino.ai/latest/notebooks/101-tensorflow-to-openvino-with-output.html)
   * [Start with ONNX and PyTorch models with OpenVINO](https://docs.openvino.ai/latest/notebooks/102-pytorch-onnx-to-openvino-with-output.html)
   * [Start with PaddlePaddle models with OpenVINO](https://docs.openvino.ai/latest/notebooks/103-paddle-onnx-to-openvino-classification-with-output.html)

Developing in C++:
    <placeholder>

## <a name="uninstall"></a>Uninstall the Intel® Distribution of OpenVINO™ Toolkit

To uninstall the toolkit, follow the steps on the [Uninstalling page](uninstalling-openvino.md).

## Troubleshooting
@sphinxdirective
.. raw:: html

   <div class="collapsible-section">

@endsphinxdirective
PRC developers might encounter pip errors during OpenVINO™ installation. To resolve the issues, try one of the following options:
* Add the download source using the `-i` parameter with the Python `pip` command. For example: 

   ```
   pip install numpy -i https://mirrors.aliyun.com/pypi/simple/
   ```
Use the `--trusted-host` parameter if the URL above is `http` instead of `https`.

* If you run into incompatibility issues between components after installing new OpenVINO version, try running `requirements.txt` with the following command:

   ```
   pip install -r <INSTALL_DIR>/intel/openvino_2022/tools/requirements.txt
   ```

@sphinxdirective
.. raw:: html

   </div>

@endsphinxdirective

## Additional Resources
@sphinxdirective
.. raw:: html

   <div class="collapsible-section">

@endsphinxdirective
   - Convert models for use with OpenVINO™: [Model Optimizer Developer Guide](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
   - Write your own OpenVINO™ applications: [Inference Engine Developer Guide](../IE_DG/Deep_Learning_Inference_Engine_DevGuide.md)
   - Information on sample applications: [Inference Engine Samples Overview](../IE_DG/Samples_Overview.md)
   - Information on a supplied set of models: [Overview of OpenVINO™ Toolkit Pre-Trained Models](../model_zoo.md)
   - IoT libraries and code samples in the GitHUB repository: [Intel® IoT Developer Kit](https://github.com/intel-iot-devkit)
   
   To learn more about converting models from specific frameworks, go to:
   
   - [Convert Your Caffe* Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_Caffe.md)
   - [Convert Your TensorFlow* Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_TensorFlow.md)
   - [Convert Your MXNet* Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_MxNet.md)
   - [Convert Your Kaldi* Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_Kaldi.md)
   - [Convert Your ONNX* Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_ONNX.md)

@sphinxdirective
.. raw:: html

   </div>

@endsphinxdirective