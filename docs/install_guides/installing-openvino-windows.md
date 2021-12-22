# Install and Configure Intel® Distribution of OpenVINO™ toolkit for Windows* 10 {#openvino_docs_install_guides_installing_openvino_windows}

## Introduction

The [OpenVINO™ Toolkit](https://docs.openvinotoolkit.org/latest/index.html) installation on this page installs the following components:

* [Inference Engine](../IE_DG/Deep_Learning_Inference_Engine_DevGuide.md) - This is the engine that runs the deep learning model. It includes a set of libraries for easy inference integration into your applications.
* [Inference Engine Code Samples](../IE_DG/Samples_Overview.md) - A set of simple command-line applications demonstrating how to utilize OpenVINO capabilities.

> **NOTE**: Since the OpenVINO™ 2022.1 release, the following development tools: Model Optimizer, Post-Training Optimization Tool, Model Downloader and other Open Model Zoo tools, Accuracy Checker, and Annotation Converter are not part of the installer. These tools are now only available on [pypi.org](https://pypi.org/project/openvino-dev/).

## System Requirements

@sphinxdirective
.. tab:: Operating Systems

  * Microsoft Windows* 10, 64-bit

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

.. tab:: Software

  * `Microsoft Visual Studio 2019 with MSBuild <http://visualstudio.microsoft.com/downloads/>`_
  * `CMake 3.14 or higher, 64-bit <https://cmake.org/download/>`_
  * `Python 3.6 - 3.8, 64-bit <https://www.python.org/downloads/windows/>`_
  
  .. note::
    You can choose to download Community version. Use `Microsoft Visual Studio installation guide <https://docs.microsoft.com/en-us/visualstudio/install/install-visual-studio?view=vs-2019>`_ to walk you through the installation. During installation in the **Workloads** tab, choose **Desktop development with C++**.

  .. note::
    You can either use `cmake<version>.msi` which is the installation wizard or `cmake<version>.zip` where you have to go into the `bin` folder and then manually add the path to environmental variables.
  
  .. important::
    As part of this installation, make sure you click the option **Add Python 3.x to PATH** to `add Python <https://docs.python.org/3/using/windows.html#installation-steps>`_ to your `PATH` environment variable.

@endsphinxdirective

## Overview

This guide provides step-by-step instructions on how to install the Intel® Distribution of OpenVINO™ toolkit. Links are provided for each type of compatible hardware including downloads, initialization and configuration steps. The following steps will be covered:

1. <a href="#install-openvino">Install the Intel® Distribution of OpenVINO™ Toolkit</a>
2. <a href="#set-the-environment-variables">Configure the Environment</a>
3. <a href="#model-optimizer">Download additional components (Optional)</a>
4. <a href="#optional-steps">Configure Inference on non-CPU Devices (Optional)</a>  
5. <a href="#get-started">What's next?</a>

## <a name="install-openvino"></a>Step 1: Install the Intel® Distribution of OpenVINO™ toolkit Core Components

1. Download the Intel® Distribution of OpenVINO™ toolkit package file from [Intel® Distribution of OpenVINO™ toolkit for Windows*](https://software.intel.com/en-us/openvino-toolkit/choose-download).
   Select the Intel® Distribution of OpenVINO™ toolkit for Windows* package from the dropdown menu.
   
2. Go to the `Downloads` folder and double-click `w_openvino_toolkit_p_<version>.exe`. A window opens to let you choose your installation directory and components. The directory will be referred to as <INSTALL_DIR> elsewhere in the documentation. Once the files extracted, you should see the following dialog box open up:

   @sphinxdirective
   
   .. image:: _static/images/openvino-install-win-installer-1.png
     :width: 400px
     :align: center
   
   @endsphinxdirective
   
3. Follow the instructions on your screen. During the installation you will be asked to accept the license agreement. The acceptance is required to continue. Check out the installation process on the image below:<br>
   ![](../img/openvino-install-win-run-boostrapper-script-2.gif)
   Click on the image to see the details.

The core components are now installed. Continue to the next section to install additional dependencies.

## <a name="set-the-environment-variables">Step 2: Configure the Environment

> **NOTE**: If you installed the Intel® Distribution of OpenVINO™ to a non-default install directory, replace `C:\Program Files (x86)\Intel` with that directory in this guide's instructions.

You must update several environment variables before you can compile and run OpenVINO™ applications. Open the Command Prompt, and run the `setupvars.bat` batch file to temporarily set your environment variables:

```sh
"<INSTALL DIR>\openvino_2022\bin\setupvars.bat"
```

> **IMPORTANT**: Windows PowerShell* is not recommended to run the configuration commands. Please use the command prompt (cmd) instead.

**Optional**: OpenVINO toolkit environment variables are removed when you close the command prompt window. As an option, you can permanently set the environment variables manually.

> **NOTE**: If you see an error indicating Python is not installed when you know you installed it, your computer might not be able to find the program. For the instructions to add Python to your system environment variables, see <a href="#Update-Path">Update Your Windows Environment Variables</a>.

The environment variables are set. Next, you can download some additional tools.

## <a name="model-optimizer">Step 3 (Optional): Download additional components

> **NOTE**: Since the OpenVINO™ 2022.1 release, the following development tools: Model Optimizer, Post-Training Optimization Tool, Model Downloader and other Open Model Zoo tools, Accuracy Checker, and Annotation Converter are not part of the installer. These tools are now only available on [pypi.org](https://pypi.org/project/openvino-dev/).

Among the developer tools you can find e.g. Model Optimizer which imports, converts, and optimizes models, and Model Downloader which gives you access to the collection of pre-trained deep learning public and intel-trained models.  

For details, visit [pypi.org](https://pypi.org/project/openvino-dev/) site.

## <a name="optional-steps"></a>Step 4 (Optional): Configure Inference on non-CPU Devices

@sphinxdirective
.. tab:: GPU

   Only if you want to enable the toolkit components to use processor graphics (GPU) on your system, follow the steps in :ref:`GPU Setup Guide <gpu guide windows>`.

.. tab:: VPU

   To install and configure your Intel® Vision Accelerator Design with Intel® Movidius™ VPUs, see the :ref:`VPUs Configuration Guide <vpu guide windows>`.

@endsphinxdirective

## <a name="get-started"></a>Step 5: What's next?

Now you are ready to try out the toolkit.

Developing in Python:
   * [Start with tensorflow models with OpenVINO](https://docs.openvino.ai/latest/notebooks/101-tensorflow-to-openvino-with-output.html)
   * [Start with ONNX and PyTorch models with OpenVINO](https://docs.openvino.ai/latest/notebooks/102-pytorch-onnx-to-openvino-with-output.html)
   * [Start with PaddlePaddle models with OpenVINO](https://docs.openvino.ai/latest/notebooks/103-paddle-onnx-to-openvino-classification-with-output.html)

Developing in C++:
    <placeholder>
## <a name="uninstall"></a>Uninstall the Intel® Distribution of OpenVINO™ Toolkit

To uninstall the toolkit, follow the steps on the [Uninstalling page](uninstalling-openvino.md).

## <a name="Update-Path"></a>(Optional) Update Your Windows Environment Variables
@sphinxdirective
.. raw:: html

   <div class="collapsible-section">

@endsphinxdirective
> **NOTE**: These steps are only required under special circumstances, such as if you forgot to check the box during the CMake\* or Python\* installation to add the application to your Windows `PATH` environment variable.

Use these steps to update your Windows `PATH` if a command you execute returns an error message stating that an application cannot be found.

1. In your **Search Windows** box, type **Edit the system environment variables** and press **Enter**. A window like the following appears:
   ![](../img/System_Properties.PNG)

2. At the bottom of the screen, click **Environment Variables**.

3. Under **System variables**, click **Path** and then **Edit**:
   ![](../img/Environment_Variables-select_Path.PNG)

4. In the opened window, click **Browse**. A browse window opens:
   ![](../img/Add_Environment_Variable.PNG)

5. If you need to add CMake to the `PATH`, browse to the directory in which you installed CMake. The default directory is `C:\Program Files\CMake`.

6. If you need to add Python to the `PATH`, browse to the directory in which you installed Python. The default directory is `C:\Users\<USER_ID>\AppData\Local\Programs\Python\Python36\Python`. Note that the `AppData` folder is hidden by default. To view hidden files and folders, see these [Windows 10 instructions](https://support.microsoft.com/en-us/windows/view-hidden-files-and-folders-in-windows-10-97fbc472-c603-9d90-91d0-1166d1d9f4b5). 

7. Click **OK** repeatedly to close each screen.

Your `PATH` environment variable is updated. If the changes don't take effect immediately, you may need to reboot.
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
- Information on a supplied set of models: [Overview of OpenVINO™ Toolkit Pre-Trained Models](@ref omz_models_group_intel)
- IoT libraries and code samples: [Intel® IoT Developer Kit](https://github.com/intel-iot-devkit)

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