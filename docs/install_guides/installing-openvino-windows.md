# Install and Configure Intel® Distribution of OpenVINO™ toolkit for Windows* 10 {#openvino_docs_install_guides_installing_openvino_windows}

## Introduction

By default, the [OpenVINO™ Toolkit](https://docs.openvinotoolkit.org/latest/index.html) installation on this page installs the following components:

| Component                                                                                           | Description                                                                                                                                                                                                                                                                                                   |  
|-----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Model Optimizer](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md) | This tool imports, converts, and optimizes models that were trained in popular frameworks to a format usable by Intel tools, especially the Inference Engine. <br> Popular frameworks include Caffe\*, TensorFlow\*, MXNet\*, ONNX\* and Kaldi\*. |
| [Inference Engine](../IE_DG/Deep_Learning_Inference_Engine_DevGuide.md) | This is the engine that runs the deep learning model. It includes a set of libraries for an easy inference integration into your applications. |
| [OpenCV\*](https://docs.opencv.org/master/) | OpenCV\* community version compiled for Intel® hardware |
| [Inference Engine Code Samples](../IE_DG/Samples_Overview.md) | A set of simple command-line applications demonstrating how to utilize specific OpenVINO capabilities in an application and how to perform specific tasks, such as loading a model, running inference, querying specific device capabilities, and more. |
| [Demo Applications](@ref omz_demos) | A set of command-line applications that serve as robust templates to help you implement multi-stage pipelines and specific deep learning scenarios. |
| Additional Tools | A set of tools to work with your models including [Accuracy Checker utility](@ref omz_tools_accuracy_checker), [Post-Training Optimization Tool](@ref pot_README), [Model Downloader](@ref omz_tools_downloader) and others |
| [Documentation for Pre-Trained Models ](@ref omz_models_group_intel) | Documentation for the pre-trained models available in the [Open Model Zoo repo](https://github.com/openvinotoolkit/open_model_zoo). |

## System Requirements

**Hardware**

Optimized for these processors:
* 6th to 12th generation Intel® Core™ processors and Intel® Xeon® processors 
* 3rd generation Intel® Xeon® Scalable processor (formerly code named Cooper Lake)
* Intel® Xeon® Scalable processor (formerly Skylake and Cascade Lake)
* Intel Atom® processor with support for Intel® Streaming SIMD Extensions 4.1 (Intel® SSE4.1)
* Intel Pentium® processor N4200/5, N3350/5, or N3450/5 with Intel® HD Graphics
* Intel® Iris® Xe MAX Graphics
* Intel® Neural Compute Stick 2
* Intel® Vision Accelerator Design with Intel® Movidius™ VPUs

> **NOTE**: Processor graphics are not included in all processors. See [Product Specifications](https://ark.intel.com/) for information about your processor.

**Operating System**

- Microsoft Windows* 10, 64-bit

**Software**
- [Microsoft Visual Studio 2019 with MSBuild](https://visualstudio.microsoft.com/vs/older-downloads/#visual-studio-2019-and-other-products)
- [CMake 3.14 or higher, 64-bit](https://cmake.org/download/)
- [Python 3.6 - 3.8, 64-bit](https://www.python.org/downloads/windows/)
    > **IMPORTANT**: As part of this installation, make sure you click the option **Add Python 3.x to PATH** to [add Python](https://docs.python.org/3/using/windows.html#installation-steps) to your `PATH` environment variable.

## Overview

This guide provides step-by-step instructions on how to install the Intel® Distribution of OpenVINO™ toolkit. Links are provided for each type of compatible hardware including downloads, initialization and configuration steps. The following steps will be covered:

1. <a href="#install-external-dependencies">Install External Software Dependencies</a>
2. <a href="#install-openvino">Install the Intel® Distribution of OpenVINO™ Toolkit</a>
3. <a href="#set-the-environment-variables">Configure the Environment</a>
4. <a href="#model-optimizer">Configure the Model Optimizer</a>
5. <a href="#optional-steps">Configure Inference on non-CPU Devices (Optional):</a>
   - <a href="#install-gpu">Steps for Intel® Processor Graphics (GPU)</a>
   - <a href="#hddl-myriad">Steps for Intel® Vision Accelerator Design with Intel® Movidius™ VPU</a><br>
   After installing your Intel® Movidius™ VPU, you will return to this guide to complete OpenVINO™ installation.<br>   
5. <a href="#get-started">Start Using the Toolkit</a>

- [Steps to uninstall the Intel® Distribution of OpenVINO™ Toolkit](../install_guides/uninstalling-openvino.md)

## <a name="install-external-dependencies"></a>Step 1: Install External Software Dependencies
    
Install these dependencies:

1. [Microsoft Visual Studio* 2019 with MSBuild](https://visualstudio.microsoft.com/vs/older-downloads/#visual-studio-2019-and-other-products)
   > **NOTE**: You can choose to download Community version. Use [Microsoft Visual Studio installation guide](https://docs.microsoft.com/en-us/visualstudio/install/install-visual-studio?view=vs-2019) to walk you through the installation. During installation in the **Workloads** tab, choose **Desktop development with C++**.
   
2. [CMake 3.14 or higher 64-bit](https://cmake.org/download/)
   > **NOTE**: You can either use `cmake<version>.msi` which is the installation wizard or `cmake<version>.zip` where you have to go into the `bin` folder and then manually add the path to environmental variables.
   
3. [Python **3.6** - **3.8** 64-bit](https://www.python.org/downloads/windows/)
   
   > **IMPORTANT**: As part of this installation, make sure you click the option **Add Python 3.x to PATH** to [add Python](https://docs.python.org/3/using/windows.html#installation-steps) to your `PATH` environment variable.

## <a name="install-openvino"></a>Step 2: Install the Intel® Distribution of OpenVINO™ toolkit Core Components

1. Download the Intel® Distribution of OpenVINO™ toolkit package file from [Intel® Distribution of OpenVINO™ toolkit for Windows*](https://software.intel.com/en-us/openvino-toolkit/choose-download).
   Select the Intel® Distribution of OpenVINO™ toolkit for Windows* package from the dropdown menu.
   
2. Go to the `Downloads` folder and double-click `w_openvino_toolkit_p_<version>.exe`. A window opens to let you choose your installation directory and components. 
   ![](../img/openvino-install-windows-01.png)
   
3. Follow the instructions on your screen. Watch for informational messages such as the following in case you must complete additional steps:
   ![](../img/openvino-install-windows-02.png)

4. By default, the Intel® Distribution of OpenVINO™ is installed to the following directory, referred to as `<INSTALL_DIR>` elsewhere in the documentation: `C:\Program Files (x86)\Intel\openvino_<version>`. For simplicity, a shortcut to the latest installation is also created: `C:\Program Files (x86)\Intel\openvino_2021`.

5. **Optional**: You can choose **Customize** to change the installation directory or the components you want to install.
   > **NOTE**: If there is an OpenVINO™ toolkit version previously installed on your system, the installer will use the same destination directory for next installations. If you want to install a newer version to a different directory, you need to uninstall the previously installed versions.

6. The **Finish** screen indicates that the core components have been installed: 
   ![](../img/openvino-install-windows-03.png)
 
7. Click **Finish** to close the installation wizard.

   Once you click **Finish** to close the installation wizard, a new browser window opens with the document you’re reading now (in case you installed without it) and jumps to the section with the next installation steps.

The core components are now installed. Continue to the next section to install additional dependencies.

## <a name="set-the-environment-variables">Step 3: Configure the Environment

> **NOTE**: If you installed the Intel® Distribution of OpenVINO™ to a non-default install directory, replace `C:\Program Files (x86)\Intel` with that directory in this guide's instructions.

You must update several environment variables before you can compile and run OpenVINO™ applications. Open the Command Prompt, and run the `setupvars.bat` batch file to temporarily set your environment variables:

```sh
"C:\Program Files (x86)\Intel\openvino_2021\bin\setupvars.bat"
```

> **IMPORTANT**: Windows PowerShell* is not recommended to run the configuration commands. Please use the command prompt (cmd) instead.

**Optional**: OpenVINO toolkit environment variables are removed when you close the command prompt window. As an option, you can permanently set the environment variables manually.

> **NOTE**: If you see an error indicating Python is not installed when you know you installed it, your computer might not be able to find the program. For the instructions to add Python to your system environment variables, see <a href="#Update-Path">Update Your Windows Environment Variables</a>.

The environment variables are set. Next, you will configure the Model Optimizer.

## <a name="model-optimizer">Step 4: Configure the Model Optimizer</a>

The Model Optimizer is a Python\*-based command line tool for importing
trained models from popular deep learning frameworks such as Caffe\*,
TensorFlow\*, Apache MXNet\*, ONNX\* and Kaldi\*.

The Model Optimizer is a key component of the Intel Distribution of OpenVINO toolkit. Performing inference on a model 
(with the exception of ONNX and nGraph models) requires running the model through the Model Optimizer. When you convert a pre-trained 
model through the Model Optimizer, your output is an Intermediate Representation (IR) of the network. The Intermediate 
Representation is a pair of files that describe the whole model:

- `.xml`: Describes the network topology
- `.bin`: Contains the weights and biases binary data

For more information about the Model Optimizer, refer to the [Model Optimizer Developer Guide](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md). 

If you see error messages, make sure you installed all dependencies. These steps use a command prompt to make sure you see error messages.

1. Open a command prompt by typing `cmd` in your **Search Windows** box and then pressing **Enter**.
Type commands in the opened window:
   ![](../img/command_prompt.PNG)

2. Go to the Model Optimizer prerequisites directory.<br>
   ```sh
   cd C:\Program Files (x86)\Intel\openvino_2021\deployment_tools\model_optimizer\install_prerequisites
   ```

2.  Run this batch file to configure the Model Optimizer for Caffe, TensorFlow 2.x, MXNet, Kaldi\*, and ONNX:<br>
   ```sh
   install_prerequisites.bat
   ```

3. **Optional:** You can choose to configure each framework separately instead. If you see error messages, make sure you installed all dependencies. From the Model Optimizer prerequisites directory, run the scripts for the model frameworks you want support for. You can run more than one script.
   
   > **NOTE**: You can choose to install Model Optimizer support for only certain frameworks. In the same directory are individual scripts for Caffe, TensorFlow 1.x, TensorFlow 2.x, MXNet, Kaldi*, and ONNX (install_prerequisites_caffe.bat, etc.).
   
The Model Optimizer is configured for one or more frameworks.

You have now completed all required installation, configuration and build steps in this guide to use your CPU to work with your trained models. 

If you want to use a GPU or VPU, or update your Windows* environment variables, read through the <a href="#optional-steps">Optional Steps</a> section:

- <a href="#install-gpu">Steps for Intel® Processor Graphics (GPU)</a>
- <a href="#hddl-myriad">Steps for Intel® Vision Accelerator Design with Intel® Movidius™ VPUs</a>
- <a href="#Update-Path">Add CMake* or Python* to your Windows* environment variables</a><br>

Or proceed to the <a href="#get-started">Start Using the Toolkit</a> section to learn the basic OpenVINO™ toolkit workflow and run code samples and demo applications.

## <a name="optional-steps"></a>Step 5 (Optional): Configure Inference on non-CPU Devices:

### <a name="install-gpu"></a>Optional: Steps for Intel® Processor Graphics (GPU)

> **NOTE**: These steps are required only if you want to use an Intel® integrated GPU.

This section will help you check if you require driver installation. Install indicated version or higher.

If your applications offload computation to **Intel® Integrated Graphics**, you must have the Intel Graphics Driver for Windows installed on your hardware.
[Download and install the recommended version](https://downloadcenter.intel.com/download/30079/Intel-Graphics-Windows-10-DCH-Drivers). 

To check if you have this driver installed:

1. Type **device manager** in your **Search Windows** box and press Enter. The **Device Manager** opens.

2. Click the drop-down arrow to view the **Display adapters**. You can see the adapter that is installed in your computer:
   ![](../img/DeviceManager.PNG)

3. Right-click the adapter name and select **Properties**.

4. Click the **Driver** tab to see the driver version. 
   ![](../img/DeviceDriverVersion.PNG)

You are done updating your device driver and are ready to use your GPU. Proceed to the <a href="#get-started">Start Using the Toolkit</a> section to learn the basic OpenVINO™ toolkit workflow and run code samples and demo applications.

### <a name="hddl-myriad"></a> Optional: Steps for Intel® Vision Accelerator Design with Intel® Movidius™ VPUs

> **NOTE**: These steps are required only if you want to use Intel® Vision Accelerator Design with Intel® Movidius™ VPUs.

To enable inference on Intel® Vision Accelerator Design with Intel® Movidius™ VPUs, the following additional installation steps are required:

  1. Download and install <a href="https://www.microsoft.com/en-us/download/details.aspx?id=48145">Visual C++ Redistributable for Visual Studio 2017</a>
  2. Check with a support engineer if your Intel® Vision Accelerator Design with Intel® Movidius™ VPUs card requires SMBUS connection to PCIe slot (most unlikely). Install the SMBUS driver only if confirmed (by default, it's not required):
      1. Go to the `<INSTALL_DIR>\deployment_tools\inference-engine\external\hddl\drivers\SMBusDriver` directory, where `<INSTALL_DIR>` is the directory in which the Intel Distribution of OpenVINO toolkit is installed.
      2. Right click on the `hddlsmbus.inf` file and choose **Install** from the pop up menu.

You are done installing your device driver and are ready to use your Intel® Vision Accelerator Design with Intel® Movidius™ VPUs.

See also: 

* For advanced configuration steps for your IEI Mustang-V100-MX8 accelerator, see [Intel® Movidius™ VPUs Setup Guide for Use with Intel® Distribution of OpenVINO™ toolkit](movidius-setup-guide.md).

* After you've configurated your Intel® Vision Accelerator Design with Intel® Movidius™ VPUs, see [Intel® Movidius™ VPUs Programming Guide for Use with Intel® Distribution of OpenVINO™ toolkit](movidius-programming-guide.md) to learn how to distribute a model across all 8 VPUs to maximize performance.

After configuration is done, you are ready to go to <a href="#get-started">Start Using the Toolkit</a> section to learn the basic OpenVINO™ toolkit workflow and run code samples and demo applications.

## <a name="Update-Path"></a>Optional: Update Your Windows Environment Variables

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

## <a name="get-started"></a>Step 6: Start Using the Toolkit

Now you are ready to try out the toolkit. To continue, see the [Get Started Guide](../get_started.md) section to learn the basic OpenVINO™ toolkit workflow and run code samples and demo applications with pre-trained models on different inference devices.

## Troubleshooting

PRC developers might encounter downloading issues during <a name="model-optimizer">Step 4: Configure the Model Optimizer</a>. To resolve the issues, try one of the following options:
* Add the download source using the `-i` parameter with the Python `pip` command. For example: 

   ```
   pip install openvino-dev -i https://mirrors.aliyun.com/pypi/simple/
   ```
   Use the `--trusted-host` parameter if the URL above is `http` instead of `https`.
   You can also run the following command to install specific framework. For example:
   
   ```
   pip install openvino-dev[tensorflow2] -i https://mirrors.aliyun.com/pypi/simple/
   ```

* Modify or create the `~/.pip/pip.conf` file to change the default download source with the content below:

   ```
   [global]
   index-url = http://mirrors.aliyun.com/pypi/simple/
   [install]
   trusted-host = mirrors.aliyun.com
   ```

## <a name="uninstall"></a>Uninstall the Intel® Distribution of OpenVINO™ Toolkit

To uninstall the toolkit, follow the steps on the [Uninstalling](uninstalling_openvino.md) page.

## Additional Resources

- Get started with samples and demos: [Get Started Guide](../get_started.md)
- Intel® Distribution of OpenVINO™ toolkit home page: [https://software.intel.com/en-us/openvino-toolkit](https://software.intel.com/en-us/openvino-toolkit)
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