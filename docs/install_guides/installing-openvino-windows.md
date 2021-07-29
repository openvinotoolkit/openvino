# Install Intel® Distribution of OpenVINO™ toolkit for Windows* 10 {#openvino_docs_install_guides_installing_openvino_windows}

> **NOTES**:
> - This guide applies to Microsoft Windows\* 10 64-bit. For Linux* OS information and instructions, see the [Installation Guide for Linux](installing-openvino-linux.md).
> - [Intel® System Studio](https://software.intel.com/en-us/system-studio) is an all-in-one, cross-platform tool suite, purpose-built to simplify system bring-up and improve system and IoT device application performance on Intel® platforms. If you are using the Intel® Distribution of OpenVINO™ with Intel® System Studio, go to [Get Started with Intel® System Studio](https://software.intel.com/en-us/articles/get-started-with-openvino-and-intel-system-studio-2019).

## Introduction

> **IMPORTANT**:
> - All steps in this guide are required, unless otherwise stated.<br>
> - In addition to the download package, you must install dependencies and complete configuration steps.

Your installation is complete when these are all completed:

1. Install the <a href="#Install-Core-Components">Intel® Distribution of OpenVINO™ toolkit core components</a>

2. Install the dependencies:

   - [Microsoft Visual Studio* 2019 with MSBuild](http://visualstudio.microsoft.com/downloads/)
   - [CMake 3.14 or higher 64-bit](https://cmake.org/download/)
   - [Python **3.6** - **3.8** 64-bit](https://www.python.org/downloads/windows/)
   > **IMPORTANT**: As part of this installation, make sure you click the option **[Add Python 3.x to PATH](https://docs.python.org/3/using/windows.html#installation-steps)** to add Python to your `PATH` environment variable.

3. <a href="#set-the-environment-variables">Set Environment Variables</a>         

4. <a href="#Configure_MO">Configure the Model Optimizer</a>

5. Optional: 

    - <a href="#Install-GPU">Install the Intel® Graphics Driver for Windows*</a>

    - <a href="#hddl-myriad">Install the drivers and software for the Intel® Vision Accelerator Design with Intel® Movidius™ VPUs</a>

    - <a href="#Update-Path">Update Windows* environment variables</a> (necessary if you didn't choose the option to add Python to the path when you installed Python)

Also, the following steps will be covered in the guide:
- <a href="#get-started">Get Started with Code Samples and Demo Applications</a>
- <a href="#uninstall">Uninstall the Intel® Distribution of OpenVINO™ Toolkit</a>

### About the Intel® Distribution of OpenVINO™ toolkit

OpenVINO™ toolkit is a comprehensive toolkit for quickly developing applications and solutions that solve a variety of tasks including emulation of human vision, automatic speech recognition, natural language processing, recommendation systems, and many others. Based on latest generations of artificial neural networks, including Convolutional Neural Networks (CNNs), recurrent and attention-based networks, the toolkit extends computer vision and non-vision workloads across Intel® hardware, maximizing performance. It accelerates applications with high-performance, AI and deep learning inference deployed from edge to cloud.

For more information, see the online [Intel® Distribution of OpenVINO™ toolkit Overview](https://software.intel.com/en-us/OpenVINO-toolkit) page.

The Intel® Distribution of OpenVINO™ toolkit for Windows\* 10 OS:

- Enables CNN-based deep learning inference on the edge
- Supports heterogeneous execution across Intel® CPU, Intel® Processor Graphics (GPU), Intel® Neural Compute Stick 2, and Intel® Vision Accelerator Design with Intel® Movidius™ VPUs
- Speeds time-to-market through an easy-to-use library of computer vision functions and pre-optimized kernels
- Includes optimized calls for computer vision standards including OpenCV\* and OpenCL™

#### <a name="InstallPackageContents"></a>Included in the Installation Package

The following components are installed by default:  

| Component                                                                                          | Description                                                                                                                                                                                                                                   |
|:---------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|[Model Optimizer](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md) |This tool imports, converts, and optimizes models that were trained in popular frameworks to a format usable by Intel tools, especially the Inference Engine.<br><strong>NOTE</strong>: Popular frameworks include such frameworks as Caffe\*, TensorFlow\*, MXNet\*, and ONNX\*.         |
|[Inference Engine](../IE_DG/Deep_Learning_Inference_Engine_DevGuide.md)               |This is the engine that runs the deep learning model. It includes a set of libraries for an easy inference integration into your applications.                                                                                                 |
|[OpenCV\*](https://docs.opencv.org/master/)                                                         |OpenCV* community version compiled for Intel® hardware                                                                                                                                                                                         |
|[Inference Engine Samples](../IE_DG/Samples_Overview.md)                             |A set of simple console applications demonstrating how to use Intel's Deep Learning Inference Engine in your applications.  |
| [Demos](@ref omz_demos)                                   | A set of console applications that demonstrate how you can use the Inference Engine in your applications to solve specific use-cases  |
| Additional Tools                                   | A set of tools to work with your models including [Accuracy Checker utility](@ref omz_tools_accuracy_checker), [Post-Training Optimization Tool Guide](@ref pot_README), [Model Downloader](@ref omz_tools_downloader) and other  |
| [Documentation for Pre-Trained Models ](@ref omz_models_group_intel)                                   | Documentation for the pre-trained models available in the [Open Model Zoo repo](https://github.com/openvinotoolkit/open_model_zoo)  |

**Could Be Optionally Installed**

[Deep Learning Workbench](@ref workbench_docs_Workbench_DG_Introduction) (DL Workbench) is a platform built upon OpenVINO™ and provides a web-based graphical environment that enables you to optimize, fine-tune, analyze, visualize, and compare performance of deep learning models on various Intel® architecture
configurations. In the DL Workbench, you can use most of OpenVINO™ toolkit components:
* [Model Downloader](@ref omz_tools_downloader)
* [Intel® Open Model Zoo](@ref omz_models_group_intel)
* [Model Optimizer](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
* [Post-training Optimization Tool](@ref pot_README)
* [Accuracy Checker](@ref omz_tools_accuracy_checker)
* [Benchmark Tool](../../inference-engine/samples/benchmark_app/README.md)

Proceed to an [easy installation from Docker](@ref workbench_docs_Workbench_DG_Install_from_Docker_Hub) to get started.

### System Requirements

**Hardware**

* 6th to 11th generation Intel® Core™ processors and Intel® Xeon® processors 
* 3rd generation Intel® Xeon® Scalable processor (formerly code named Cooper Lake)
* Intel® Xeon® Scalable processor (formerly Skylake and Cascade Lake)
* Intel Atom® processor with support for Intel® Streaming SIMD Extensions 4.1 (Intel® SSE4.1)
* Intel Pentium® processor N4200/5, N3350/5, or N3450/5 with Intel® HD Graphics
* Intel® Neural Compute Stick 2
* Intel® Vision Accelerator Design with Intel® Movidius™ VPUs

> **NOTE**: With OpenVINO™ 2020.4 release, Intel® Movidius™ Neural Compute Stick is no longer supported.

**Processor Notes:**

- Processor graphics are not included in all processors. See [Processors specifications](https://ark.intel.com/#@Processors) for information about your processor.
- A chipset that supports processor graphics is required if you're using an Intel Xeon processor. See [Chipset specifications](https://ark.intel.com/#@Chipsets) for information about your chipset.

**Operating System**

- Microsoft Windows\* 10 64-bit

**Software**
- [Microsoft Visual Studio* with C++ **2019 or 2017** with MSBuild](http://visualstudio.microsoft.com/downloads/)
- [CMake **3.10 or higher** 64-bit](https://cmake.org/download/)
   > **NOTE**: If you want to use Microsoft Visual Studio 2019, you are required to install CMake 3.14.
- [Python **3.6** - **3.8** 64-bit](https://www.python.org/downloads/windows/)

## Installation Steps

### <a name="Install-Core-Components"></a>Install the Intel® Distribution of OpenVINO™ toolkit Core Components

1. If you have not downloaded the Intel® Distribution of OpenVINO™ toolkit, [download the latest version](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/download.html). By default, the file is saved to the `Downloads` directory as `w_openvino_toolkit_p_<version>.exe`.
2. Go to the `Downloads` folder and double-click `w_openvino_toolkit_p_<version>.exe`. A window opens to let you choose your installation directory and components. 
   ![](../img/openvino-install-windows-01.png)
   The default installation directory is `C:\Program Files (x86)\Intel\openvino_<version>`, for simplicity, a shortcut to the latest installation is also created: `C:\Program Files (x86)\Intel\openvino_2021`. If you choose a different installation directory, the installer will create the directory for you.
   > **NOTE**: If there is an OpenVINO™ toolkit version previously installed on your system, the installer will use the same destination directory for next installations. If you want to install a newer version to a different directory, you need to uninstall the previously installed versions.    
3. Click **Next**.
4. You are asked if you want to provide consent to gather information. Choose the option of your choice. Click **Next**.
5. If you are missing external dependencies, you will see a warning screen. Write down the dependencies you are missing. **You need to take no other action at this time**. After installing the Intel® Distribution of OpenVINO™ toolkit core components, install the missing dependencies.
The screen example below indicates you are missing two dependencies:
   ![](../img/openvino-install-windows-02.png)
6. Click **Next**.
7. When the first part of installation is complete, the final screen informs you that the core components have been installed and additional steps still required:
   ![](../img/openvino-install-windows-03.png)
8. Click **Finish** to close the installation wizard. A new browser window opens to the next section of the installation guide to set the environment variables. You are in the same document. The new window opens in case you ran the installation without first opening this installation guide. 
9. If the installation indicated you must install dependencies, install them first. If there are no missing dependencies, you can go ahead and <a href="#set-the-environment-variables">set the environment variables</a>.  

### Set the Environment Variables <a name="set-the-environment-variables"></a>

> **NOTE**: If you installed the Intel® Distribution of OpenVINO™ to the non-default install directory, replace `C:\Program Files (x86)\Intel` with the directory in which you installed the software.

You must update several environment variables before you can compile and run OpenVINO™ applications. Open the Command Prompt, and run the `setupvars.bat` batch file to temporarily set your environment variables:
```sh
"C:\Program Files (x86)\Intel\openvino_2021\bin\setupvars.bat"
```
> **IMPORTANT**: Windows PowerShell* is not recommended to run the configuration commands, please use the Command Prompt instead.

<strong>(Optional)</strong>: OpenVINO toolkit environment variables are removed when you close the Command Prompt window. As an option, you can permanently set the environment variables manually.

> **NOTE**: If you see an error indicating Python is not installed when you know you installed it, your computer might not be able to find the program. For the instructions to add Python to your system environment variables, see <a href="#Update-Path">Update Your Windows Environment Variables</a>.

The environment variables are set. Continue to the next section to configure the Model Optimizer.

## Configure the Model Optimizer <a name="Configure_MO"></a>

> **IMPORTANT**: These steps are required. You must configure the Model Optimizer for at least one framework. The Model Optimizer will fail if you do not complete the steps in this section.

The Model Optimizer is a key component of the Intel® Distribution of OpenVINO™ toolkit. You cannot do inference on your trained model without running the model through the Model Optimizer. When you run a pre-trained model through the Model Optimizer, your output is an Intermediate Representation (IR) of the network. The IR is a pair of files that describe the whole model:

- `.xml`: Describes the network topology
- `.bin`: Contains the weights and biases binary data

The Inference Engine reads, loads, and infers the IR files, using a common API across the CPU, GPU, or VPU hardware.  

The Model Optimizer is a Python*-based command line tool (`mo.py`), which is located in `C:\Program Files (x86)\Intel\openvino_2021\deployment_tools\model_optimizer`. Use this tool on models trained with popular deep learning frameworks such as Caffe\*, TensorFlow\*, MXNet\*, and ONNX\* to convert them to an optimized IR format that the Inference Engine can use.

This section explains how to use scripts to configure the Model Optimizer either for all of the supported frameworks at the same time or for individual frameworks. If you want to manually configure the Model Optimizer instead of using scripts, see the **Using Manual Configuration Process** section on the [Configuring the Model Optimizer](../MO_DG/prepare_model/Config_Model_Optimizer.md) page.

For more information about the Model Optimizer, see the [Model Optimizer Developer Guide](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).


### Model Optimizer Configuration Steps

You can configure the Model Optimizer either for all supported frameworks at once or for one framework at a time. Choose the option that best suits your needs. If you see error messages, make sure you installed all dependencies.

> **IMPORTANT**: The Internet access is required to execute the following steps successfully. If you have access to the Internet through the proxy server only, please make sure that it is configured in your environment.

> **NOTE**:
> In the steps below:
> - If you you want to use the Model Optimizer from another installed versions of Intel® Distribution of OpenVINO™ toolkit installed, replace `openvino_2021` with `openvino_<version>`, where `<version>` is the required version.
> - If you installed the Intel® Distribution of OpenVINO™ toolkit to the non-default installation directory, replace `C:\Program Files (x86)\Intel` with the directory where you installed the software.

These steps use a command prompt to make sure you see error messages.

#### Option 1: Configure the Model Optimizer for all supported frameworks at the same time:

1. Open a command prompt. To do so, type `cmd` in your **Search Windows** box and then press **Enter**.
Type commands in the opened window:

   ![](../img/command_prompt.PNG)

2. Go to the Model Optimizer prerequisites directory.<br>
```sh
cd C:\Program Files (x86)\Intel\openvino_2021\deployment_tools\model_optimizer\install_prerequisites
```

3. Run the following batch file to configure the Model Optimizer for Caffe\*, TensorFlow\* 1.x, MXNet\*, Kaldi\*, and ONNX\*:<br>
```sh
install_prerequisites.bat
```

#### Option 2: Configure the Model Optimizer for each framework separately:

1. Go to the Model Optimizer prerequisites directory:<br>
```sh
cd C:\Program Files (x86)\Intel\openvino_2021\deployment_tools\model_optimizer\install_prerequisites
```

2. Run the batch file for the framework you will use with the Model Optimizer. You can use more than one:

   * For **Caffe**:<br>
   ```sh
   install_prerequisites_caffe.bat
   ```

   * For **TensorFlow 1.x**:<br>
   ```sh
   install_prerequisites_tf.bat
   ```

   * For **TensorFlow 2.x**:<br>
   ```sh
   install_prerequisites_tf2.bat
   ```

   * For **MXNet**:<br>
   ```sh
   install_prerequisites_mxnet.bat
   ```

   * For **ONNX**:
   ```sh
   install_prerequisites_onnx.bat
   ```

   * For **Kaldi**:
   ```sh
   install_prerequisites_kaldi.bat
   ```

The Model Optimizer is configured for one or more frameworks. Success is indicated by a screen similar to this:

![](../img/Configure-MO.PNG)

You have completed all required installation, configuration and build steps in this guide to use your CPU to work with your trained models. 

If you want to use a GPU or VPU, or update your Windows* environment variables, read through the <a href="#optional-steps">Optional Steps</a> section:

- <a href="#Install-GPU">Steps for Intel® Processor Graphics (GPU)</a>
- <a href="#hddl-myriad">Steps for Intel® Vision Accelerator Design with Intel® Movidius™ VPUs</a>
- <a href="#Update-Path">Add CMake* or Python* to your Windows* environment variables</a><br>

Or proceed to the <a href="#get-started">Get Started</a> to get started with running code samples and demo applications.
## <a name="optional-steps"></a>Optional Steps

###  <a name="Install-GPU"></a>Optional: Additional Installation Steps for Intel® Processor Graphics (GPU)

> **NOTE**: These steps are required only if you want to use an Intel® integrated GPU.

If your applications offload computation to **Intel® Integrated Graphics**, you must have the Intel Graphics Driver for Windows installed for your hardware. 
[Download and install the recommended version](https://downloadcenter.intel.com/download/30079/Intel-Graphics-Windows-10-DCH-Drivers). 

To check if you have this driver installed:

1. Type **device manager** in your **Search Windows** box. The **Device Manager** opens.

2. Click the drop-down arrow to view the **Display adapters**. You see the adapter that is installed in your computer:

   ![](../img/DeviceManager.PNG)

3. Right-click the adapter name and select **Properties**.

4. Click the **Driver** tab to see the driver version. 

   ![](../img/DeviceDriverVersion.PNG)

You are done updating your device driver and are ready to use your GPU. Proceed to the <a href="#get-started">Get Started</a> to get started with running code samples and demo applications.

### <a name="hddl-myriad"></a> Optional: Additional Installation Steps for the Intel® Vision Accelerator Design with Intel® Movidius™ VPUs

> **NOTE**: These steps are required only if you want to use Intel® Vision Accelerator Design with Intel® Movidius™ VPUs.

To perform inference on Intel® Vision Accelerator Design with Intel® Movidius™ VPUs, the following additional installation steps are required:

  1. Download and install <a href="https://www.microsoft.com/en-us/download/details.aspx?id=48145">Visual C++ Redistributable for Visual Studio 2017</a>
  2. Check with a support engineer if your Intel® Vision Accelerator Design with Intel® Movidius™ VPUs card requires SMBUS connection to PCIe slot (most unlikely). Install the SMBUS driver only if confirmed (by default, it's not required):
      1. Go to the `<INSTALL_DIR>\deployment_tools\inference-engine\external\hddl\drivers\SMBusDriver` directory, where `<INSTALL_DIR>` is the directory in which the Intel Distribution of OpenVINO toolkit is installed.
      2. Right click on the `hddlsmbus.inf` file and choose **Install** from the pop up menu.

You are done installing your device driver and are ready to use your Intel® Vision Accelerator Design with Intel® Movidius™ VPUs.

See also: 

* For advanced configuration steps for your IEI Mustang-V100-MX8 accelerator, see [Intel® Movidius™ VPUs Setup Guide for Use with Intel® Distribution of OpenVINO™ toolkit](movidius-setup-guide.md).

* After you've configurated your Intel® Vision Accelerator Design with Intel® Movidius™ VPUs, see [Intel® Movidius™ VPUs Programming Guide for Use with Intel® Distribution of OpenVINO™ toolkit](movidius-programming-guide.md) to learn how to distribute a model across all 8 VPUs to maximize performance.

After configuration is done, you are ready to <a href="#get-started">Get Started</a> with running code samples and demo applications.

### <a name="Update-Path"></a>Optional: Update Your Windows Environment Variables

> **NOTE**: These steps are only required under special circumstances, such as if you forgot to check the box during the CMake\* or Python\* installation to add the application to your Windows `PATH` environment variable.

Use these steps to update your Windows `PATH` if a command you execute returns an error message stating that an application cannot be found. This might happen if you do not add CMake or Python to your `PATH` environment variable during the installation.

1. In your **Search Windows** box, type **Edit the system environment variables** and press **Enter**. A window similar to the following displays:
   ![](../img/System_Properties.PNG)

2. At the bottom of the screen, click **Environment Variables**.

3. Under **System variables**, click **Path** and then **Edit**:
   ![](../img/Environment_Variables-select_Path.PNG)

4. In the opened window, click **Browse**. A browse window opens:
   ![](../img/Add_Environment_Variable.PNG)

5. If you need to add CMake to the `PATH`, browse to the directory in which you installed CMake. The default directory is `C:\Program Files\CMake`.

6. If you need to add Python to the `PATH`, browse to the directory in which you installed Python. The default directory is `C:\Users\<USER_ID>\AppData\Local\Programs\Python\Python36\Python`. Note that the `AppData` folder is hidden by default. To view hidden files and folders, see the [Windows 10 instructions](https://support.microsoft.com/en-us/windows/view-hidden-files-and-folders-in-windows-10-97fbc472-c603-9d90-91d0-1166d1d9f4b5). 

7. Click **OK** repeatedly to close each screen.

Your `PATH` environment variable is updated. If the changes don't take effect immediately, you may need to reboot.

## <a name="get-started"></a>Get Started

Now you are ready to get started. To continue, see the following pages:
* [OpenVINO™ Toolkit Overview](../index.md)
* [Get Started Guide for Windows](../get_started/get_started_windows.md) to learn the basic OpenVINO™ toolkit workflow and run code samples and demo applications with pre-trained models on different inference devices.

## <a name="uninstall"></a>Uninstall the Intel® Distribution of OpenVINO™ Toolkit
Follow the steps below to uninstall the Intel® Distribution of OpenVINO™ Toolkit from your system:
1. Choose the **Apps & Features** option from the Windows* Settings app.
2. From the list of installed applications, select the Intel® Distribution of OpenVINO™ Toolkit and click **Uninstall**.
3. Follow the uninstallation wizard instructions.
4. When uninstallation is complete, click **Finish**. 

## <a name="Summary"></a>Summary

In this document, you installed the Intel® Distribution of OpenVINO™ toolkit and its dependencies. You also configured the Model Optimizer for one or more frameworks. After the software was installed and configured, you ran two verification scripts. You might have also installed drivers that will let you use a GPU or VPU to infer your models and run the Image Classification Sample application.

You are now ready to learn more about converting models trained with popular deep learning frameworks to the Inference Engine format, following the links below, or you can move on to running the [sample applications](../IE_DG/Samples_Overview.md).

To learn more about converting deep learning models, go to:

- [Convert Your Caffe* Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_Caffe.md)
- [Convert Your TensorFlow* Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_TensorFlow.md)
- [Convert Your MXNet* Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_MxNet.md)
- [Convert Your ONNX* Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_ONNX.md)

## Additional Resources

- [Intel Distribution of OpenVINO Toolkit home page](https://software.intel.com/en-us/openvino-toolkit)
- [OpenVINO™ Release Notes](https://software.intel.com/en-us/articles/OpenVINO-RelNotes)
- [Introduction to Inference Engine](../IE_DG/Deep_Learning_Inference_Engine_DevGuide.md)
- [Inference Engine Developer Guide](../IE_DG/Deep_Learning_Inference_Engine_DevGuide.md)
- [Model Optimizer Developer Guide](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
- [Inference Engine Samples Overview](../IE_DG/Samples_Overview.md)
- [Overview of OpenVINO™ Toolkit Pre-Trained Models](@ref omz_models_group_intel)
- [Intel® Neural Compute Stick 2 Get Started](https://software.intel.com/en-us/neural-compute-stick/get-started)


[myriad_driver]: ../img/myriad_driver.png
