# Install and Configure Intel® Distribution of OpenVINO™ toolkit for macOS* {#openvino_docs_install_guides_installing_openvino_macos}

> **NOTE**: The Intel® Distribution of OpenVINO™ is supported on macOS\* version 10.15.x with Intel® processor-based machines.

## Introduction

By default, the [OpenVINO™ Toolkit](https://docs.openvino.ai/latest/index.html) installation on this page installs the following components:

| Component                                                                                           | Description                                                                                                                                                                                                                                                                                                   |
|-----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Model Optimizer](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md) | This tool imports, converts, and optimizes models that were trained in popular frameworks to a format usable by Intel tools, especially the Inference Engine. <br> Popular frameworks include Caffe*, TensorFlow*, MXNet\*, Kaldi\* and ONNX\*. |
| [Inference Engine](../OV_Runtime_UG/Deep_Learning_Inference_Engine_DevGuide.md) | This is the engine that runs the deep learning model. It includes a set of libraries for an easy inference integration into your applications |
| [OpenCV\*](https://docs.opencv.org/master/) | OpenCV\* community version compiled for Intel® hardware |
| [Inference Engine Code Samples](../OV_Runtime_UG/Samples_Overview.md) | A set of simple command-line applications demonstrating how to utilize specific OpenVINO capabilities in an application and how to perform specific tasks, such as loading a model, running inference, querying specific device capabilities, and more. |
| [Demo Applications](@ref omz_demos) | A set of command-line applications that serve as robust templates to help you implement multi-stage pipelines and specific deep learning scenarios. |
| Additional Tools | A set of tools to work with your models, including [Accuracy Checker utility](@ref omz_tools_accuracy_checker), [Post-Training Optimization Tool](@ref pot_README), [Model Downloader](@ref omz_tools_downloader), and others |
| [Documentation for Pre-Trained Models ](@ref omz_models_group_intel) | Documentation for the pre-trained models available in the [Open Model Zoo repo](https://github.com/openvinotoolkit/open_model_zoo). |

## System Requirements

**Hardware**

> **NOTE**: The current version of the Intel® Distribution of OpenVINO™ toolkit for macOS* supports inference on Intel CPUs and Intel® Neural Compute Stick 2 devices only.

* 6th to 12th generation Intel® Core™ processors and Intel® Xeon® processors 
* 3rd generation Intel® Xeon® Scalable processor (formerly code named Cooper Lake)
* Intel® Xeon® Scalable processor (formerly Skylake and Cascade Lake)
* Intel® Neural Compute Stick 2

**Software Requirements**

* CMake 3.13 or higher
	+ [Install](https://cmake.org/download/) (choose "macOS 10.13 or later")
	+ Add `/Applications/CMake.app/Contents/bin` to path (for default install) 
* Python 3.6 - 3.8
	+ [Install](https://www.python.org/downloads/mac-osx/) (choose 3.6.x - 3.8.x, not latest)
	+ Add to path
* Apple Xcode\* Command Line Tools
	+ In the terminal, run `xcode-select --install` from any directory
* (Optional) Apple Xcode\* IDE (not required for OpenVINO, but useful for development)

**Operating Systems**

- macOS\* 10.15

## Overview

This guide provides step-by-step instructions on how to install the Intel® Distribution of OpenVINO™ toolkit for macOS*. The following steps will be covered:

1. <a href="#install-core">Install the Intel® Distribution of OpenVINO™ Toolkit</a>
2. <a href="#set-the-environment-variables">Configure the Environment</a>
3. <a href="#model-optimizer">Configure the Model Optimizer</a>
4. <a href="#configure-ncs2">Configure the Intel® Neural Compute Stick 2 (Optional)</a>
5. <a href="#get-started">Start Using the Toolkit</a>

- [Steps to uninstall the Intel® Distribution of OpenVINO™ Toolkit](../install_guides/uninstalling-openvino.md)

## <a name="install-core"></a>Step 1: Install the Intel® Distribution of OpenVINO™ Toolkit Core Components

If you have a previous version of the Intel® Distribution of OpenVINO™ toolkit installed, rename or delete these two directories:

- `~/inference_engine_samples`
- `~/openvino_models`

[Download the latest version of OpenVINO toolkit for macOS*](https://software.intel.com/en-us/openvino-toolkit/choose-download/free-download-macos), then return to this guide to proceed with the installation.

Install the OpenVINO toolkit core components:

1. Go to the directory where you downloaded the Intel® Distribution of OpenVINO™ toolkit. This document assumes this is your `Downloads` directory. By default, the disk image file is saved as `m_openvino_toolkit_p_<version>.dmg`.

2. Double-click the `m_openvino_toolkit_p_<version>.dmg` file to mount.
The disk image is mounted to `/Volumes/m_openvino_toolkit_p_<version>` and automatically opens in a separate window.

3. Run the installation wizard application `m_openvino_toolkit_p_<version>.app`.

4. On the **User Selection** screen, choose a user account for the installation:
    - Root
    - Administrator
    - Current user

   The default installation directory path depends on the privileges you choose for the installation.

5. Follow the instructions on your screen. Watch for informational messages such as the following in case you must complete additional steps:
   ![](../img/openvino-install-macos-02.png)

6. The **Installation summary** screen shows you the default component set to install. By default, the Intel® Distribution of OpenVINO™ is installed in the following directory, referred to as `<INSTALL_DIR>` elsewhere in the documentation:

   * For root or administrator: `/opt/intel/openvino_<version>/`
   * For regular users: `~/intel/openvino_<version>/`

   For simplicity, a symbolic link to the latest installation is also created: `<INSTALL_DIR>/intel/openvino_2021/`.

7. **Optional**: You can choose **Customize** to change the installation directory or the components you want to install.
> **NOTE**: If there is an OpenVINO™ toolkit version previously installed on your system, the installer will use the same destination directory for next installations. If you want to install a newer version to a different directory, you need to uninstall the previously installed versions.

8. The **Finish** screen indicates that the core components have been installed:
   ![](../img/openvino-install-macos-05.png)

   Once you click **Finish** to close the installation wizard, a new browser window opens with the document you’re reading now (in case you installed without it) and jumps to the section with the next installation steps.

The core components are now installed. If you received a message that you were missing external software dependencies (list available under Software Requirements at the top of this guide), you need to install them now before continuing to the next section.

## <a name="set-the-environment-variables"></a>Step 2: Configure the Environment

> **NOTE**: If you installed OpenVINO to a non-default installation directory, replace `/opt/intel/` with the directory where you installed the software.

You must update several environment variables before you can compile and run OpenVINO™ applications. Set persistent environment variables as follows, using vi (as below) or your preferred editor in the terminal:

1. Open the `.bash_profile` file in the current user home directory:
   ```sh
   vi ~/.bash_profile
   ```

2. Press the **i** key to switch to insert mode.

3. Add this line to the end of the file:
   ```sh
   source /opt/intel/openvino_2022/setupvars.sh
   ```

   If you didn't choose the default installation option, replace `/opt/intel/openvino_2022` with your directory.

4. Save and close the file: press the **Esc** key, type `:wq` and press the **Enter** key.

5. To verify the change, open a new terminal. You will see `[setupvars.sh] OpenVINO environment initialized`.

   **Optional:** If you don't want to change your shell profile, you can run the following script to temporarily set your environment variables when working with the OpenVINO* toolkit:

   ```sh
   source /opt/intel/openvino_2021/bin/setupvars.sh
   ```  

The environment variables are set. Continue to the next section to configure the Model Optimizer.

## <a name="model-optimizer"></a>Step 3: Configure the Model Optimizer

The Model Optimizer is a Python\*-based command line tool for importing
trained models from popular deep learning frameworks such as Caffe\*,
TensorFlow\*, Apache MXNet\*, ONNX\* and Kaldi\*.

The Model Optimizer is a key component of the OpenVINO toolkit. You cannot perform inference on your trained model without running the model through the Model Optimizer. When you run a pre-trained model through the Model Optimizer, your output is an Intermediate Representation (IR) of the network. The IR is a pair of files that describe the whole model:

- `.xml`: Describes the network topology
- `.bin`: Contains the weights and biases binary data

The Inference Engine reads, loads, and infers the IR files, using a common API on the CPU hardware.

For more information about the Model Optimizer, see the [Model Optimizer Developer Guide](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).

If you see error messages, verify that you installed all dependencies listed under **Software Requirements** at the top of this guide.

1. Go to the Model Optimizer prerequisites directory:
   ```sh
   cd /opt/intel/openvino_2022/tools/model_optimizer/install_prerequisites
   ```

2. Run the script to configure the Model Optimizer for Caffe, TensorFlow, MXNet, Kaldi\*, and ONNX:
   ```sh
   sudo ./install_prerequisites.sh
   ```

3. **Optional:** You can choose to install Model Optimizer support for only certain frameworks. In the same directory are individual scripts for Caffe, TensorFlow, MXNet, Kaldi, and ONNX (`install_prerequisites_caffe.sh`, etc.). If you see error messages, make sure you installed all dependencies.

The Model Optimizer is configured for one or more frameworks.

You have completed all required installation, configuration and build steps in this guide to use your CPU to work with your trained models. 

To enable inference on Intel® Neural Compute Stick 2, see the <a href="#configure-ncs2">Configure the Intel® Neural Compute Stick 2 </a>. 

Or proceed to the <a href="#get-started">Start Using the Toolkit</a> section to learn the basic OpenVINO™ toolkit workflow and run code samples and demo applications.

## <a name="configure-ncs2"></a>Step 4 (Optional): Configure the Intel® Neural Compute Stick 2 

These steps are required only if you want to perform inference on Intel® Neural Compute Stick 2
powered by the Intel® Movidius™ Myriad™ X VPU. For more details, see also the
[Get Started page for Intel® Neural Compute Stick 2](https://software.intel.com/en-us/neural-compute-stick/get-started).

To perform inference on Intel® Neural Compute Stick 2, the `libusb` library is required. You can build it from the [source code](https://github.com/libusb/libusb) or install using the macOS package manager you prefer: [Homebrew*](https://brew.sh/), [MacPorts*](https://www.macports.org/) or other.

For example, to install the `libusb` library using Homebrew\*, use the following command:
```sh
brew install libusb
```

You've completed all required configuration steps to perform inference on your Intel® Neural Compute Stick 2.
Proceed to the <a href="#get-started">Start Using the Toolkit</a> section to learn the basic OpenVINO™ toolkit workflow and run code samples and demo applications.

## <a name="get-started"></a>Step 5: Start Using the Toolkit

Now you are ready to try out the toolkit. To continue, see the [Get Started Guide](../get_started.md) section to learn the basic OpenVINO™ toolkit workflow and run code samples and demo applications with pre-trained models on different inference devices.

## <a name="uninstall"></a>Uninstall the Intel® Distribution of OpenVINO™ Toolkit

To uninstall, follow the steps on the [Uninstalling page](uninstalling-openvino.md).

## Additional Resources

- Get started with samples and demos: [Get Started Guide](../get_started.md)
- Intel® Distribution of OpenVINO™ toolkit home page: [https://software.intel.com/en-us/openvino-toolkit](https://software.intel.com/en-us/openvino-toolkit)
- Convert models for use with OpenVINO™: [Model Optimizer Developer Guide](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
- Write your own OpenVINO™ applications: [Inference Engine Developer Guide](../OV_Runtime_UG/Deep_Learning_Inference_Engine_DevGuide.md)
- Information on sample applications: [Inference Engine Samples Overview](../OV_Runtime_UG/Samples_Overview.md)
- Information on a supplied set of models: [Overview of OpenVINO™ Toolkit Pre-Trained Models](@ref omz_models_group_intel)
- IoT libraries and code samples: [Intel® IoT Developer Kit](https://github.com/intel-iot-devkit)

To learn more about converting models from specific frameworks, go to:

- [Convert Your Caffe* Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_Caffe.md)
- [Convert Your TensorFlow* Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_TensorFlow.md)
- [Convert Your MXNet* Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_MxNet.md)
- [Convert Your Kaldi* Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_Kaldi.md)
- [Convert Your ONNX* Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_ONNX.md)
