# Install Intel® Distribution of OpenVINO™ toolkit for macOS* {#openvino_docs_install_guides_installing_openvino_macos}

> **NOTES**:
> - The Intel® Distribution of OpenVINO™ is supported on macOS\* 10.15.x versions.
> - An internet connection is required to follow the steps in this guide. If you have access to the Internet through the proxy server only, please make sure that it is configured in your OS environment.

## Introduction

The Intel® Distribution of OpenVINO™ toolkit quickly deploys applications and solutions that emulate human vision. Based on Convolutional Neural Networks (CNN), the toolkit extends computer vision (CV) workloads across Intel® hardware, maximizing performance.

The Intel® Distribution of OpenVINO™ toolkit for macOS* includes the Inference Engine, OpenCV* libraries and Model Optimizer tool to deploy applications for accelerated inference on Intel® CPUs and Intel® Neural Compute Stick 2.  

The Intel® Distribution of OpenVINO™ toolkit for macOS*:

- Enables CNN-based deep learning inference on the edge  
- Supports heterogeneous execution across Intel® CPU and Intel® Neural Compute Stick 2 with Intel® Movidius™ VPUs
- Speeds time-to-market via an easy-to-use library of computer vision functions and pre-optimized kernels
- Includes optimized calls for computer vision standards including OpenCV\*

**Included with the Installation**

The following components are installed by default:

| Component                                                                                           | Description                                                                                                                                                                                                                                                  |
| :-------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Model Optimizer](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md) | This tool imports, converts, and optimizes models, which were trained in popular frameworks, to a format usable by Intel tools, especially the Inference Engine. <br> Popular frameworks include Caffe*, TensorFlow*, MXNet\*, and ONNX\*. |
| [Inference Engine](../IE_DG/Deep_Learning_Inference_Engine_DevGuide.md)               | This is the engine that runs a deep learning model. It includes a set of libraries for an easy inference integration into your applications                                                                                                               |
| [OpenCV\*](https://docs.opencv.org/master/)                                                         | OpenCV\* community version compiled for Intel® hardware                                                                                                                                                                                                      |
| [Sample Applications](../IE_DG/Samples_Overview.md)                                                                                | A set of simple console applications demonstrating how to use the Inference Engine in your applications. |
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

## Development and Target Platform

The development and target platforms have the same requirements, but you can select different components during the installation, based on your intended use.

**Hardware**

> **NOTE**: The current version of the Intel® Distribution of OpenVINO™ toolkit for macOS* supports inference on Intel CPUs and Intel® Neural Compute Sticks 2 only.

* 6th to 11th generation Intel® Core™ processors and Intel® Xeon® processors 
* 3rd generation Intel® Xeon® Scalable processor (formerly code named Cooper Lake)
* Intel® Xeon® Scalable processor (formerly Skylake and Cascade Lake)
* Intel® Neural Compute Stick 2

**Software Requirements**

* CMake 3.10 or higher
	+ [Install](https://cmake.org/download/) (choose "macOS 10.13 or later")
	+ Add `/Applications/CMake.app/Contents/bin` to path (for default install) 
* Python 3.6 - 3.7
	+ [Install](https://www.python.org/downloads/mac-osx/) (choose 3.6.x or 3.7.x, not latest)
	+ Add to path
* Apple Xcode\* Command Line Tools
	+ In the terminal, run `xcode-select --install` from any directory
* (Optional) Apple Xcode\* IDE (not required for OpenVINO, but useful for development)

**Operating Systems**

- macOS\* 10.15

## Overview

This guide provides step-by-step instructions on how to install the Intel® Distribution of OpenVINO™ 2020.1 toolkit for macOS*.

The following steps will be covered:

1. <a href="#Install-Core">Install the Intel® Distribution of OpenVINO™ Toolkit</a>.
2. <a href="#set-the-environment-variables">Set the OpenVINO environment variables and (optional) Update to <code>.bash_profile</code></a>.
3. <a href="#configure-the-model-optimizer">Configure the Model Optimizer</a>.
4. <a href="#get-started">Get Started with Code Samples and Demo Applications</a>.
5. <a href="#uninstall">Uninstall the Intel® Distribution of OpenVINO™ Toolkit</a>.

## <a name="Install-Core"></a>Install the Intel® Distribution of OpenVINO™ Toolkit Core Components

If you have a previous version of the Intel® Distribution of OpenVINO™ toolkit installed, rename or delete these two directories:

- `/home/<user>/inference_engine_samples`
- `/home/<user>/openvino_models`

[Download the latest version of OpenVINO toolkit for macOS*](https://software.intel.com/en-us/openvino-toolkit/choose-download/free-download-macos) then return to this guide to proceed with the installation.

Install the OpenVINO toolkit core components:

1. Go to the directory in which you downloaded the Intel® Distribution of OpenVINO™ toolkit. This document assumes this is your `Downloads` directory. By default, the disk image file is saved as `m_openvino_toolkit_p_<version>.dmg`.

2. Double-click the `m_openvino_toolkit_p_<version>.dmg` file to mount.
The disk image is mounted to `/Volumes/m_openvino_toolkit_p_<version>` and automatically opened in a separate window.

3. Run the installation wizard application `m_openvino_toolkit_p_<version>.app`

4. On the **User Selection** screen, choose a user account for the installation:
    - Root
    - Administrator
    - Current user

    ![](../img/openvino-install-macos-01.png)

   The default installation directory path depends on the privileges you choose for the installation.

5. Click **Next** and follow the instructions on your screen.

6. If you are missing external dependencies, you will see a warning screen. Take note of any dependencies you are missing. After installing the Intel® Distribution of OpenVINO™ toolkit core components, you will need to install the missing dependencies. For example, the screen example below indicates you are missing two dependencies:
   ![](../img/openvino-install-macos-02.png)

7. Click **Next**.

8. The **Installation summary** screen shows you the default component set to install:
   ![](../img/openvino-install-macos-03.png)
   By default, the Intel® Distribution of OpenVINO™ is installed to the following directory, referred to as `<INSTALL_DIR>`:

   * For root or administrator: `/opt/intel/openvino_<version>/`
   * For regular users: `/home/<USER>/intel/openvino_<version>/`

   For simplicity, a symbolic link to the latest installation is also created: `/home/<user>/intel/openvino_2021/`.
9. If needed, click **Customize** to change the installation directory or the components you want to install:
   ![](../img/openvino-install-macos-04.png)
   > **NOTE**: If there is an OpenVINO™ toolkit version previously installed on your system, the installer will use the same destination directory for next installations. If you want to install a newer version to a different directory, you need to uninstall the previously installed versions.
10. Click **Next** to save the installation options and show the Installation summary screen.

11. On the **Installation summary** screen, click **Install** to begin the installation.

12. When the first part of installation is complete, the final screen informs you that the core components have been installed
   and additional steps still required:
   ![](../img/openvino-install-macos-05.png)

13. Click **Finish** to close the installation wizard. A new browser window opens to the next section of the Installation Guide to set the environment variables. If the installation did not indicate you must install dependencies, you can move ahead to [Set the Environment Variables](#set-the-environment-variables).  If you received a message that you were missing external software dependencies, listed under **Software Requirements** at the top of this guide, you need to install them now before continuing on to the next section.

## <a name="set-the-environment-variables"></a>Set the Environment Variables

You need to update several environment variables before you can compile and run OpenVINO™ applications. Open the macOS Terminal\* or a command-line interface shell you prefer and run the following script to temporarily set your environment variables:

   ```sh
   source /opt/intel/openvino_2021/bin/setupvars.sh
   ```  

If you didn't choose the default installation option, replace `/opt/intel/openvino_2021` with your directory.

<strong>Optional</strong>: The OpenVINO environment variables are removed when you close the shell. You can permanently set the environment variables as follows:

1. Open the `.bash_profile` file in the current user home directory:
   ```sh
   vi ~/.bash_profile
   ```
2. Press the **i** key to switch to insert mode.

3. Add this line to the end of the file:
   ```sh
   source /opt/intel/openvino_2021/bin/setupvars.sh
   ```

If you didn't choose the default installation option, replace `/opt/intel/openvino_2021` with your directory.

4. Save and close the file: press the **Esc** key, type `:wq` and press the **Enter** key.

5. To verify your change, open a new terminal. You will see `[setupvars.sh] OpenVINO environment initialized`.

The environment variables are set. Continue to the next section to configure the Model Optimizer.

## <a name="configure-the-model-optimizer"></a>Configure the Model Optimizer

The Model Optimizer is a Python\*-based command line tool for importing
trained models from popular deep learning frameworks such as Caffe\*,
TensorFlow\*, Apache MXNet\*, ONNX\* and Kaldi\*.

The Model Optimizer is a key component of the OpenVINO toolkit. You cannot perform inference on your trained model without running the model through the Model Optimizer. When you run a pre-trained model through the Model Optimizer, your output is an Intermediate Representation (IR) of the network. The IR is a pair of files that describe the whole model:

- `.xml`: Describes the network topology
- `.bin`: Contains the weights and biases binary data

The Inference Engine reads, loads, and infers the IR files, using a common API on the CPU hardware.

For more information about the Model Optimizer, see the [Model Optimizer Developer Guide](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).

### Model Optimizer Configuration Steps

You can choose to either configure the Model Optimizer for all supported frameworks at once, **OR** for one framework at a time. Choose the option that best suits your needs. If you see error messages, verify that you installed all dependencies listed under **Software Requirements** at the top of this guide.

> **NOTE**: If you installed OpenVINO to a non-default installation directory, replace `/opt/intel/` with the directory where you installed the software.

**Option 1: Configure the Model Optimizer for all supported frameworks at the same time:**

1. Go to the Model Optimizer prerequisites directory:
   ```sh
   cd /opt/intel/openvino_2021/deployment_tools/model_optimizer/install_prerequisites
   ```

2. Run the script to configure the Model Optimizer for Caffe, TensorFlow 1.x, MXNet, Kaldi\*, and ONNX:
   ```sh
   sudo ./install_prerequisites.sh
   ```

**Option 2: Configure the Model Optimizer for each framework separately:**

Configure individual frameworks separately **ONLY** if you did not select **Option 1** above.

1. Go to the Model Optimizer prerequisites directory:
   ```sh
   cd /opt/intel/openvino_2021/deployment_tools/model_optimizer/install_prerequisites
   ```

2. Run the script for your model framework. You can run more than one script:

   - For **Caffe**:
   ```sh
   sudo ./install_prerequisites_caffe.sh
   ```

   - For **TensorFlow 1.x**:
   ```sh
   sudo ./install_prerequisites_tf.sh
   ```

   - For **TensorFlow 2.x**:
   ```sh
   sudo ./install_prerequisites_tf2.sh
   ```

   - For **MXNet**:
   ```sh
   sudo ./install_prerequisites_mxnet.sh
   ```

   - For **ONNX**:
   ```sh
   sudo ./install_prerequisites_onnx.sh
   ```

   - For **Kaldi**:
   ```sh
   sudo ./install_prerequisites_kaldi.sh
   ```

The Model Optimizer is configured for one or more frameworks.

You have completed all required installation, configuration and build steps in this guide to use your CPU to work with your trained models. 

To enable inference on Intel® Neural Compute Stick 2, see the <a href="#additional-NCS2-steps">Steps for Intel® Neural Compute Stick 2</a>. 

Or proceed to the <a href="#get-started">Get Started</a> to get started with running code samples and demo applications.

## <a name="additional-NCS2-steps"></a>Steps for Intel® Neural Compute Stick 2

These steps are only required if you want to perform inference on Intel® Neural Compute Stick 2
powered by the Intel® Movidius™ Myriad™ X VPU. See also the
[Get Started page for Intel® Neural Compute Stick 2](https://software.intel.com/en-us/neural-compute-stick/get-started).

To perform inference on Intel® Neural Compute Stick 2, the `libusb` library is required. You can build it from the [source code](https://github.com/libusb/libusb) or install using the macOS package manager you prefer: [Homebrew*](https://brew.sh/), [MacPorts*](https://www.macports.org/) or other.

For example, to install the `libusb` library using Homebrew\*, use the following command:
```sh
brew install libusb
```

You've completed all required configuration steps to perform inference on your Intel® Neural Compute Stick 2. 
Proceed to the <a href="#get-started">Get Started</a> to get started with running code samples and demo applications.

## <a name="get-started"></a>Get Started

Now you are ready to get started. To continue, see the following pages:
* [OpenVINO™ Toolkit Overview](../index.md)
* [Get Started Guide for macOS](../get_started/get_started_macos.md) to learn the basic OpenVINO™ toolkit workflow and run code samples and demo applications with pre-trained models on different inference devices.

## <a name="uninstall"></a>Uninstall the Intel® Distribution of OpenVINO™ Toolkit

Follow the steps below to uninstall the Intel® Distribution of OpenVINO™ Toolkit from your system:

1. From the the installation directory (by default, `/opt/intel/openvino_2021`), locate and open `openvino_toolkit_uninstaller.app`.
2. Follow the uninstallation wizard instructions.
3. When uninstallation is complete, click **Finish**. 


## Additional Resources

- To learn more about the verification applications, see `README.txt` in `/opt/intel/openvino_2021/deployment_tools/demo/`.

- For detailed description of the pre-trained models, go to the [Overview of OpenVINO toolkit Pre-Trained Models](@ref omz_models_group_intel) page.

- More information on [sample applications](../IE_DG/Samples_Overview.md).

- [Convert Your Caffe* Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_Caffe.md)

- [Convert Your TensorFlow* Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_TensorFlow.md)

- [Convert Your MXNet* Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_MxNet.md)

- [Convert Your ONNX* Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_ONNX.md)

- [Intel Distribution of OpenVINO Toolkit home page](https://software.intel.com/en-us/openvino-toolkit)

- [Intel Distribution of OpenVINO Toolkit documentation](https://docs.openvinotoolkit.org)
