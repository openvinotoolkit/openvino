# Install and Configure Intel® Distribution of OpenVINO™ toolkit for Linux* {#openvino_docs_install_guides_installing_openvino_linux}

> **NOTE**: These steps apply to Ubuntu\* and with some modifications shown below, also Red Hat\* Enterprise Linux\*.

## Introduction

By default, the [OpenVINO™ Toolkit](https://docs.openvinotoolkit.org/latest/index.html) installation on this page installs the following components:

| Component                                                                                           | Description                                                                                                                                                                                                                                                                                                   |  
|-----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Model Optimizer](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md) | This tool imports, converts, and optimizes models that were trained in popular frameworks to a format usable by Intel tools, especially the Inference Engine. <br> Popular frameworks include Caffe\*, TensorFlow\*, MXNet\*, and ONNX\*. |
| [Inference Engine](../IE_DG/Deep_Learning_Inference_Engine_DevGuide.md) | This is the engine that runs the deep learning model. It includes a set of libraries for an easy inference integration into your applications. |
| Intel® Media SDK | Offers access to hardware accelerated video codecs and frame processing |
| [OpenCV\*](https://docs.opencv.org/master/) | OpenCV\* community version compiled for Intel® hardware |
| [Inference Engine Code Samples](../IE_DG/Samples_Overview.md) | A set of simple command-line applications demonstrating how to utilize specific OpenVINO capabilities in an application and how to perform specific tasks, such as loading a model, running inference, querying specific device capabilities, and more. |
| [Demo Applications](@ref omz_demos) | A set of command-line applications that serve as robust templates to help you implement multi-stage pipelines and specific deep learning scenarios. |
| Additional Tools | A set of tools to work with your models including [Accuracy Checker utility](@ref omz_tools_accuracy_checker), [Post-Training Optimization Tool](@ref pot_README), [Model Downloader](@ref omz_tools_downloader) and others |
| [Documentation for Pre-Trained Models ](@ref omz_models_group_intel) | Documentation for the pre-trained models available in the [Open Model Zoo repo](https://github.com/openvinotoolkit/open_model_zoo). |
| Deep Learning Streamer (DL Streamer) | Streaming analytics framework, based on GStreamer, for constructing graphs of media analytics components. For the DL Streamer documentation, see [DL Streamer Samples](@ref gst_samples_README), [API Reference](https://openvinotoolkit.github.io/dlstreamer_gst/), [Elements](https://github.com/openvinotoolkit/dlstreamer_gst/wiki/Elements), [Tutorial](https://github.com/openvinotoolkit/dlstreamer_gst/wiki/DL-Streamer-Tutorial). |

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

> **NOTE**: Since the OpenVINO™ 2020.4 release, Intel® Movidius™ Neural Compute Stick is not supported.

**Processor Notes**

- Processor graphics are not included in all processors. See [Product Specifications](https://ark.intel.com/) for information about your processor.

**Operating Systems**

- Ubuntu 18.04.x long-term support (LTS), 64-bit
- Ubuntu 20.04.0 long-term support (LTS), 64-bit
- CentOS 7.6, 64-bit (for deployment only, not development)
- For deployment on Red Hat* Enterprise Linux* 8.2, 64-bit, you can use the Intel® Distribution of OpenVINO™ toolkit runtime package that includes the Inference Engine core libraries, nGraph, OpenCV, Python bindings, and CPU and GPU plugins. The package is available as: 
   - [Downloadable archive](https://storage.openvinotoolkit.org/repositories/openvino/packages/2021.4.1/l_openvino_toolkit_runtime_rhel8_p_2021.4.689.tgz)
   - [PyPi package](https://pypi.org/project/openvino/)
   - [Docker image](https://catalog.redhat.com/software/containers/intel/openvino-runtime/606ff4d7ecb5241699188fb3)

**Software**

  * `CMake 3.13 or higher, 64-bit <https://cmake.org/download/>`_
  * GCC 4.8 (for CentOS 7.6), GCC 7.5.0 (for Ubuntu 18.04) or GCC 9.3.0 (for Ubuntu 20.04)
  * `Python 3.6 - 3.9, 64-bit <https://www.python.org/downloads/windows/>`_

## Overview

This guide provides step-by-step instructions on how to install the Intel® Distribution of OpenVINO™ toolkit. Links are provided for each type of compatible hardware including downloads, initialization and configuration steps. The following steps will be covered:

1. <a href="#install-openvino">Install the Intel® Distribution of OpenVINO™ Toolkit</a>
2. <a href="#install-external-dependencies">Install External Software Dependencies</a>
3. <a href="#set-the-environment-variables">Configure the Environment</a>
4. <a href="#model-optimizer">Configure the Model Optimizer</a>
5. <a href="#optional-steps">Configure Inference on non-CPU Devices (Optional):</a>
   - <a href="#install-gpu">Steps for Intel® Processor Graphics (GPU)</a>
   - <a href="#install-ncs2">Steps for Intel® Neural Compute Stick 2</a>
   - <a href="#install-vpu">Steps for Intel® Vision Accelerator Design with Intel® Movidius™ VPU</a><br>
   After installing your Intel® Movidius™ VPU, you will return to this guide to complete OpenVINO™ installation.  
6. <a href="#get-started">Start Using the Toolkit</a>

- [Steps to uninstall the Intel® Distribution of OpenVINO™ Toolkit](uninstalling-openvino.md)

## <a name="install-openvino"></a>Step 1: Install the Intel® Distribution of OpenVINO™ Toolkit Core Components

1. Download the Intel® Distribution of OpenVINO™ toolkit package file from [Intel® Distribution of OpenVINO™ toolkit for Linux*](https://software.intel.com/en-us/openvino-toolkit/choose-download).
   Select the Intel® Distribution of OpenVINO™ toolkit for Linux package from the dropdown menu.

2. Open a command prompt terminal window. You can use the keyboard shortcut: Ctrl+Alt+T
3. Change directories to where you downloaded the Intel Distribution of
OpenVINO toolkit for Linux\* package file.<br>
   If you downloaded the package file to the current user's `Downloads` directory:
   ```sh
   cd ~/Downloads/
   ```
   By default, the file is saved as `l_openvino_toolkit_p_<version>.tgz`, e.g., `l_openvino_toolkit_p_2021.4.689.tgz`.
4. Unpack the .tgz file:
   ```sh
   tar -xvzf l_openvino_toolkit_p_<version>.tgz
   ```
   The files are unpacked to the `l_openvino_toolkit_p_<version>` directory.
5. Go to the `l_openvino_toolkit_p_<version>` directory:
   ```sh
   cd l_openvino_toolkit_p_<version>
   ```
   If you have a previous version of the Intel Distribution of OpenVINO
toolkit installed, rename or delete these two directories:
   - `~/inference_engine_samples_build`
   - `~/openvino_models`

6. Choose your installation option and run the related script as root to use either a graphical user interface (GUI) installation wizard or command line instructions (CLI).<br>    
   Screenshots are provided for the GUI, but not for CLI. The following information also applies to CLI and will be helpful to your installation, where you will be presented with the same choices and tasks.
   - **Option 1:** GUI Installation Wizard:
   ```sh
   sudo ./install_GUI.sh
   ```
   - **Option 2:** Command Line Instructions:
   ```sh
   sudo ./install.sh
   ```
   - **Option 3:** Command Line Silent Instructions:
   ```sh
   sudo sed -i 's/decline/accept/g' silent.cfg
   sudo ./install.sh -s silent.cfg
   ```   
   In Option 3 you can select which OpenVINO components will be installed by modifying the `COMPONENTS` parameter in the `silent.cfg` file. For example, to install only CPU runtime for the Inference Engine, set `COMPONENTS=intel-openvino-ie-rt-cpu__x86_64` in `silent.cfg`. To get a full list of available components for installation, run the `./install.sh --list_components` command from the unpacked OpenVINO™ toolkit package.

7. Follow the instructions on your screen. Watch for informational messages such as the following in case you must complete additional steps:
   ![](../img/openvino-install-linux-01.png)

   By default, the Intel® Distribution of OpenVINO™ is installed to the following directory:
   * For root or administrator: `/opt/intel/openvino_<version>/`
   * For regular users: `/home/<USER>/intel/openvino_<version>/`

   For simplicity, a symbolic link to the latest installation is also created: `/opt/intel/openvino_2021/` or `/home/<USER>/intel/openvino_2021/`

8. **Optional**: You can choose **Customize** to change the installation directory or the components you want to install.
   > **NOTE**: If there is an OpenVINO™ toolkit version previously installed on your system, the installer will use the same destination directory for the next installation. If you want to install a newer version to a different directory, you need to uninstall the previously installed versions.
   
   > **NOTE**: The Intel® Media SDK component is always installed in the `/opt/intel/mediasdk` directory regardless of the OpenVINO installation path chosen.

9. The **Finish** screen indicates that the core components have been installed:
   ![](../img/openvino-install-linux-04.png)

   Once you click **Finish** to close the installation wizard, a new browser window will open with this documentation. It jumps to the section with your next installation steps.

The core components are now installed. Continue to the next section to install additional dependencies.

## <a name="install-external-dependencies"></a>Step 2: Install External Software Dependencies

> **NOTE**: If you installed the Intel® Distribution of OpenVINO™ to a non-default directory, replace `/opt/intel` with the directory in which you installed the software.

These dependencies are required for:

- Intel-optimized build of OpenCV library
- Deep Learning Inference Engine
- Deep Learning Model Optimizer tools

1. Go to the `install_dependencies` directory:
   ```sh
   cd /opt/intel/openvino_2021/install_dependencies
   ```
2. Run a script to download and install the external software dependencies:
   ```sh
   sudo -E ./install_openvino_dependencies.sh
   ```
   
   Once the dependencies are installed, continue to the next section to set your environment variables.

## <a name="set-the-environment-variables"></a>Step 3: Configure the Environment

You must update several environment variables before you can compile and run OpenVINO™ applications. Set persistent environment variables as follows, using vi (as below) or your preferred editor:

1. Open the `.bashrc` file in `/home/<USER>`:
   ```sh
   vi ~/.bashrc
   ```

2. Press the **i** key to switch to insert mode.

3. Add this line to the end of the file:
   ```sh
   source /opt/intel/openvino_2021/bin/setupvars.sh
   ```

4. Save and close the file: press the **Esc** key and type `:wq`.

5. To verify the change, open a new terminal. You will see `[setupvars.sh] OpenVINO environment initialized`.

   **Optional:** If you don't want to change your shell profile, you can run the following script to temporarily set your environment variables for each terminal instance when working with OpenVINO™:

   ```sh
   source /opt/intel/openvino_2021/bin/setupvars.sh
   ```  

The environment variables are set. Next, you will configure the Model Optimizer.

## <a name="model-optimizer">Step 4: Configure the Model Optimizer

> **NOTE**: Since the TensorFlow framework is not officially supported on CentOS*, the Model Optimizer for TensorFlow can't be configured and run on that operating system.  

The Model Optimizer is a Python\*-based command line tool for importing
trained models from popular deep learning frameworks such as Caffe\*,
TensorFlow\*, Apache MXNet\*, ONNX\* and Kaldi\*.

The Model Optimizer is a key component of the Intel Distribution of OpenVINO toolkit. Performing inference on a model 
(with the exception of ONNX and nGraph models) requires running the model through the Model Optimizer. When you run a pre-trained 
model through the Model Optimizer, your output is an Intermediate Representation (IR) of the network. The Intermediate 
Representation is a pair of files that describe the whole model:

- `.xml`: Describes the network topology
- `.bin`: Contains the weights and biases binary data

For more information about the Model Optimizer, refer to the [Model Optimizer Developer Guide](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md). 

1. Go to the Model Optimizer prerequisites directory:
   ```sh
   cd /opt/intel/openvino_2021/deployment_tools/model_optimizer/install_prerequisites
   ```

2. Run the script to configure the Model Optimizer for Caffe, TensorFlow 2.x, MXNet, Kaldi, and ONNX:
   ```sh
   sudo ./install_prerequisites.sh
   ```

3. **Optional:** You can choose to configure each framework separately instead. If you see error messages, make sure you installed all dependencies. From the Model Optimizer prerequisites directory, run the scripts for the model frameworks you want support for. You can run more than one script.
   
   > **NOTE**: You can choose to install Model Optimizer support for only certain frameworks. In the same directory are individual scripts for Caffe, TensorFlow 1.x, TensorFlow 2.x, MXNet, Kaldi, and ONNX (install_prerequisites_caffe.sh, etc.).
   
The Model Optimizer is configured for one or more frameworks.

You have now completed all required installation, configuration, and build steps in this guide to use your CPU to work with your trained models. 

To enable inference on other hardware, see below:
- <a href="#install-gpu">Steps for Intel® Processor Graphics (GPU)</a>
- <a href="#install-ncs2">Steps for Intel® Neural Compute Stick 2</a>
- <a href="#install-vpu">Steps for Intel® Vision Accelerator Design with Intel® Movidius™ VPUs</a><br>

Or proceed to the <a href="#get-started">Start Using the Toolkit</a> section to learn the basic OpenVINO™ toolkit workflow and run code samples and demo applications.

## <a name="optional-steps"></a>Step 5 (Optional): Configure Inference on non-CPU Devices:

### <a name="install-gpu"></a>Optional: Steps for Intel® Processor Graphics (GPU)

The steps in this section are required only if you want to enable the toolkit components to use processor graphics (GPU) on your system.

1. Go to the install_dependencies directory:
   ```sh
   cd /opt/intel/openvino_2021/install_dependencies/
   ```

2. Install the **Intel® Graphics Compute Runtime for OpenCL™** driver components required to use the GPU plugin and write custom layers for Intel® Integrated Graphics. The drivers are not included in the package. To install, run this script:
   ```sh
   sudo -E ./install_NEO_OCL_driver.sh
   ```
   > **NOTE**: To use the **Intel® Iris® Xe MAX Graphics**, see the [Intel® Iris® Xe MAX Graphics with Linux*](https://dgpu-docs.intel.com/devices/iris-xe-max-graphics/index.html) page for driver installation instructions.
   
   The script compares the driver version on the system to the current version. If the driver version on the system is higher or equal to the current version, the script does 
   not install a new driver. If the version of the driver is lower than the current version, the script uninstalls the lower version and installs the current version with your permission:
   ![](../img/NEO_check_agreement.png) 

   Higher hardware versions require a higher driver version, namely 20.35 instead of 19.41. If the script fails to uninstall the driver, uninstall it manually. During the script execution, you may see the following command line output:  
   ```sh
   Add OpenCL user to video group    
   ```
   Ignore this suggestion and continue.<br>
   You can also find the most recent version of the driver, installation procedure and other information on the [Intel® software for general purpose GPU capabilities](https://dgpu-docs.intel.com/index.html) site.

3. **Optional:** Install header files to allow compilation of new code. You can find the header files at [Khronos OpenCL™ API Headers](https://github.com/KhronosGroup/OpenCL-Headers.git).

You've completed all required configuration steps to perform inference on processor graphics. 
Proceed to the <a href="#get-started">Start Using the Toolkit</a> section to learn the basic OpenVINO™ toolkit workflow and run code samples and demo applications.

### <a name="install-ncs2"></a>Optional: Steps for Intel® Neural Compute Stick 2

These steps are only required if you want to perform inference on Intel® Movidius™ NCS powered by the Intel® Movidius™ Myriad™ 2 VPU or Intel® Neural Compute Stick 2 powered by the Intel® Movidius™ Myriad™ X VPU. For more details, see the [Get Started page for Intel® Neural Compute Stick 2:](https://software.intel.com/en-us/neural-compute-stick/get-started).

1. Go to the install_dependencies directory:
   ```sh
   cd /opt/intel/openvino_2021/install_dependencies/
   ```
2. Run the `install_NCS_udev_rules.sh` script:
   ```
   ./install_NCS_udev_rules.sh
   ```
3. You may need to reboot your machine for this to take effect.

You've completed all required configuration steps to perform inference on Intel® Neural Compute Stick 2. 
Proceed to the <a href="#get-started">Start Using the Toolkit</a> section to learn the basic OpenVINO™ toolkit workflow and run code samples and demo applications.

### <a name="install-vpu"></a>Optional: Steps for Intel® Vision Accelerator Design with Intel® Movidius™ VPUs

To install and configure your Intel® Vision Accelerator Design with Intel® Movidius™ VPUs, see the [Intel® Vision Accelerator Design with Intel® Movidius™ VPUs Configuration Guide](installing-openvino-linux-ivad-vpu.md).

> **NOTE**: After installing your Intel® Movidius™ VPU, you will return to this guide to complete the Intel® Distribution of OpenVINO™ installation.

After configuration is done, you are ready to run the verification scripts with the HDDL Plugin for your Intel® Vision Accelerator Design with Intel® Movidius™ VPUs:

1. Go to the **Inference Engine demo** directory:
   ```sh
   cd /opt/intel/openvino_2021/deployment_tools/demo
   ```

2. Run the **Image Classification verification script**. If you have access to the Internet through the proxy server only, please make sure that it is configured in your OS environment.
   ```sh
   ./demo_squeezenet_download_convert_run.sh -d HDDL
   ```

3. Run the **Inference Pipeline verification script**:
   ```sh
   ./demo_security_barrier_camera.sh -d HDDL
   ```

You've completed all required configuration steps to perform inference on Intel® Vision Accelerator Design with Intel® Movidius™ VPUs. 
Proceed to the <a href="#get-started">Start Using the Toolkit</a> section to learn the basic OpenVINO™ toolkit workflow and run code samples and demo applications.

## <a name="get-started"></a>Step 6: Start Using the Toolkit

Now you are ready to try out the toolkit. To continue, see the [Get Started Guide](../get_started/get_started.md) section to learn the basic OpenVINO™ toolkit workflow and run code samples and demo applications with pre-trained models on different inference devices.

## Troubleshooting

PRC developers might encounter pip errors during OpenVINO™ installation. To resolve the issues, try one of the following options:
* Add the download source using the `-i` parameter with the Python `pip` command. For example: 

   ```
   pip install numpy.py -i https://mirrors.aliyun.com/pypi/simple/
   ```
Use the `--trusted-host` parameter if the URL above is `http` instead of `https`.

* Modify or create the `~/.pip/pip.conf` file to change the default download source with the content below:

   ```
   [global]
   index-url = http://mirrors.aliyun.com/pypi/simple/
   [install]
   trusted-host = mirrors.aliyun.com
   ```
## <a name="uninstall"></a>Uninstall the Intel® Distribution of OpenVINO™ Toolkit

To uninstall the toolkit, follow the steps on the [Uninstalling page](uninstalling-openvino.md).

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
