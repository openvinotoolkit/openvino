# Install OpenVINO™ toolkit for Raspbian* OS {#openvino_docs_install_guides_installing_openvino_raspbian}

> **NOTE**:
> - These steps apply to 32-bit Raspbian\* OS, which is an official OS for Raspberry Pi\* boards.
> - These steps have been validated with Raspberry Pi 3*.
> - All steps in this guide are required unless otherwise stated.
> - An internet connection is required to follow the steps in this guide. If you have access to the Internet through the proxy server only, please make sure that it is configured in your OS environment.

## Introduction

The OpenVINO™ toolkit quickly deploys applications and solutions that emulate human vision. Based on Convolutional Neural Networks (CNN), the toolkit extends computer vision (CV) workloads across Intel® hardware, maximizing performance. The OpenVINO toolkit includes the Intel® Deep Learning Deployment Toolkit (Intel® DLDT).

The OpenVINO™ toolkit for Raspbian* OS includes the Inference Engine and the MYRIAD plugins. You can use it with the Intel® Neural Compute Stick 2 plugged in one of USB ports.

### Included in the Installation Package

The OpenVINO toolkit for Raspbian OS is an archive with pre-installed header files and libraries. The following components are installed by default:

| Component                                                                                           | Description                                                                                                                                                                                                                                                  |
| :-------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Inference Engine](../IE_DG/inference_engine_intro.md)               | This is the engine that runs the deep learning model. It includes a set of libraries for an easy inference integration into your applications.                                                                                                               |
| [OpenCV\*](https://docs.opencv.org/master/)                                                         | OpenCV\* community version compiled for Intel® hardware. |
| [Sample Applications](../IE_DG/Samples_Overview.md)                                             | A set of simple console applications demonstrating how to use Intel's Deep Learning Inference Engine in your applications.               |

> **NOTE**:
> * The package does not include the [Model Optimizer](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md). To convert models to Intermediate Representation (IR), you need to install it separately to your host machine.
> * The package does not include the Open Model Zoo demo applications. You can download them separately from the [Open Models Zoo repository](https://github.com/opencv/open_model_zoo).

> **TIP**: Instead of installing the toolkit on your system, you can work with OpenVINO™ components inside the web-based graphical environment of the [OpenVINO™ Deep Learning Workbench](@ref openvino_docs_get_started_get_started_dl_workbench) after a [fast installation from Docker](@ref workbench_docs_Workbench_DG_Docker_Container).
> DL Workbench enables you to visualize, fine-tune, and compare performance of deep learning models on various Intel® architecture configurations, such as CPU,
> Intel® Processor Graphics (GPU), Intel® Movidius™ Neural Compute Stick 2 (NCS 2), and Intel® Vision Accelerator Design with Intel® Movidius™ VPUs. 
> The intuitive web-based interface of the DL Workbench enables you to easily use various sophisticated
> OpenVINO™ toolkit components: [Model Downloader](@ref omz_tools_downloader_README), [Intel® Open Model Zoo](@ref omz_models_intel_index), 
> [Model Optimizer](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md), [Post-Training Optimization tool](@ref pot_README),
> [Accuracy Checker](@ref omz_tools_accuracy_checker_README), and [Benchmark Tool](@ref openvino_inference_engine_samples_benchmark_app_README).

## Development and Target Platforms

**Hardware**

- Raspberry Pi\* board with ARM* ARMv7-A CPU architecture. Check that `uname -m` returns `armv7l`.
- One of Intel® Movidius™ Visual Processing Units (VPU):
  - Intel® Neural Compute Stick 2

> **NOTE**: With OpenVINO™ 2020.4 release, Intel® Movidius™ Neural Compute Stick is no longer supported.  

**Operating Systems**

- Raspbian\* Buster, 32-bit
- Raspbian\* Stretch, 32-bit

**Software**

- CMake* 3.7.2 or higher
- Python* 3.5, 32-bit


## Overview

This guide provides step-by-step instructions on how to install the OpenVINO™ toolkit for Raspbian* OS. Links are provided for each type of compatible hardware including downloads, initialization and configuration steps. The following steps will be covered:

1. [Install the OpenVINO™ toolkit](#install-package)
2. [Install External Software Dependencies](#install-dependencies)
3. [Set the environment variables](#set-environment-variables)
4. [Add USB rules](#add-usb-rules)
5. [Run the Object Detection Sample](#run-sample) to validate Inference Engine installation
6. [Learn About Workflow for Raspberry Pi](#workflow-for-raspberry-pi)

## <a name="install-package"></a>Install the OpenVINO™ Toolkit for Raspbian* OS Package

The guide assumes you downloaded the OpenVINO toolkit for Raspbian* OS. If you do not have a copy of the toolkit package file `l_openvino_toolkit_runtime_raspbian_p_<version>.tgz`, download the latest version from the [Intel® Open Source Technology Center](https://download.01.org/opencv/2020/openvinotoolkit/) and then return to this guide to proceed with the installation.

> **NOTE**: The OpenVINO toolkit for Raspbian OS is distributed without installer, so you need to perform extra steps comparing to the [Intel® Distribution of OpenVINO™ toolkit for Linux* OS](installing-openvino-linux.md).

1. Open the Terminal\* or your preferred console application.

2. Go to the directory in which you downloaded the OpenVINO toolkit. This document assumes this is your `~/Downloads` directory. If not, replace `~/Downloads` with the directory where the file is located.
```sh
cd ~/Downloads/
```
By default, the package file is saved as `l_openvino_toolkit_runtime_raspbian_p_<version>.tgz`.

3. Create an installation folder.
```sh
sudo mkdir -p /opt/intel/openvino
```

4. Unpack the archive:
```sh
sudo tar -xf  l_openvino_toolkit_runtime_raspbian_p_<version>.tgz --strip 1 -C /opt/intel/openvino
```

Now the OpenVINO toolkit components are installed. Additional configuration steps are still required. Continue to the next sections to install External Software Dependencies, configure the environment and set up USB rules.

## <a name="install-dependencies"></a>Install External Software Dependencies

CMake* version 3.7.2 or higher is required for building the Inference Engine sample application. To install, open a Terminal* window and run the following command:
```sh
sudo apt install cmake
```

CMake is installed. Continue to the next section to set the environment variables.

## <a name="set-environment-variables"></a>Set the Environment Variables

You must update several environment variables before you can compile and run OpenVINO toolkit applications. Run the following script to temporarily set the environment variables:
```sh
source /opt/intel/openvino/bin/setupvars.sh
```

**(Optional)** The OpenVINO environment variables are removed when you close the shell. As an option, you can permanently set the environment variables as follows:
```sh
echo "source /opt/intel/openvino/bin/setupvars.sh" >> ~/.bashrc
```

To test your change, open a new terminal. You will see the following:
```
[setupvars.sh] OpenVINO environment initialized
```

Continue to the next section to add USB rules for Intel® Neural Compute Stick 2 devices.

## <a name="add-usb-rules"></a>Add USB Rules

1. Add the current Linux user to the `users` group:
```sh
sudo usermod -a -G users "$(whoami)"
```
Log out and log in for it to take effect.

2. If you didn't modify `.bashrc` to permanently set the environment variables, run `setupvars.sh` again after logging in:
```sh
source /opt/intel/openvino/bin/setupvars.sh
```

3. To perform inference on the Intel® Neural Compute Stick 2, install the USB rules running the `install_NCS_udev_rules.sh` script:
```sh
sh /opt/intel/openvino/install_dependencies/install_NCS_udev_rules.sh
```
4. Plug in your Intel® Neural Compute Stick 2.

You are ready to compile and run the Object Detection sample to verify the Inference Engine installation.

## <a name="run-sample"></a>Build and Run Object Detection Sample

Follow the next steps to run pre-trained Face Detection network using Inference Engine samples from the OpenVINO toolkit.

1. Navigate to a directory that you have write access to and create a samples build directory. This example uses a directory named `build`:
```sh
mkdir build && cd build
```

2. Build the Object Detection Sample:
```sh
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=armv7-a" /opt/intel/openvino/deployment_tools/inference_engine/samples/cpp
```
```sh
make -j2 object_detection_sample_ssd
```

3. Download the pre-trained Face Detection model or copy it from the host machine:

   - To download the `.bin` file with weights:
   ```sh
   wget --no-check-certificate https://download.01.org/opencv/2020/openvinotoolkit/2020.1/open_model_zoo/models_bin/1/face-detection-adas-0001/FP16/face-detection-adas-0001.bin
   ```

   - To download the `.xml` file with the network topology:
   ```sh
   wget --no-check-certificate https://download.01.org/opencv/2020/openvinotoolkit/2020.1/open_model_zoo/models_bin/1/face-detection-adas-0001/FP16/face-detection-adas-0001.xml
   ```

4. Run the sample with specifying the model and a path to the input image:
```sh
./armv7l/Release/object_detection_sample_ssd -m face-detection-adas-0001.xml -d MYRIAD -i <path_to_image>
```
The application outputs an image (`out_0.bmp`) with detected faced enclosed in rectangles.

Congratulations, you have finished the OpenVINO™ toolkit for Raspbian* OS installation. You have completed all required installation, configuration and build steps in this guide.

Read the next topic if you want to learn more about OpenVINO workflow for Raspberry Pi.

## <a name="workflow-for-raspberry-pi"></a>Workflow for Raspberry Pi*

If you want to use your model for inference, the model must be converted to the .bin and .xml Intermediate Representation (IR) files that are used as input by Inference Engine. OpenVINO™ toolkit support on Raspberry Pi only includes the Inference Engine module of the Intel® Distribution of OpenVINO™ toolkit. The Model Optimizer is not supported on this platform. To get the optimized models you can use one of the following options:

* Download a set of ready-to-use pre-trained models for the appropriate version of OpenVINO from the Intel® Open Source Technology Center:

   * Models for the 2020.1 release of OpenVINO are available at [https://download.01.org/opencv/2020/openvinotoolkit/2020.1/open_model_zoo/](https://download.01.org/opencv/2020/openvinotoolkit/2020.1/open_model_zoo/).
   * Models for the 2019 R1 release of OpenVINO are available at [https://download.01.org/opencv/2019/open_model_zoo/R1/](https://download.01.org/opencv/2019/open_model_zoo/R1/).
   * Models for the 2018 R5 release of OpenVINO are available at [https://download.01.org/openvinotoolkit/2018_R5/open_model_zoo/](https://download.01.org/openvinotoolkit/2018_R5/open_model_zoo/).

   For more information on pre-trained models, see [Pre-Trained Models Documentation](@ref omz_models_intel_index)

* Convert the model using the Model Optimizer from a full installation of Intel® Distribution of OpenVINO™ toolkit on one of the supported platforms. Installation instructions are available:

   * [Installation Guide for macOS*](installing-openvino-macos.md)
   * [Installation Guide for Windows*](installing-openvino-windows.md)
   * [Installation Guide for Linux*](installing-openvino-linux.md)

   For more information about how to use the Model Optimizer, see the [Model Optimizer Developer Guide](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
