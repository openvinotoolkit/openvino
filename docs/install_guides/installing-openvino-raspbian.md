# Install OpenVINO™ toolkit for Raspbian* OS {#openvino_docs_install_guides_installing_openvino_raspbian}

> **NOTE**:
> - These steps apply to 32-bit Raspbian\* OS, which is an official OS for Raspberry Pi\* boards.
> - These steps have been validated with Raspberry Pi 3*.
> - All steps in this guide are required unless otherwise stated.
> - An internet connection is required to follow the steps in this guide. If you have access to the Internet through the proxy server only, please make sure that it is configured in your OS environment.

## Introduction

The OpenVINO™ toolkit quickly deploys applications and solutions that emulate human vision. Based on Convolutional Neural Networks (CNN), the toolkit extends computer vision (CV) workloads across Intel® hardware, maximizing performance. The OpenVINO toolkit includes the Intel® Deep Learning Deployment Toolkit (Intel® DLDT).

The OpenVINO™ toolkit for Raspbian* OS includes the Inference Engine and the MYRIAD plugins. You can use it with the Intel® Neural Compute Stick 2 plugged into one of USB ports. This device is required for using the Intel® Distribution of OpenVINO™ toolkit.

> **NOTE**: There is also an open-source version of OpenVINO™ that can be compiled for arch64 (see [build instructions](https://github.com/openvinotoolkit/openvino/wiki/BuildingForRaspbianStretchOS)).

Because OpenVINO for Raspbian* OS doesn't include Model Optimizer, the ideal scenario is to use another machine to convert your model with Model Optimizer, then do your application development on the Raspberry Pi* for a convenient build/test cycle on the target platform.

### Included in the Installation Package

The OpenVINO toolkit for Raspbian OS is an archive with pre-installed header files and libraries. The following components are installed by default:

| Component                                                                                           | Description                                                                                                                                                                                                                                                  |
| :-------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Inference Engine](../IE_DG/Deep_Learning_Inference_Engine_DevGuide.md)               | This is the engine that runs the deep learning model. It includes a set of libraries for an easy inference integration into your applications.                                                                                                               |
| [OpenCV\*](https://docs.opencv.org/master/)                                                         | OpenCV\* community version compiled for Intel® hardware. |
| [Sample Applications](../IE_DG/Samples_Overview.md)                                             | A set of simple console applications demonstrating how to use Intel's Deep Learning Inference Engine in your applications.               |

> **NOTE**:
> * The package does not include the [Model Optimizer](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md). To convert models to Intermediate Representation (IR), you need to install it separately to your host machine.
> * The package does not include the Open Model Zoo demo applications. You can download them separately from the [Open Models Zoo repository](https://github.com/openvinotoolkit/open_model_zoo).

## Development and Target Platforms

**Hardware**

- Raspberry Pi\* board with ARM* ARMv7-A CPU architecture. Check that `uname -m` returns `armv7l`.
- Intel® Neural Compute Stick 2, which as one of the Intel® Movidius™ Visual Processing Units (VPUs)

> **NOTE**: With OpenVINO™ 2020.4 release, Intel® Movidius™ Neural Compute Stick (1) is no longer supported.  

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

The guide assumes you downloaded the OpenVINO toolkit for Raspbian* OS. If you do not have a copy of the toolkit package file `l_openvino_toolkit_runtime_raspbian_p_<version>.tgz`, download the latest version from the [OpenVINO™ Toolkit packages storage](https://storage.openvinotoolkit.org/repositories/openvino/packages/) and then return to this guide to proceed with the installation.

> **NOTE**: The OpenVINO toolkit for Raspbian OS is distributed without an installer, so you need to perform some extra steps compared to the [Intel® Distribution of OpenVINO™ toolkit for Linux* OS](installing-openvino-linux.md).

1. Open the Terminal\* or your preferred console application.
2. Go to the directory in which you downloaded the OpenVINO toolkit. This document assumes this is your `~/Downloads` directory. If not, replace `~/Downloads` with the directory where the file is located.
   ```sh
   cd ~/Downloads/
   ```
   By default, the package file is saved as `l_openvino_toolkit_runtime_raspbian_p_<version>.tgz`.
3. Create an installation folder.
   ```sh
   sudo mkdir -p /opt/intel/openvino_2021
   ```
4. Unpack the archive:
   ```sh
   sudo tar -xf  l_openvino_toolkit_runtime_raspbian_p_<version>.tgz --strip 1 -C /opt/intel/openvino_2021
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
source /opt/intel/openvino_2021/bin/setupvars.sh
```

**(Optional)** The OpenVINO environment variables are removed when you close the shell. As an option, you can permanently set the environment variables as follows:
```sh
echo "source /opt/intel/openvino_2021/bin/setupvars.sh" >> ~/.bashrc
```

To test your change, open a new terminal. You will see the following:
```
[setupvars.sh] OpenVINO environment initialized
```

## <a name="add-usb-rules"></a>Add USB Rules for an Intel® Neural Compute Stick 2 device
This task applies only if you have an Intel® Neural Compute Stick 2 device.

1. Add the current Linux user to the `users` group:
   ```sh
   sudo usermod -a -G users "$(whoami)"
   ```
   Log out and log in for it to take effect.
2. If you didn't modify `.bashrc` to permanently set the environment variables, run `setupvars.sh` again after logging in:
   ```sh
   source /opt/intel/openvino_2021/bin/setupvars.sh
   ```
3. To perform inference on the Intel® Neural Compute Stick 2, install the USB rules running the `install_NCS_udev_rules.sh` script:
   ```sh
   sh /opt/intel/openvino_2021/install_dependencies/install_NCS_udev_rules.sh
   ```
4. Plug in your Intel® Neural Compute Stick 2.

You are now ready to compile and run the Object Detection sample to verify the Inference Engine installation.

## <a name="run-sample"></a>Build and Run Object Detection Sample

Follow the next steps to use the pre-trained face detection model using Inference Engine samples from the OpenVINO toolkit.

1. Navigate to a directory that you have write access to and create a samples build directory. This example uses a directory named `build`:
   ```sh
   mkdir build && cd build
   ```
2. Build the Object Detection Sample:
   ```sh
   cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=armv7-a" /opt/intel/openvino_2021/deployment_tools/inference_engine/samples/cpp
   ```
   ```sh
   make -j2 object_detection_sample_ssd
   ```
3. Download the pre-trained Face Detection model with the Model Downloader or copy it from the host machine:
   ```sh
   git clone --depth 1 https://github.com/openvinotoolkit/open_model_zoo
   cd open_model_zoo/tools/downloader
   python3 -m pip install -r requirements.in
   python3 downloader.py --name face-detection-adas-0001 
   ```
4. Run the sample specifying the model, a path to the input image, and the VPU required to run with the Raspbian* OS:
   ```sh
   ./armv7l/Release/object_detection_sample_ssd -m <path_to_model>/face-detection-adas-0001.xml -d MYRIAD -i <path_to_image>
   ```
   The application outputs an image (`out_0.bmp`) with detected faced enclosed in rectangles.

Congratulations, you have finished the OpenVINO™ toolkit for Raspbian* OS installation. You have completed all required installation, configuration and build steps in this guide.

Read the next topic if you want to learn more about OpenVINO workflow for Raspberry Pi.

## <a name="workflow-for-raspberry-pi"></a>Workflow for Raspberry Pi*

If you want to use your model for inference, the model must be converted to the .bin and .xml Intermediate Representation (IR) files that are used as input by Inference Engine. OpenVINO™ toolkit support on Raspberry Pi only includes the Inference Engine module of the Intel® Distribution of OpenVINO™ toolkit. The Model Optimizer is not supported on this platform. To get the optimized models you can use one of the following options:

* Download public and Intel's pre-trained models from the [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo) using [Model Downloader tool](@ref omz_tools_downloader).

   For more information on pre-trained models, see [Pre-Trained Models Documentation](@ref omz_models_group_intel)

* Convert the model using the Model Optimizer from a full installation of Intel® Distribution of OpenVINO™ toolkit on one of the supported platforms. Installation instructions are available:

   * [Installation Guide for macOS*](installing-openvino-macos.md)
   * [Installation Guide for Windows*](installing-openvino-windows.md)
   * [Installation Guide for Linux*](installing-openvino-linux.md)

   For more information about how to use the Model Optimizer, see the [Model Optimizer Developer Guide](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
