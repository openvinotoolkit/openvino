# Install Intel® Distribution of OpenVINO™ toolkit for Linux* with FPGA Support {#openvino_docs_install_guides_installing_openvino_linux_fpga}

**NOTES**:
- [Intel® System Studio](https://software.intel.com/en-us/system-studio) is an all-in-one, cross-platform tool suite, purpose-built to simplify system bring-up and improve system and IoT device application performance on Intel® platforms. If you are using the Intel® Distribution of OpenVINO™ with Intel® System Studio, go to [Get Started with Intel® System Studio](https://software.intel.com/en-us/articles/get-started-with-openvino-and-intel-system-studio-2019).
- The Intel® Distribution of OpenVINO™ toolkit was formerly known as the Intel® Computer Vision SDK.
- These steps apply to Ubuntu\*, CentOS\*, and Yocto\*. 
- If you are using Intel® Distribution of OpenVINO™ toolkit on Windows\* OS, see the [Installation Guide for Windows*](installing-openvino-windows.md).
- For the Intel Distribution of OpenVINO toolkit without FPGA
support, see [Installation Guide for Linux*](installing-openvino-linux.md).
- CentOS and Yocto installations will require some modifications that
are not covered in this guide.
- An internet connection is required to follow the steps in this guide.

## Introduction

The Intel® Distribution of OpenVINO™ toolkit quickly deploys applications and solutions that emulate human vision. Based on Convolutional Neural Networks (CNN), the toolkit extends computer vision (CV) workloads across Intel® hardware, maximizing performance. The Intel® Distribution of OpenVINO™ toolkit includes the Intel® Deep Learning Deployment Toolkit (Intel® DLDT).

The Intel® Distribution of OpenVINO™ toolkit for Linux\* with FPGA Support:

- Enables CNN-based deep learning inference on the edge
- Supports heterogeneous execution across Intel® CPU, Intel® Integrated Graphics, Intel® FPGA, Intel® Neural Compute Stick 2
- Speeds time-to-market via an easy-to-use library of computer vision functions and pre-optimized kernels
- Includes optimized calls for computer vision standards including OpenCV\* and OpenCL™

**Included with the Installation and installed by default:**

| Component                                                                                           | Description                                                                                                                                                                                                                                                                                                   |  
|-----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Model Optimizer](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md) | This tool imports, converts, and optimizes models that were trained in popular frameworks to a format usable by Intel tools, especially the Inference Engine. <br>Popular frameworks include Caffe\*, TensorFlow\*, MXNet\*, and ONNX\*.                                                                                                |
| [Inference Engine](../IE_DG/inference_engine_intro.md)               | This is the engine that runs the deep learning model. It includes a set of libraries for an easy inference integration into your applications.                                                                                                                                                                                           |
| Drivers and runtimes for OpenCL™ version 2.1                                                        | Enables OpenCL on the GPU/CPU for Intel® processors                                                                                                                                                                                                                                                           |
| Intel® Media SDK                                                                                    | Offers access to hardware accelerated video codecs and frame processing                                                                                                                                                                                                                                       |
| Pre-compiled FPGA bitstream samples                                                                 | Pre-compiled bitstream samples for the Intel® Programmable Acceleration Card with Intel® Arria® 10 GX FPGA, and Intel® Vision Accelerator Design with an Intel® Arria 10 FPGA SG2.                                                                                                                     |
| Intel® FPGA SDK for OpenCL™ software technology                                                     | The Intel® FPGA RTE for OpenCL™ provides utilities, host runtime libraries, drivers, and RTE-specific libraries and files                                                                                                                                                                                     |
| [OpenCV](https://docs.opencv.org/master/)                                   | OpenCV\* community version compiled for Intel® hardware   |
| [Inference Engine Code Samples](../IE_DG/Samples_Overview.md)           | A set of simple console applications demonstrating how to utilize specific OpenVINO capabilities in an application and how to perform specific tasks, such as loading a model, running inference, querying specific device capabilities, and more. |
| [Demo Applications](@ref omz_demos_README)           | A set of simple console applications that provide robust application templates to help you implement specific deep learning scenarios. |


## Development and Target Platform

The development and target platforms have the same requirements, but you can select different components during the installation, based on your intended use.

**Hardware**

* 6th to 10th generation Intel® Core™ processors and Intel® Xeon® processors 
* Intel® Xeon® processor E family (formerly code named Sandy Bridge, Ivy Bridge, Haswell, and Broadwell)
* 3rd generation Intel® Xeon® Scalable processor (formerly code named Cooper Lake)
* Intel® Xeon® Scalable processor (formerly Skylake and Cascade Lake)
* Intel Atom® processor with support for Intel® Streaming SIMD Extensions 4.1 (Intel® SSE4.1)
* Intel Pentium® processor N4200/5, N3350/5, or N3450/5 with Intel® HD Graphics
* Intel® Neural Compute Stick 2
* Intel® Vision Accelerator Design with Intel® Movidius™ VPUs
* Intel® Programmable Acceleration Card (PAC) with Intel® Arria® 10 GX FPGA
* Intel® Vision Accelerator Design with an Intel® Arria 10 FPGA (Mustang-F100-A10) SG2

> **NOTE**: With OpenVINO™ 2020.4 release, Intel® Movidius™ Neural Compute Stick is no longer supported.

> **NOTE**: Intel® Arria 10 FPGA (Mustang-F100-A10) SG1 is no longer supported. If you use Intel® Vision Accelerator Design with an Intel® Arria 10 FPGA (Mustang-F100-A10) Speed Grade 1, we recommend continuing to use the [Intel® Distribution of OpenVINO™ toolkit 2020.1](https://docs.openvinotoolkit.org/2020.1/_docs_install_guides_VisionAcceleratorFPGA_Configure.html) release.

> **NOTE**: Intel® Arria® 10 GX FPGA Development Kit is no longer supported. For the Intel® Arria® 10 FPGA GX Development Kit configuration guide, refer to the [2019 R1.1 documentation](http://docs.openvinotoolkit.org/2019_R1.1/_docs_install_guides_GX_Configure_2019R1.html).

**Processor Notes:**

- Processor graphics are not included in all processors. See [Product Specifications](https://ark.intel.com/) for information about your processor.
- A chipset that supports processor graphics is required for Intel® Xeon® processors.

**Operating Systems:**

- Ubuntu 18.04 or 16.04 long-term support (LTS), 64-bit: Minimum supported kernel is 4.15
- CentOS 7.4, 64-bit
- Yocto Project v3.0, 64-bit (for target only and requires modifications)

## Overview

This guide provides step-by-step instructions on how to install the Intel® Distribution of OpenVINO™ toolkit with FPGA Support. Links are provided for each type of compatible hardware including downloads, initialization and configuration steps. The following steps will be covered:

1. <a href="#install-openvino">Install the Intel® Distribution of OpenVINO™ Toolkit </a>
2. <a href="#install-external-dependencies">Install External software dependencies</a>
3. <a href="#configure-model-optimizer">Configure the Model Optimizer </a>
4. <a href="#run-the-demos">Run the Verification Scripts to Verify Installation and Compile Samples</a>
5. <a href="#install-hardware">Install your compatible hardware from the list of supported hardware</a><br>
6. <a href="#Hello-World-Face-Detection-Tutorial">Use the Face Detection Tutorial</a>

## <a name="install-openvino"></a>Install the Intel® Distribution of OpenVINO™ Toolkit Core Components

Download the Intel® Distribution of OpenVINO™ toolkit package file from [Intel® Distribution of OpenVINO™ toolkit for Linux* with FPGA Support](https://software.intel.com/en-us/openvino-toolkit/choose-download). 
Select the Intel® Distribution of OpenVINO™ toolkit for Linux with FPGA Support package from the dropdown menu.

1. Open a command prompt terminal window.
2. Change directories to where you downloaded the Intel Distribution of
OpenVINO toolkit for Linux\* with FPGA Support package file.<br>
     If you downloaded the package file to the current user's `Downloads`
     directory:
```sh
cd ~/Downloads/
```
By default, the file is saved as `l_openvino_toolkit_fpga_p_<version>.tgz`.

3. Unpack the .tgz file:
```sh
tar -xvzf l_openvino_toolkit_fpga_p_<version>.tgz
```
The files are unpacked to the `l_openvino_toolkit_fpga_p_<version>` directory.

4. Go to the `l_openvino_toolkit_fpga_p_<version>` directory:
```sh
cd l_openvino_toolkit_fpga_p_<version>
```
If you have a previous version of the Intel Distribution of OpenVINO toolkit installed, rename or delete these two directories:
- `/home/<user>/inference_engine_samples`
- `/home/<user>/openvino_models`

**Installation Notes:**
- Choose an installation option and run the related script as root.
- You can use either a GUI installation wizard or command-line instructions (CLI).
- Screenshots are provided for the GUI, but not for CLI. The following information also applies to CLI and will be helpful to your installation where you will be presented with the same choices and tasks.

5. Choose your installation option:
   - **Option 1:** GUI Installation Wizard:
```sh
sudo ./install_GUI.sh
```
   - **Option 2:** Command-Line Instructions:
```sh
sudo ./install.sh
```
6. Follow the instructions on your screen. Watch for informational
messages such as the following in case you must complete additional
steps:
![](../img/install-linux-fpga-01.png)

7. If you select the default options, the **Installation summary** GUI screen looks like this:
![](../img/install-linux-fpga-02.png)
    - **Optional:** You can choose **Customize** and select only the bitstreams for your card. This will allow you to minimize
        the size of the download by several gigabytes.
    - The following bitstreams listed at the bottom of the customization screen are highlighted below. Choose the one for your FPGA:
        ![](../img/install-linux-fpga-04.png)
    - When installed as **root** the default installation directory for the Intel Distribution of OpenVINO is
    `/opt/intel/openvino_fpga_2019.<version>/`.<br>
        For simplicity, a symbolic link to the latest installation is also created: `/opt/intel/openvino/`.

8. A Complete screen indicates that the core components have been installed:
![](../img/install-linux-fpga-05.png)

The first core components are installed. Continue to the next section to install additional dependencies.

## <a name="install-external-dependencies"></a>Install External Software Dependencies

These dependencies are required for:

- Intel-optimized build of OpenCV library
- Deep Learning Inference Engine
- Deep Learning Model Optimizer tools

1. Change to the `install_dependencies` directory:
```sh
cd /opt/intel/openvino/install_dependencies   
```
2. Run a script to download and install the external software dependencies:
```sh
sudo -E ./install_openvino_dependencies.sh
```

The dependencies are installed. Continue to the next section to configure the Model Optimizer.

## <a name="configure-model-optimizer"></a>Configure the Model Optimizer

The Model Optimizer is a Python\*-based command line tool for importing
trained models from popular deep learning frameworks such as Caffe\*,
TensorFlow\*, Apache MXNet\*, ONNX\* and Kaldi\*.

The Model Optimizer is a key component of the Intel Distribution of
OpenVINO toolkit. You cannot perform inference on your trained model without
running the model through the Model Optimizer. When you run a
pre-trained model through the Model Optimizer, your output is an
Intermediate Representation (IR) of the network. The Intermediate
Representation is a pair of files that describe the whole model:

-   `.xml`: Describes the network topology
-   `.bin`: Contains the weights and biases binary data

For more information about the Model Optimizer, refer to the [Model Optimizer Developer Guide](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md). 

### Model Optimizer Configuration Steps

> **IMPORTANT**: The Internet access is required to execute the following steps successfully. If you have access to the Internet through the proxy server only, please make sure that it is configured in your environment.

You can choose to either configure all supported frameworks at once **OR** configure one framework at a time. Choose the option that best suits your needs. If you see error messages, make sure you installed all dependencies.

> **NOTE**: If you installed the Intel® Distribution of OpenVINO™ to the non-default install directory, replace `/opt/intel` with the directory in which you installed the software.

**Option 1: Configure all supported frameworks at the same time**

1.  Go to the Model Optimizer prerequisites directory:
```sh
cd /opt/intel/openvino/deployment_tools/model_optimizer/install_prerequisites 
```
2.  Run the script to configure the Model Optimizer for Caffe,
    TensorFlow, MXNet, Kaldi\*, and ONNX:
```sh
sudo ./install_prerequisites.sh
```

**Option 2: Configure each framework separately**

Configure individual frameworks separately **ONLY** if you did not select **Option 1** above.
1.  Go to the Model Optimizer prerequisites directory:
```sh
cd /opt/intel/openvino/deployment_tools/model_optimizer/install_prerequisites
```
2.  Run the script for your model framework. You can run more than one script:
-   For **Caffe**:

```sh
sudo ./install_prerequisites_caffe.sh
```
-   For **TensorFlow**:
```sh
sudo ./install_prerequisites_tf.sh
```
-   For **MXNet**:
```sh
sudo ./install_prerequisites_mxnet.sh
```
-   For **ONNX**:
```sh
sudo ./install_prerequisites_onnx.sh
```
-   For **Kaldi**:
```sh
sudo ./install_prerequisites_kaldi.sh
```
The Model Optimizer is configured for one or more frameworks.

You are ready to compile the samples by <a href="#run-the-demos">running the verification scripts</a>.

## <a name="run-the-demos"></a>Run the Verification Scripts to Verify Installation and Compile Samples

To verify the installation and compile two samples, run the verification applications provided with the product on the CPU:

1. Go to the **Inference Engine demo** directory:
```sh
cd /opt/intel/openvino/deployment_tools/demo
```

2. Run the **Image Classification verification script**:
```sh
./demo_squeezenet_download_convert_run.sh
```
This verification script downloads a SqueezeNet model, uses the Model Optimizer to convert the model to the .bin and .xml Intermediate Representation (IR) files. The Inference Engine requires this model conversion so it can use the IR as input and achieve optimum performance on Intel hardware.<br>
This verification script builds the [Image Classification Sample Async](../../inference-engine/samples/classification_sample_async/README.md) application and run it with the `car.png` image in the demo directory. When the verification script completes, you will have the label and confidence for the top-10 categories:
![](../img/image_classification_script_output_lnx.png)

3. Run the **Inference Pipeline verification script**:
```sh
./demo_security_barrier_camera.sh
```
This verification script builds the [Security Barrier Camera Demo](@ref omz_demos_security_barrier_camera_demo_README) application included in the package. 

   This verification script uses the `car_1.bmp` image in the demo directory to show an inference pipeline using three of the pre-trained models. The verification script uses vehicle recognition in which vehicle attributes build on each other to narrow in on a specific attribute.

   First, an object is identified as a vehicle. This identification is used as input to the next model, which identifies specific vehicle attributes, including the license plate. Finally, the attributes identified as the license plate are used as input to the third model, which recognizes specific characters in the license plate.

  When the verification script completes, you will see an image that displays the resulting frame with detections rendered as bounding boxes, and text:
  ![](../img/security-barrier-results.png)

4. Close the image viewer window to complete the verification script.


To learn about the verification scripts, see the `README.txt` file in `/opt/intel/openvino/deployment_tools/demo`.

For a description of the Intel Distribution of OpenVINO™ pre-trained object detection and object recognition models, see [Overview of OpenVINO™ Toolkit Pre-Trained Models](@ref omz_models_intel_index).

You have completed all required installation, configuration and build steps in this guide to use your CPU to work with your trained models. To use other hardware, see <a href="#install hardware">Install and Configure your Compatible Hardware</a> below.

## <a name="install-hardware"></a>Install and Configure Your Compatible Hardware

Install your compatible hardware from the list of supported components below.

> **NOTE**: Once you've completed your hardware installation, you'll return to this guide to finish installation and configuration of the Intel® Distribution of OpenVINO™ toolkit.

Links to install and configure compatible hardware
- [The Intel® Programmable Acceleration Card (PAC) with Intel® Arria® 10 GX FPGA](PAC_Configure.md)
- [The Intel® Vision Accelerator Design with an Intel® Arria 10 FPGA SG2 (Mustang-F100-A10)](VisionAcceleratorFPGA_Configure.md)
- [Intel® Vision Accelerator Design with Intel® Movidius™ VPUs](installing-openvino-linux-ivad-vpu.md)

Congratulations, you have finished the Intel® Distribution of OpenVINO™ toolkit installation for FPGA. To learn more about how the Intel® Distribution of OpenVINO™ toolkit works, the Hello World tutorial and other resources are provided below.

## <a name="Hello-World-Face-Detection-Tutorial"></a>Hello World Face Detection Tutorial

Refer to the [OpenVINO™ with FPGA Hello World Face Detection Exercise](https://github.com/intel-iot-devkit/openvino-with-fpga-hello-world-face-detection).

**Additional Resources**

- Intel® Distribution of OpenVINO™ toolkit home page: [https://software.intel.com/en-us/openvino-toolkit](https://software.intel.com/en-us/openvino-toolkit)
- OpenVINO™ toolkit online documentation: [https://docs.openvinotoolkit.org](https://docs.openvinotoolkit.org)
- [Inference Engine FPGA plugin documentation](../IE_DG/supported_plugins/FPGA.md)
- [Model Optimizer Developer Guide](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
- For more information on Sample Applications, see the [Inference Engine Samples Overview](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_Samples_Overview.html)
- To learn about pre-trained models for OpenVINO™ toolkit, see the [Pre-Trained Models Overview](https://docs.openvinotoolkit.org/latest/_docs_docs_Pre_Trained_Models.html)
- For information on Inference Engine Tutorials, see the [Inference Tutorials](https://github.com/intel-iot-devkit/inference-tutorials-generic)
- For IoT Libraries & Code Samples see the [Intel® IoT Developer Kit](https://github.com/intel-iot-devkit).

To learn more about converting models, go to:

- [Convert Your Caffe* Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_Caffe.md)
- [Convert Your TensorFlow* Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_TensorFlow.md)
- [Convert Your MXNet* Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_MxNet.md)
- [Convert Your ONNX* Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_ONNX.md)


