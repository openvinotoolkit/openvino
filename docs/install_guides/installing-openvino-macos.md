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
| [Inference Engine](../IE_DG/inference_engine_intro.md)               | This is the engine that runs a deep learning model. It includes a set of libraries for an easy inference integration into your applications.                                                                                                               |
| [OpenCV\*](https://docs.opencv.org/master/)                                                         | OpenCV\* community version compiled for Intel® hardware                                                                                                                                                                                                      |
| [Sample Applications](../IE_DG/Samples_Overview.md)                                                                                | A set of simple console applications demonstrating how to use the Inference Engine in your applications. |
| [Demos](@ref omz_demos_README)                                   | A set of console applications that demonstrate how you can use the Inference Engine in your applications to solve specific use-cases  |
| Additional Tools                                   | A set of tools to work with your models including [Accuracy Checker utility](@ref omz_tools_accuracy_checker_README), [Post-Training Optimization Tool Guide](@ref pot_README), [Model Downloader](@ref omz_tools_downloader_README) and other  |
| [Documentation for Pre-Trained Models ](@ref omz_models_intel_index)                                   | Documentation for the pre-trained models available in the [Open Model Zoo repo](https://github.com/opencv/open_model_zoo)  |

### Could Be Optionally Installed

Instead of installing the toolkit on your system, you can work with OpenVINO™ components inside the web-based graphical environment of the [OpenVINO™ Deep Learning Workbench](@ref openvino_docs_get_started_get_started_dl_workbench) after a [fast installation from Docker](@ref workbench_docs_Workbench_DG_Docker_Container). <br>
DL Workbench enables you to visualize, fine-tune, and compare performance of deep learning models on various Intel® architecture configurations using sophisticated
OpenVINO™ toolkit components: [Model Downloader](@ref omz_tools_downloader_README), [Intel® Open Model Zoo](@ref omz_models_intel_index), 
[Model Optimizer](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md), [Post-Training Optimization tool](@ref pot_README),
[Accuracy Checker](@ref omz_tools_accuracy_checker_README), and [Benchmark Tool](@ref openvino_inference_engine_samples_benchmark_app_README).

## Development and Target Platform

The development and target platforms have the same requirements, but you can select different components during the installation, based on your intended use.

**Hardware**

> **NOTE**: The current version of the Intel® Distribution of OpenVINO™ toolkit for macOS* supports inference on Intel CPUs and Intel® Neural Compute Sticks 2 only.

* 6th to 11th generation Intel® Core™ processors and Intel® Xeon® processors 
* Intel® Xeon® processor E family (formerly code named Sandy Bridge, Ivy Bridge, Haswell, and Broadwell)
* 3rd generation Intel® Xeon® Scalable processor (formerly code named Cooper Lake)
* Intel® Xeon® Scalable processor (formerly Skylake and Cascade Lake)
* Intel® Neural Compute Stick 2

**Software Requirements**

- CMake 3.10 or higher
- Python 3.6 - 3.7
- Apple Xcode\* Command Line Tools
- (Optional) Apple Xcode\* IDE (not required for OpenVINO, but useful for development)

**Operating Systems**

- macOS\* 10.15

## Overview

This guide provides step-by-step instructions on how to install the Intel® Distribution of OpenVINO™ 2020.1 toolkit for macOS*.

The following steps will be covered:

1. <a href="#Install-Core">Install the Intel® Distribution of OpenVINO™ Toolkit </a>.
2. <a href="#set-the-environment-variables">Set the OpenVINO environment variables and (optional) Update to <code>.bash_profile</code></a>.
4. <a href="#configure-the-model-optimizer">Configure the Model Optimizer</a>.
5. <a href="#Run-Demos">Run verification scripts to verify installation and compile samples</a>.

## <a name="Install-Core"></a>Install the Intel® Distribution of OpenVINO™ toolkit Core Components

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

   - If you used **root** or **administrator** privileges to run the installer, it installs the OpenVINO toolkit to `/opt/intel/openvino_<version>/`

     For simplicity, a symbolic link to the latest installation is also created: `/opt/intel/openvino_2021/`

   - If you used **regular user** privileges to run the installer, it installs the OpenVINO toolkit to `/home/<user>/intel/openvino_<version>/`

     For simplicity, a symbolic link to the latest installation is also created: `/home/<user>/intel/openvino_2021/`

9. If needed, click **Customize** to change the installation directory or the components you want to install:
    ![](../img/openvino-install-macos-04.png)

    Click **Next** to save the installation options and show the Installation summary screen.

10. On the **Installation summary** screen, press **Install** to begin the installation.

11. When the first part of installation is complete, the final screen informs you that the core components have been installed
   and additional steps still required:
   ![](../img/openvino-install-macos-05.png)

12. Click **Finish** to close the installation wizard. A new browser window opens to the next section of the Installation Guide to set the environment variables. If the installation did not indicate you must install dependencies, you can move ahead to [Set the Environment Variables](#set-the-environment-variables).  If you received a message that you were missing external software dependencies, listed under **Software Requirements** at the top of this guide, you need to install them now before continuing on to the next section.

## <a name="set-the-environment-variables"></a>Set the Environment Variables

You need to update several environment variables before you can compile and run OpenVINO™ applications. Open the macOS Terminal\* or a command-line interface shell you prefer and run the following script to temporarily set your environment variables:

   ```sh
   source /opt/intel/openvino_2021/bin/setupvars.sh
   ```  

<strong>Optional</strong>: The OpenVINO environment variables are removed when you close the shell. You can permanently set the environment variables as follows:

1. Open the `.bash_profile` file in the current user home directory:
   ```sh
   vi ~/.bash_profile
   ```
2. Press the **i** key to switch to the insert mode.

3. Add this line to the end of the file:
   ```sh
   source /opt/intel/openvino_2021/bin/setupvars.sh
   ```

3. Save and close the file: press the **Esc** key, type `:wq` and press the **Enter** key.

4. To verify your change, open a new terminal. You will see `[setupvars.sh] OpenVINO environment initialized`.

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

You are ready to verify the installation by <a href="#Run-Demos">running the verification scripts</a>.

## <a name="Run-Demos"></a>Run the Verification Scripts to Verify Installation and Compile Samples

> **NOTES**:
> - The steps shown here assume you used the default installation directory to install the OpenVINO toolkit. If you installed the software to a directory other than `/opt/intel/`, update the directory path with the location where you installed the toolkit.
> - If you installed the product as a root user, you must switch to the root mode before you continue: `sudo -i`.

To verify the installation and compile two Inference Engine samples, run the verification applications provided with the product on the CPU:

### Run the Image Classification Verification Script

1. Go to the **Inference Engine demo** directory:
   ```sh
   cd /opt/intel/openvino_2021/deployment_tools/demo
   ```

2. Run the **Image Classification verification script**:
   ```sh
   ./demo_squeezenet_download_convert_run.sh
   ```  

The Image Classification verification script downloads a public SqueezeNet Caffe* model and runs the Model Optimizer to convert the model to `.bin` and `.xml` Intermediate Representation (IR) files. The Inference Engine requires this model conversion so it can use the IR as input and achieve optimum performance on Intel hardware.

This verification script creates the directory `/home/<user>/inference_engine_samples/`, builds the [Image Classification Sample](../../inference-engine/samples/classification_sample_async/README.md) application and runs with the model IR and `car.png` image located in the `demo` directory. When the verification script completes, you will have the label and confidence for the top-10 categories:

![](../img/image_classification_script_output_lnx.png)

For a brief description of the Intermediate Representation `.bin` and `.xml` files, see [Configuring the Model Optimizer](#configure-the-model-optimizer).

This script is complete. Continue to the next section to run the Inference Pipeline verification script.

### Run the Inference Pipeline Verification Script

While still in `/opt/intel/openvino_2021/deployment_tools/demo/`, run the Inference Pipeline verification script:
   ```sh
   ./demo_security_barrier_camera.sh
   ```

This verification script downloads three pre-trained model IRs, builds the [Security Barrier Camera Demo](@ref omz_demos_security_barrier_camera_demo_README) application and runs it with the downloaded models and the `car_1.bmp` image from the `demo` directory to show an inference pipeline. The verification script uses vehicle recognition in which vehicle attributes build on each other to narrow in on a specific attribute.

First, an object is identified as a vehicle. This identification is used as input to the next model, which identifies specific vehicle attributes, including the license plate. Finally, the attributes identified as the license plate are used as input to the third model, which recognizes specific characters in the license plate.

When the verification script completes, you will see an image that displays the resulting frame with detections rendered as bounding boxes, and text:
![](../img/inference_pipeline_script_mac.png)

Close the image viewer screen to end the demo.

**Congratulations**, you have completed the Intel® Distribution of OpenVINO™ 2020.1 installation for macOS. To learn more about what you can do with the Intel® Distribution of OpenVINO™ toolkit, see the additional resources provided below.

## <a name="additional-NCS2-steps"></a>Steps for Intel® Neural Compute Stick 2

These steps are only required if you want to perform inference on Intel® Neural Compute Stick 2
powered by the Intel® Movidius™ Myriad™ X VPU. See also the
[Get Started page for Intel® Neural Compute Stick 2](https://software.intel.com/en-us/neural-compute-stick/get-started).

To perform inference on Intel® Neural Compute Stick 2, the `libusb` library is required. You can build it from the [source code](https://github.com/libusb/libusb) or install using the macOS package manager you prefer: [Homebrew*](https://brew.sh/), [MacPorts*](https://www.macports.org/) or other.

For example, to install the `libusb` library using Homebrew\*, use the following command:
```sh
brew install libusb
```

## <a name="Hello-World-Tutorial"></a>Hello World Tutorials

Visit the Intel Distribution of OpenVINO Toolkit [Inference Tutorials for Face Detection and Car Detection Exercises](https://github.com/intel-iot-devkit/inference-tutorials-generic/tree/openvino_toolkit_r3_0)


## Additional Resources

- To learn more about the verification applications, see `README.txt` in `/opt/intel/openvino_2021/deployment_tools/demo/`.

- For detailed description of the pre-trained models, go to the [Overview of OpenVINO toolkit Pre-Trained Models](@ref omz_models_intel_index) page.

- More information on [sample applications](../IE_DG/Samples_Overview.md).

- [Convert Your Caffe* Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_Caffe.md)

- [Convert Your TensorFlow* Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_TensorFlow.md)

- [Convert Your MXNet* Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_MxNet.md)

- [Convert Your ONNX* Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_ONNX.md)

- [Intel Distribution of OpenVINO Toolkit home page](https://software.intel.com/en-us/openvino-toolkit)

- [Intel Distribution of OpenVINO Toolkit documentation](https://docs.openvinotoolkit.org)
