# Get Started with OpenVINO™ Toolkit on Raspbian* OS {#openvino_docs_get_started_get_started_raspbian}

The OpenVINO™ toolkit optimizes and runs Deep Learning Neural Network models on Intel® hardware. This guide helps you get started with the OpenVINO™ toolkit you installed on Raspbian* OS.

In this guide, you will:
* Learn the OpenVINO™ inference workflow.
* Build and run sample code using detailed instructions.

## <a name="openvino-components"></a>OpenVINO™ Toolkit Components
On Raspbian* OS, the OpenVINO™ toolkit consists of the following components:
* **Inference Engine:** The software libraries that run inference against the Intermediate Representation (optimized model) to produce inference results.
* **MYRIAD Plugin:** The plugin developed for inference of neural networks on Intel® Neural Compute Stick 2.

> **NOTE**:
> * The OpenVINO™ package for Raspberry* does not include the [Model Optimizer](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md). To convert models to Intermediate Representation (IR), you need to install it separately to your host machine.
> * The package does not include the Open Model Zoo demo applications. You can download them separately from the [Open Models Zoo repository](https://github.com/openvinotoolkit/open_model_zoo).

In addition, [code samples](../IE_DG/Samples_Overview.md) are provided to help you get up and running with the toolkit.

## <a name="openvino-installation"></a>Intel® Distribution of OpenVINO™ Toolkit Directory Structure
This guide assumes you completed all Intel® Distribution of OpenVINO™ toolkit installation and configuration steps. If you have not yet installed and configured the toolkit, see [Install Intel® Distribution of OpenVINO™ toolkit for Raspbian*](../install_guides/installing-openvino-raspbian.md).

The OpenVINO toolkit for Raspbian* OS is distributed without installer. This document refers to the directory to which you unpacked the toolkit package as `<INSTALL_DIR>`.

The primary tools for deploying your models and applications are installed to the `<INSTALL_DIR>/deployment_tools` directory.
<details>
    <summary><strong>Click for the <code>deployment_tools</code> directory structure</strong></summary>
   

| Directory&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Description                                                                           |  
|:----------------------------------------|:--------------------------------------------------------------------------------------|
| `inference_engine/`                     | Inference Engine directory. Contains Inference Engine API binaries and source files, samples and extensions source files, and resources like hardware drivers.|
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`external/`     | Third-party dependencies and drivers.|
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`include/`      | Inference Engine header files. For API documentation, see the [Inference Engine API Reference](./annotated.html). |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`lib/`          | Inference Engine libraries.|
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`samples/`      | Inference Engine samples. Contains source code for C++ and Python* samples and build scripts. See the [Inference Engine Samples Overview](../IE_DG/Samples_Overview.md). |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`share/`        | CMake configuration files for linking with Inference Engine.|

</details>

## <a name="workflow-overview"></a>OpenVINO™ Workflow Overview

The OpenVINO™ workflow on Raspbian* OS is as follows:
1. **Get a pre-trained model** for your inference task. If you want to use your model for inference, the model must be converted to the `.bin` and `.xml` Intermediate Representation (IR) files, which are used as input by Inference Engine. On Raspberry PI, OpenVINO™ toolkit includes only the Inference Engine module. The Model Optimizer is not supported on this platform. To get the optimized models you can use one of the following options:
   
   * Download public and Intel's pre-trained models from the [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo) using [Model Downloader tool](@ref omz_tools_downloader).
    <br> For more information on pre-trained models, see [Pre-Trained Models Documentation](@ref omz_models_group_intel)
   
   * Convert a model using the Model Optimizer from a full installation of Intel® Distribution of OpenVINO™ toolkit on one of the supported platforms. Installation instructions are available:
     * [Installation Guide for macOS*](../install_guides/installing-openvino-macos.md)
     * [Installation Guide for Windows*](../install_guides/installing-openvino-windows.md)
     * [Installation Guide for Linux*](../install_guides/installing-openvino-linux.md)  
2. **Use the Inference Engine API in the application** to run inference against the Intermediate Representation (optimized model) and output inference results. The application can be an OpenVINO™ sample or your own application. 

## <a name="using-sample"></a>Build and Run Code Samples

Follow the steps below to run pre-trained Face Detection network using Inference Engine samples from the OpenVINO toolkit.

1. Create a samples build directory. This example uses a directory named `build`:
   ```sh
   mkdir build && cd build
   ```
2. Build the Object Detection Sample with the following command:
   ```sh
   cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=armv7-a" /opt/intel/openvino_2021/deployment_tools/inference_engine/samples/cpp
   make -j2 object_detection_sample_ssd
   ```
3. Download the pre-trained Face Detection model with the [Model Downloader tool](@ref omz_tools_downloader):
```sh
git clone --depth 1 https://github.com/openvinotoolkit/open_model_zoo
cd open_model_zoo/tools/downloader
python3 -m pip install -r requirements.in
python3 downloader.py --name face-detection-adas-0001 
```
4. Run the sample, specifying the model and path to the input image:
```sh
./armv7l/Release/object_detection_sample_ssd -m face-detection-adas-0001.xml -d MYRIAD -i <path_to_image>
```
The application outputs an image (`out_0.bmp`) with detected faced enclosed in rectangles.

## <a name="basic-guidelines-sample-application"></a>Basic Guidelines for Using Code Samples

Following are some basic guidelines for executing the OpenVINO™ workflow using the code samples:

1. Before using the OpenVINO™ samples, always set up the environment: 
```sh
source <INSTALL_DIR>/bin/setupvars.sh
``` 
2. Have the directory path for the following:
   - Code Sample binaries
   - Media: Video or image. Many sources are available from which you can download video media to use the code samples and demo applications, like https://videos.pexels.com and https://images.google.com.
   - Model in the IR format (.bin and .xml files).
## Additional Resources

Use these resources to learn more about the OpenVINO™ toolkit:

* [OpenVINO™ Release Notes](https://software.intel.com/en-us/articles/OpenVINO-RelNotes)
* [OpenVINO™ Toolkit Overview](../index.md)
* [Inference Engine Developer Guide](../IE_DG/Deep_Learning_Inference_Engine_DevGuide.md)
* [Model Optimizer Developer Guide](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
* [Inference Engine Samples Overview](../IE_DG/Samples_Overview.md)
* [Overview of OpenVINO™ Toolkit Pre-Trained Models](https://software.intel.com/en-us/openvino-toolkit/documentation/pretrained-models)
