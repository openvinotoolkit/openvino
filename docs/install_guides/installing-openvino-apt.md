# Install Intel® Distribution of OpenVINO™ toolkit for Linux* Using APT Repository {#openvino_docs_install_guides_installing_openvino_apt}

This guide provides installation steps for Intel® Distribution of OpenVINO™ toolkit for Linux* distributed through the APT repository.

> **IMPORTANT**: By downloading and using this container and the included software, you agree to the terms and conditions of the [software license agreements](https://software.intel.com/content/dam/develop/external/us/en/documents/intel-openvino-license-agreements.pdf). Please, review the content inside the `<openvino_install_root>/licensing` folder for more details.

> **NOTE**: Intel® Graphics Compute Runtime for OpenCL™ is not a part of OpenVINO™ APT distribution. You can install it from the [Intel® Graphics Compute Runtime for OpenCL™ GitHub repo](https://github.com/intel/compute-runtime).

## System Requirements

The complete list of supported hardware is available in the [Release Notes](https://software.intel.com/content/www/us/en/develop/articles/openvino-relnotes.html#inpage-nav-8).

**Operating Systems**

- Ubuntu 18.04.x long-term support (LTS), 64-bit
- Ubuntu 20.04.0 long-term support (LTS), 64-bit

## Included with Runtime Package

The following components are installed with the OpenVINO runtime package:

| Component | Description|
|-----------|------------|
| [Inference Engine](../IE_DG/Deep_Learning_Inference_Engine_DevGuide.md)| The engine that runs a deep learning model. It includes a set of libraries for an easy inference integration into your applications. |
| [OpenCV*](https://docs.opencv.org/master/) | OpenCV* community version compiled for Intel® hardware. |
| Deep Learning Streamer (DL Streamer) | Streaming analytics framework, based on GStreamer, for constructing graphs of media analytics components. For the DL Streamer documentation, see [DL Streamer Samples](@ref gst_samples_README), [API Reference](https://openvinotoolkit.github.io/dlstreamer_gst/), [Elements](https://github.com/openvinotoolkit/dlstreamer_gst/wiki/Elements), [Tutorial](https://github.com/openvinotoolkit/dlstreamer_gst/wiki/DL-Streamer-Tutorial). |

## Included with Developer Package

The following components are installed with the OpenVINO developer package:

| Component | Description|
|-----------|------------|
| [Model Optimizer](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md) | This tool imports, converts, and optimizes models that were trained in popular frameworks to a format usable by Intel tools, especially the Inference Engine. <br>Popular frameworks include Caffe\*, TensorFlow\*, MXNet\*, and ONNX\*. |
| [Inference Engine](../IE_DG/Deep_Learning_Inference_Engine_DevGuide.md) | The engine that runs a deep learning model. It includes a set of libraries for an easy inference integration into your applications.|
| [OpenCV*](https://docs.opencv.org/master/) | OpenCV\* community version compiled for Intel® hardware |
| [Sample Applications](../IE_DG/Samples_Overview.md)           | A set of simple console applications demonstrating how to use the Inference Engine in your applications. |
| [Demo Applications](@ref omz_demos) | A set of console applications that demonstrate how you can use the Inference Engine in your applications to solve specific use cases. |
| Additional Tools                                   | A set of tools to work with your models including [Accuracy Checker utility](@ref omz_tools_accuracy_checker), [Post-Training Optimization Tool Guide](@ref pot_README), [Model Downloader](@ref omz_tools_downloader) and other  |
| [Documentation for Pre-Trained Models ](@ref omz_models_group_intel)                                   | Documentation for the pre-trained models available in the [Open Model Zoo repo](https://github.com/openvinotoolkit/open_model_zoo).  |
| Deep Learning Streamer (DL Streamer)   | Streaming analytics framework, based on GStreamer\*, for constructing graphs of media analytics components. For the DL Streamer documentation, see [DL Streamer Samples](@ref gst_samples_README), [API Reference](https://openvinotoolkit.github.io/dlstreamer_gst/), [Elements](https://github.com/openvinotoolkit/dlstreamer_gst/wiki/Elements), [Tutorial](https://github.com/openvinotoolkit/dlstreamer_gst/wiki/DL-Streamer-Tutorial). |


## Install Packages

### Set up the OpenVINO™ Toolkit APT Repository

#### Install the GPG key for the Repository

1. Download the public key from [https://apt.repos.intel.com/openvino/2021/GPG-PUB-KEY-INTEL-OPENVINO-2021](https://apt.repos.intel.com/openvino/2021/GPG-PUB-KEY-INTEL-OPENVINO-2021) and save it to a file. 
2. Add this key to the system keyring:
```sh
sudo apt-key add <PATH_TO_DOWNLOADED_GPG_KEY>
```
> **NOTE**: You might need to install GnuPG: `sudo apt-get install gnupg`

3. Check the list of APT keys running the following command:
```sh
sudo apt-key list
```

#### Add the Repository

Run the following command:
```sh
echo "deb https://apt.repos.intel.com/openvino/2021 all main" | sudo tee /etc/apt/sources.list.d/intel-openvino-2021.list
```

#### Update the List of Packages

Run the `update` command:
```sh
sudo apt update
```

#### Verify that the APT Repository is Properly Set Up

Run the `apt-cache` command to see a list of all available OpenVINO packages and components:
```sh
apt-cache search openvino
```
See the example commands below:

* **Runtime Packages**
  
  On Ubuntu 18.04:
  ```sh
  sudo apt-cache search intel-openvino-runtime-ubuntu18
  ```
  On Ubuntu 20.04:
  ```sh
  sudo apt-cache search intel-openvino-runtime-ubuntu20
  ```
* **Developer Packages**

  On Ubuntu 18.04:
  ```sh
  sudo apt-cache search intel-openvino-dev-ubuntu18
  ```
  On Ubuntu 20.04:
  ```sh
  sudo apt-cache search intel-openvino-dev-ubuntu20
  ```

### Install Runtime or Developer Packages using the APT Package Manager
Intel® OpenVINO™ Toolkit will be installed in: `/opt/intel/openvino_<VERSION>.<UPDATE>.<BUILD_NUM>`

A symlink will be created: `/opt/intel/openvino_<VERSION>`

#### To Install a Specific Version

1. Get a list of OpenVINO packages available for installation:
```sh
sudo apt-cache search intel-openvino-runtime-ubuntu18
```
2. Install a specific version of an OpenVINO package:
```sh
sudo apt install intel-openvino-<PACKAGE_TYPE>-ubuntu<OS_VERSION>-<VERSION>.<UPDATE>.<BUILD_NUM>
```
See the example commands below:
* **Runtime Package**<br>
  On Ubuntu 18.04:
  ```sh
  sudo apt install intel-openvino-runtime-ubuntu18-2021.1.105
  ```
  On Ubuntu 20.04:
  ```sh
  sudo apt install intel-openvino-runtime-ubuntu20-2021.1.105
  ```
* **Developer Package**<br>
  On Ubuntu 18.04:
  ```sh
  sudo apt install intel-openvino-dev-ubuntu18-2021.1.105 
  ```
  On Ubuntu 20.04:
  ```sh
  sudo apt install intel-openvino-dev-ubuntu20-2021.1.105
  ```

#### To check for Installed Packages and Versions

To get a list of installed OpenVINO packages:

```sh
apt list --installed | grep openvino
```

#### To Uninstall a Specific Version

To uninstall a specific package:
```sh
sudo apt autoremove intel-openvino-<PACKAGE_TYPE>-ubuntu<OS_VERSION>-<VERSION>.<UPDATE>.<BUILD_NUM>
```


**Additional Resources**

- Intel® Distribution of OpenVINO™ toolkit home page: [https://software.intel.com/en-us/openvino-toolkit](https://software.intel.com/en-us/openvino-toolkit).
- OpenVINO™ toolkit online documentation: [https://docs.openvinotoolkit.org](https://docs.openvinotoolkit.org).
- [Model Optimizer Developer Guide](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
- [Inference Engine Developer Guide](../IE_DG/Deep_Learning_Inference_Engine_DevGuide.md).
- For more information on Sample Applications, see the [Inference Engine Samples Overview](../IE_DG/Samples_Overview.md).
- For IoT Libraries & Code Samples see the [Intel® IoT Developer Kit](https://github.com/intel-iot-devkit).

