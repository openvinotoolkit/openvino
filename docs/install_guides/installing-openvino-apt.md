# Install Intel® Distribution of OpenVINO™ toolkit for Linux* Using APT Repository {#openvino_docs_install_guides_installing_openvino_apt}

This guide provides installation steps for Intel® Distribution of OpenVINO™ toolkit for Linux* distributed through the APT repository.

> **IMPORTANT**: By downloading and using this container and the included software, you agree to the terms and conditions of the [software license agreements](https://software.intel.com/en-us/license/eula-for-intel-software-development-products). Please, review the content inside the `<openvino_install_root>/licensing` folder for more details.

> **NOTE**: Intel® Graphics Compute Runtime for OpenCL™ is not a part of OpenVINO™ APT distribution. You can install it from the [Intel® Graphics Compute Runtime for OpenCL™ GitHub repo](https://github.com/intel/compute-runtime). 

## Set up the Repository
### Install the GPG key for the repository

1. Download the public key from [https://apt.repos.intel.com/openvino/2020/GPG-PUB-KEY-INTEL-OPENVINO-2020](https://apt.repos.intel.com/openvino/2020/GPG-PUB-KEY-INTEL-OPENVINO-2020) and save it to a file. 
2. Add this key to the system keyring:
```sh
sudo apt-key add <PATH_TO_DOWNLOADED_GPG_KEY>
```
3. Check the list of APT keys running the following command:
```sh
sudo apt-key list
```

### Add the APT Repository

Run the following command:
```sh
echo "deb https://apt.repos.intel.com/openvino/2020 all main" | sudo tee /etc/apt/sources.list.d/intel-openvino-2020.list
```

### Update the list of packages

Run the `update` command:
```sh
sudo apt update
```
There are full release Runtime and Developer packages, and also some available components.

**Runtime Packages**
- Ubuntu 18.04:  `intel-openvino-runtime-ubuntu18`
- Ubuntu 16.04:  `intel-openvino-runtime-ubuntu16`

**Developer Packages**
- Ubuntu 18.04:  `intel-openvino-dev-ubuntu18`
- Ubuntu 16.04:  `intel-openvino-dev-ubuntu16`

### Get the list of available packages

Run the `apt-cache` command to see a list of all available OpenVINO packages and components:
```sh
apt-cache search openvino
```

#### Examples

* **Runtime Packages**
  
  On Ubuntu 18.04:
  ```sh
  sudo apt-cache search intel-openvino-runtime-ubuntu18
  ```
  On Ubuntu 16.04:
  ```sh
  sudo apt-cache search intel-openvino-runtime-ubuntu16
  ```
* **Developer Packages**

  On Ubuntu 18.04:
  ```sh
  sudo apt-cache search intel-openvino-dev-ubuntu18
  ```
  On Ubuntu 16.04:
  ```sh
  sudo apt-cache search intel-openvino-dev-ubuntu16
  ```


## Install the runtime or developer packages using the APT Package Manager
Intel® OpenVINO will be installed in: `/opt/intel/openvino_<VERSION>.<UPDATE>.<BUILD_NUM>`

A symlink will be created: `/opt/intel/openvino`

---
### To Install a specific version

To get a list of OpenVINO packages available for installation:

```sh
sudo apt-cache search intel-openvino-runtime-ubuntu18
```

To install a specific version of an OpenVINO package:
```sh
sudo apt install intel-openvino-<PACKAGE_TYPE>-ubuntu<OS_VERSION>-<VERSION>.<UPDATE>.<BUILD_NUM>
```

#### Examples
* **Runtime Package**

  On Ubuntu 18.04:
  ```sh
  sudo apt install intel-openvino-runtime-ubuntu18-2020.1.023
  ```
  On Ubuntu 16.04:
  ```sh
  sudo apt install intel-openvino-runtime-ubuntu16-2020.1.023
  ```
* **Developer Package**<br>
  On Ubuntu 18.04:
  ```sh
  sudo apt install intel-openvino-dev-ubuntu18-2020.1.023 
  ```
  On Ubuntu 16.04:
  ```sh
  sudo apt install intel-openvino-dev-ubuntu16-2020.1.023
  ```

---
### To Uninstall a specific version

To uninstall a specific full runtime package:
```sh
sudo apt autoremove intel-openvino-<PACKAGE_TYPE>-ubuntu<OS_VERSION>-<VERSION>.<UPDATE>.<BUILD_NUM>
```


**Additional Resources**

- Intel® Distribution of OpenVINO™ toolkit home page: [https://software.intel.com/en-us/openvino-toolkit](https://software.intel.com/en-us/openvino-toolkit)
- OpenVINO™ toolkit online documentation: [https://docs.openvinotoolkit.org](https://docs.openvinotoolkit.org)
- [Model Optimizer Developer Guide](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).
- [Inference Engine Developer Guide](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_Deep_Learning_Inference_Engine_DevGuide.html)
- For more information on Sample Applications, see the [Inference Engine Samples Overview](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_Samples_Overview.html).
- For information on Inference Engine Tutorials, see the [Inference Tutorials](https://github.com/intel-iot-devkit/inference-tutorials-generic).
- For IoT Libraries & Code Samples see the [Intel® IoT Developer Kit](https://github.com/intel-iot-devkit).

