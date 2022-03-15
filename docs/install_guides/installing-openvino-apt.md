# Install Intel® Distribution of OpenVINO™ Toolkit for Linux Using APT Repository {#openvino_docs_install_guides_installing_openvino_apt}

This guide provides installation steps for Intel® Distribution of OpenVINO™ toolkit for Linux distributed through the APT repository.

> **NOTE**: From the 2022.1 release, the OpenVINO™ Development Tools can only be installed via PyPI. If you want to develop or optimize your models with OpenVINO, see [Install OpenVINO Development Tools](installing-model-dev-tools.md) for detailed steps.

> **IMPORTANT**: By downloading and using this container and the included software, you agree to the terms and conditions of the [software license agreements](https://software.intel.com/content/dam/develop/external/us/en/documents/intel-openvino-license-agreements.pdf). Please review the content inside the `<INSTALL_DIR>/licensing` folder for more details.

## System Requirements

The complete list of supported hardware is available in the [Release Notes](https://software.intel.com/content/www/us/en/develop/articles/openvino-relnotes.html).

**Operating Systems**

- Ubuntu 18.04 long-term support (LTS), 64-bit
- Ubuntu 20.04 long-term support (LTS), 64-bit

## Install OpenVINO Runtime

### Step 1: Set Up the OpenVINO Toolkit APT Repository

1. Install the GPG key for the repository

    a. Download the [GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB](https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB). You can also use the following command:
      ```sh
      wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
      ```    
    b. Add this key to the system keyring:
      ```sh
      sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
      ```
      > **NOTE**: You might need to install GnuPG: `sudo apt-get install gnupg`   

2.	Add the repository via the following command:
    @sphinxdirective

    .. tab:: Ubuntu 18

        .. code-block:: sh

            echo "deb https://apt.repos.intel.com/openvino/2022 bionic main" | sudo tee /etc/apt/sources.list.d/intel-openvino-2022.list

    .. tab:: Ubuntu 20

        .. code-block:: sh

            echo "deb https://apt.repos.intel.com/openvino/2022 focal main" | sudo tee /etc/apt/sources.list.d/intel-openvino-2022.list

    @endsphinxdirective

3.	Update the list of packages via the update command:
   ```sh
   sudo apt update
   ```       

4.	Verify that the APT repository is properly set up. Run the apt-cache command to see a list of all available OpenVINO packages and components:
   ```sh
   apt-cache search openvino
   ```   


### Step 2: Install OpenVINO Runtime Using the APT Package Manager

OpenVINO will be installed in: `/opt/intel/openvino_<VERSION>.<UPDATE>.<PATCH>`
    
A symlink will be created: `/opt/intel/openvino_<VERSION>`

#### To Install the Latest Version

Run the following command:
```sh
sudo apt install openvino
```

#### To Install a Specific Version


1.	Get a list of OpenVINO packages available for installation:
   ```sh
   sudo apt-cache search openvino
   ```
2.	Install a specific version of an OpenVINO package:
   ```sh
   sudo apt install openvino-<VERSION>.<UPDATE>.<PATCH>
   ```
    For example:
   ```sh
   sudo apt install openvino-2022.1.0
   ```

#### To Check for Installed Packages and Versions

Run the following command:
```sh
apt list --installed | grep openvino
```

#### To Uninstall the Latest Version

Run the following command:
```sh
sudo apt autoremove openvino
```

#### To Uninstall a Specific Version

Run the following command:
```sh
sudo apt autoremove openvino-<VERSION>.<UPDATE>.<PATCH>
```

### Step 3 (Optional): Install OpenCV from APT

OpenCV is necessary to run C++ demos from Open Model Zoo. Some OpenVINO samples can also extend their capabilities when compiled with OpenCV as a dependency. OpenVINO provides a package to install OpenCV from APT:

#### To Install the Latest Version of OpenCV

Run the following command:
```sh
sudo apt install openvino-opencv
```

#### To Install a Specific Version of OpenCV

Run the following command:
```sh
sudo apt install openvino-opencv-<VERSION>.<UPDATE>.<PATCH>
```

### Step 4 (Optional): Install Software Dependencies

After you have installed OpenVINO Runtime, if you decided to [install OpenVINO Development Tools](installing-model-dev-tools.md), make sure that you install external software dependencies first. 

Refer to <a href="installing-openvino-linux.md#install-external-dependencies">Install External Software Dependencies</a> for detailed steps.


## Configurations for Non-CPU Devices

If you are using Intel® Processor Graphics, Intel® Vision Accelerator Design with Intel® Movidius™ VPUs or Intel® Neural Compute Stick 2, please follow the configuration steps in [Configurations for GPU](configurations-for-intel-gpu.md), [Configurations for VPU](installing-openvino-config-ivad-vpu.md) or [Configurations for NCS2](configurations-for-ncs2.md) accordingly.


## Additional Resources

- Intel® Distribution of OpenVINO™ toolkit home page: <https://software.intel.com/en-us/openvino-toolkit>.
- OpenVINO™ toolkit online documentation: <https://docs.openvino.ai/>.
- [Model Optimizer Developer Guide](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
- [OpenVINO Runtime User Guide](../OV_Runtime_UG/OpenVINO_Runtime_User_Guide).
- For more information on Sample Applications, see the [OpenVINO Samples Overview](../OV_Runtime_UG/Samples_Overview.md).
- For IoT Libraries & Code Samples see the [Intel® IoT Developer Kit](https://github.com/intel-iot-devkit).
