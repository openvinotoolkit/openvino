# Install Intel® Distribution of OpenVINO™ toolkit for Linux Using APT Repository {#openvino_docs_install_guides_installing_openvino_apt}

This guide provides installation steps for Intel® Distribution of OpenVINO™ toolkit Runtime for Linux distributed through the APT repository.

From 2022.1 release, the OpenVINO Model Development Tools can only be installed via PyPI. If you want to develop or optimize your models with OpenVINO, see [Install OpenVINO Model Development Tools](installing-model-dev-tools.md) for detailed steps.

> **IMPORTANT**: By downloading and using this container and the included software, you agree to the terms and conditions of the [software license agreements](https://software.intel.com/content/dam/develop/external/us/en/documents/intel-openvino-license-agreements.pdf). Please review the content inside the `<INSTALL_DIR>/licensing` folder for more details.

## System requirements

The complete list of supported hardware is available in the [Release Notes](https://software.intel.com/content/www/us/en/develop/articles/openvino-relnotes.html#inpage-nav-8).

**Operating Systems**

- Ubuntu 18.04 long-term support (LTS), 64-bit
- Ubuntu 20.04 long-term support (LTS), 64-bit

## Install OpenVINO Runtime

### Step 1: Set up the OpenVINO™ Toolkit APT repository


1. Install the GPG key for the Repository

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

   * On Ubuntu 18
   ```sh
   echo "deb https://apt.repos.intel.com/openvino/2022/bionic all main" | sudo tee /etc/apt/sources.list.d/intel-openvino-2022.list
   ```
   
   * On Ubuntu 20
   ```sh
   echo "deb https://apt.repos.intel.com/openvino/2022/focal all main" | sudo tee /etc/apt/sources.list.d/intel-openvino-2022.list
   ```  

3.	Update the list of packages via the update command:
   ```sh
   sudo apt update
   ```       
   
4.	Verify that the APT repository is properly set up. Run the apt-cache command to see a list of all available OpenVINO packages and components:
   ```sh
   apt-cache search openvino
   ```   
   

### Step 2: Install OpenVINO Runtime using the APT Package Manager

Intel® OpenVINO™ Toolkit will be installed in: `/opt/intel/openvino_<VERSION>.<UPDATE>.<PATCH>`
    
A symlink will be created: `/opt/intel/openvino_<VERSION>`

#### To install the latest version

Run the following command:
```sh
sudo apt install openvino
```

#### To install a specific version


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

#### To check for installed packages and versions

Run the following command:
```sh
apt list --installed | grep openvino
```

#### To uninstall the latest version

Run the following command:
```sh
sudo apt autoremove openvino
```

#### To uninstall a specific version

Run the following command:
```sh
sudo apt autoremove openvino-<VERSION>.<UPDATE>.<PATCH>
```

### Step 3 (Optional): Install OpenCV* from APT

OpenCV is necessary to run demos from Open Model Zoo. Some OpenVINO toolkit samples and demos also use OpenCV as a dependency. OpenVINO provides a package to install OpenCV from APT:

#### To install the latest version of OpenCV

Run the following command:
```sh
sudo apt install openvino-opencv
```

#### To install a specific version of OpenCV

Run the following command:
```sh
sudo apt install openvino-opencv-<VERSION>.<UPDATE>.<PATCH>
```

### Step 4 (Optional): Install software dependencies

After you have installed OpenVINO Runtime, if you decided to [install OpenVINO Model Development Tools](installing-model-dev-tools.md), make sure that you install external software dependencies first. 

Refer to <a href="#install-external-dependencies">Install External Software Dependencies</a> for detailed steps.


## Additional resources

- Intel® Distribution of OpenVINO™ toolkit home page: <https://software.intel.com/en-us/openvino-toolkit>.
- OpenVINO™ toolkit online documentation: <https://docs.openvino.ai/>.
- [Model Optimizer Developer Guide](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
- [Inference Engine Developer Guide](../OV_Runtime_UG/Deep_Learning_Inference_Engine_DevGuide.md).
- For more information on Sample Applications, see the [Inference Engine Samples Overview](../OV_Runtime_UG/Samples_Overview.md).
- For IoT Libraries & Code Samples see the [Intel® IoT Developer Kit](https://github.com/intel-iot-devkit).
