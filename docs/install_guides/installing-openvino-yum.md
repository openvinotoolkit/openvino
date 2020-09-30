# Install Intel® Distribution of OpenVINO™ toolkit for Linux* Using YUM Repository {#openvino_docs_install_guides_installing_openvino_yum}

This guide provides installation steps for the Intel® Distribution of OpenVINO™ toolkit for Linux* distributed through the YUM repository.

> **IMPORTANT**: By downloading and using this container and the included software, you agree to the terms and conditions of the [software license agreements](https://software.intel.com/en-us/license/eula-for-intel-software-development-products). Please, review the content inside the `<openvino_install_root>/licensing` folder for more details.

> **NOTE**: Intel® Graphics Compute Runtime for OpenCL™ is not a part of OpenVINO™ YUM distribution. You can install it from the [Intel® Graphics Compute Runtime for OpenCL™ GitHub repo](https://github.com/intel/compute-runtime).

## Set up the Repository

> **NOTE:** You must be logged in as root to set up and install the repository.
<br>
Configure YUM with the OpenVINO repository to install OpenVINO. You have two options for this, using the `yum-config-manager` or manually by creating a text file and pointing YUM to the file. 

* **OPTION 1:** Import the `.repo` file using the `yum-config-manager`:
   1. `yum-utils` must be installed on your system.  If it’s not currently installed, run the command:
   ```sh
   sudo yum install yum-utils
   ```
   2. Add repository using the `yum-config-manager`:
   ```sh
   sudo yum-config-manager --add-repo https://yum.repos.intel.com/openvino/2020/setup/intel-openvino-2020.repo
   ```
   3. Import the gpg public key for the repository:
   ```sh
   sudo rpm --import https://yum.repos.intel.com/openvino/2020/setup/RPM-GPG-KEY-INTEL-OPENVINO-2020
   ```

* **OPTION 2:** Create the repository file manually:
   1. Navigate to the repository directory:
   ```sh
   cd /etc/yum.repos.d
   ```
   2. Edit the repo file:
   ```sh
   vi intel-openvino-2020.repo
   ```
   3. Append the following code:
   ```sh
   [intel-openvino-2020]
   name=Intel(R) Distribution of OpenVINO 2020
   baseurl=https://yum.repos.intel.com/openvino/2020
   enabled=1
   gpgcheck=1
   gpgkey=file:///etc/pki/rpm-gpg/RPM-GPG-KEY-INTEL-OPENVINO-2020
   ```
   4. Save and close the `intel-openvino-2020.repo` file.
   5. Import the gpg public key for the repository:
   ```sh
   sudo rpm --import https://yum.repos.intel.com/openvino/2020/setup/RPM-GPG-KEY-INTEL-OPENVINO-2020
   ```

### Verify that the new repo is properly setup
Run the following command:   
```sh
yum repolist | grep -i openvino
```

Results:
```sh
intel-openvino-2020 Intel(R) Distribution of OpenVINO 2020
```
  
### To list the available OpenVINO packages
Use the following command:
```sh
yum list intel-openvino*
```

---
  
## Install the runtime packages Using the YUM Package Manager

Intel® OpenVINO will be installed in: `/opt/intel/openvino_<VERSION>.<UPDATE>.<BUILD_NUM>`
<br>
A symlink will be created: `/opt/intel/openvino`

---

### To install the latest version
To install the full runtime version of the OpenVINO package:
```sh
sudo yum install intel-openvino-runtime-centos7
```

---

### To install a specific version
To install the full runtime version of the OpenVINO package:
```sh
sudo yum install intel-openvino-runtime-centos7-<VERSION>.<UPDATE>.<BUILD_NUM>
```

---

### To Uninstall a specific version

To uninstall a specific full runtime package:
```sh
sudo yum autoremove intel-openvino-runtime-centos<OS_VERSION>-<VERSION>.<UPDATE>.<BUILD_NUM>
```
**Additional Resources**

- Intel® Distribution of OpenVINO™ toolkit home page: [https://software.intel.com/en-us/openvino-toolkit](https://software.intel.com/en-us/openvino-toolkit)
- OpenVINO™ toolkit online documentation: [https://docs.openvinotoolkit.org](https://docs.openvinotoolkit.org)
- [Model Optimizer Developer Guide](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
- [Inference Engine Developer Guide](../IE_DG/Deep_Learning_Inference_Engine_DevGuide.md).
- For more information on Sample Applications, see the [Inference Engine Samples Overview](../IE_DG/Samples_Overview.md).
- For information on Inference Engine Tutorials, see the [Inference Tutorials](https://github.com/intel-iot-devkit/inference-tutorials-generic).
- For IoT Libraries & Code Samples see the [Intel® IoT Developer Kit](https://github.com/intel-iot-devkit).

