# Install Intel® Distribution of OpenVINO™ toolkit for Linux* Using YUM Repository {#openvino_docs_install_guides_installing_openvino_yum}

This guide provides installation steps for the Intel® Distribution of OpenVINO™ toolkit for Linux* distributed through the YUM repository.

> **IMPORTANT**: By downloading and using this container and the included software, you agree to the terms and conditions of the [software license agreements](https://software.intel.com/content/dam/develop/external/us/en/documents/intel-openvino-license-agreements.pdf). Please, review the content inside the `<openvino_install_root>/licensing` folder for more details.

> **NOTE**: Intel® Graphics Compute Runtime for OpenCL™ is not a part of OpenVINO™ YUM distribution. You can install it from the [Intel® Graphics Compute Runtime for OpenCL™ GitHub repo](https://github.com/intel/compute-runtime).

> **NOTE**: Only runtime packages are available via the YUM repository.

## System Requirements

The complete list of supported hardware is available in the [Release Notes](https://software.intel.com/content/www/us/en/develop/articles/openvino-relnotes.html#inpage-nav-8).

**Operating Systems**

- CentOS 7.6, 64-bit

## Included with Runtime Package

The following components are installed with the OpenVINO runtime package:

| Component | Description|
|-----------|------------|
| [Inference Engine](../IE_DG/Deep_Learning_Inference_Engine_DevGuide.md)| The engine that runs a deep learning model. It includes a set of libraries for an easy inference integration into your applications. |
| [OpenCV*](https://docs.opencv.org/master/) | OpenCV* community version compiled for Intel® hardware. |
| Deep Learning Stream (DL Streamer) | Streaming analytics framework, based on GStreamer, for constructing graphs of media analytics components. For the DL Streamer documentation, see [DL Streamer Samples](@ref gst_samples_README), [API Reference](https://openvinotoolkit.github.io/dlstreamer_gst/), [Elements](https://github.com/openvinotoolkit/dlstreamer_gst/wiki/Elements), [Tutorial](https://github.com/openvinotoolkit/dlstreamer_gst/wiki/DL-Streamer-Tutorial). |

## Install Packages

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
   sudo yum-config-manager --add-repo https://yum.repos.intel.com/openvino/2021/setup/intel-openvino-2021.repo
   ```
   3. Import the gpg public key for the repository:
   ```sh
   sudo rpm --import https://yum.repos.intel.com/openvino/2021/setup/RPM-GPG-KEY-INTEL-OPENVINO-2021
   ```

* **OPTION 2:** Create the repository file manually:

   1. Create the YUM repo file in the /tmp directory as a normal user:
   ```sh
   tee > /tmp/openvino-2021.repo << EOF
   [intel-openvino-2021]
   name=Intel(R) Distribution of OpenVINO 2021
   baseurl=https://yum.repos.intel.com/openvino/2021
   enabled=1
   gpgcheck=1
   gpgkey=file:///etc/pki/rpm-gpg/RPM-GPG-KEY-INTEL-OPENVINO-2021
   EOF
   ```
   2. Move the newly created openvino-2021.repo file to the YUM configuration directory /etc/yum.repos.d:
   ```sh 
   sudo mv /tmp/openvino-2021.repo /etc/yum.repos.d
   ```
   3. Import the gpg public key for the repository:
   ```sh
   sudo rpm --import https://yum.repos.intel.com/openvino/2021/setup/RPM-GPG-KEY-INTEL-OPENVINO-2021
   ```

### Verify that the new repo is properly setup
Run the following command:   
```sh
yum repolist | grep -i openvino
```

Results:
```sh
intel-openvino-2021 Intel(R) Distribution of OpenVINO 2021
```
  
### To list available OpenVINO packages
Use the following command:
```sh
yum list intel-openvino*
```

---
  
## Install Runtime Packages Using the YUM Package Manager

Intel® OpenVINO will be installed in: `/opt/intel/openvino_<VERSION>.<UPDATE>.<BUILD_NUM>`
<br>
A symlink will be created: `/opt/intel/openvino_<VERSION>`

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

#### Examples

```sh
sudo yum install intel-openvino-runtime-centos7-2021.3.394
 ```

---

### To check for installed packages and version

To check a specific version of an OpenVINO package: 
```sh
yum list installed intel-openvino*
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
- For IoT Libraries & Code Samples see the [Intel® IoT Developer Kit](https://github.com/intel-iot-devkit).

