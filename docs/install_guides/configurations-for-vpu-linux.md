# Configurations for Intel® Vision Accelerator Design with Intel® Movidius™ VPUs on Linux {#openvino_docs_configurations_vpu_linux}


For Intel® Vision Accelerator Design with Intel® Movidius™ VPUs, the following additional installation steps are required.

1. Set the environment variables:
```sh
source /opt/intel/openvino_2022/setupvars.sh
```
> **NOTE**: The `HDDL_INSTALL_DIR` variable is set to `<openvino_install_dir>/runtime/3rdparty/hddl`. If you installed the Intel® Distribution of OpenVINO™ to the default install directory, the `HDDL_INSTALL_DIR` was set to `/opt/intel/openvino_2022/runtime/3rdparty/hddl`.

2. Install dependencies:
```sh
${HDDL_INSTALL_DIR}/install_IVAD_VPU_dependencies.sh
```
Note, if the Linux kernel is updated after the installation, it is required to install drivers again: 
```sh
cd ${HDDL_INSTALL_DIR}/drivers
```
```sh
sudo ./setup.sh install
```
Now the dependencies are installed and you are ready to use the Intel® Vision Accelerator Design with Intel® Movidius™ with the Intel® Distribution of OpenVINO™ toolkit.

## Optional Steps

* For advanced configuration steps for your **IEI Mustang-V100-MX8-R10** accelerator, see [Intel® Movidius™ VPUs Setup Guide for Use with Intel® Distribution of OpenVINO™ toolkit](movidius-setup-guide.md). **IEI Mustang-V100-MX8-R11** accelerator doesn't require any additional steps. 

* After you've configured your Intel® Vision Accelerator Design with Intel® Movidius™ VPUs, see [Intel® Movidius™ VPUs Programming Guide for Use with Intel® Distribution of OpenVINO™ toolkit](movidius-programming-guide.md) to learn how to distribute a model across all 8 VPUs to maximize performance.
