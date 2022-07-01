# Configurations for Intel® Vision Accelerator Design with Intel® Movidius™ VPUs  {#openvino_docs_install_guides_installing_openvino_ivad_vpu}

@sphinxdirective

.. _vpu guide:

.. toctree::
   :maxdepth: 2
   :hidden:

   IEI Mustang-V100-MX8-R10 Card <openvino_docs_install_guides_movidius_setup_guide>
        
@endsphinxdirective


The steps in this guide are only required if you want to perform inference on Intel® Vision Accelerator Design with Intel® Movidius™ VPUs with OpenVINO™ on Linux or Windows.

For troubleshooting issues, please see the [Troubleshooting Guide](troubleshooting.md) for more information.

## Linux

For Intel® Vision Accelerator Design with Intel® Movidius™ VPUs, the following additional installation steps are required.

> **NOTE**: If you installed the Intel® Distribution of OpenVINO™ toolkit to the non-default install directory, replace `/opt/intel` with the directory in which you installed the software.

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

### Optional Steps

For advanced configuration steps for your **IEI Mustang-V100-MX8-R10** accelerator, see [Configurations for IEI Mustang-V100-MX8-R10 card](configurations-for-iei-card.md). **IEI Mustang-V100-MX8-R11** accelerator doesn't require any additional steps. 

@sphinxdirective

.. _vpu guide windows:

@endsphinxdirective

## Windows

To enable inference on Intel® Vision Accelerator Design with Intel® Movidius™ VPUs, the following additional installation steps are required:

  1. Download and install <a href="https://www.microsoft.com/en-us/download/details.aspx?id=48145">Visual C++ Redistributable for Visual Studio 2017</a>
  2. Check with a support engineer if your Intel® Vision Accelerator Design with Intel® Movidius™ VPUs card requires SMBUS connection to PCIe slot (most unlikely). Install the SMBUS driver only if confirmed (by default, it's not required):
      1. Go to the `<INSTALL_DIR>\runtime\3rdparty\hddl\drivers\SMBusDriver` directory, where `<INSTALL_DIR>` is the directory in which the Intel® Distribution of OpenVINO™ toolkit is installed.
      2. Right click on the `hddlsmbus.inf` file and choose **Install** from the pop up menu.

You are done installing your device driver and are ready to use your Intel® Vision Accelerator Design with Intel® Movidius™ VPUs.

For advanced configuration steps for your IEI Mustang-V100-MX8 accelerator, see [Configurations for IEI Mustang-V100-MX8-R10 card](configurations-for-iei-card.md).

After configuration is done, you are ready to go to <a href="openvino_docs_install_guides_installing_openvino_windows.html#get-started">Start Using the Toolkit</a> section to learn the basic OpenVINO™ toolkit workflow and run code samples and demo applications.
