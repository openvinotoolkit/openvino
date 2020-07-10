# Configuration Guide for Intel® Distribution of OpenVINO™ toolkit 2020.3 and the Intel® Programmable Acceleration Card with Intel® Arria® 10 FPGA GX on CentOS or Ubuntu* {#openvino_docs_install_guides_PAC_Configure}

> **NOTE**: For previous versions, see [Configuration Guide for OpenVINO 2020.2](https://docs.openvinotoolkit.org/2020.2/_docs_install_guides_PAC_Configure.html), [Configuration Guide for OpenVINO 2019R1/2019R2/2019R3](https://docs.openvinotoolkit.org/2019_R3.1/_docs_install_guides_PAC_Configure_2019RX.html), [Configuration Guide for OpenVINO 2018R5](https://docs.openvinotoolkit.org/2019_R1/_docs_install_guides_PAC_Configure_2018R5.html).

## Get Started

The following describes the set-up of the Intel® Distribution of OpenVINO™ toolkit on CentOS* 7.4 or Ubuntu* 16.04, kernel 4.15.  This is based upon a completely fresh install of the OS with developer tools included.  Official Intel® documentation for the install process can be found in the following locations and it is highly recommended that these are read, especially for new users. This document serves as a guide, and in some cases, adds additional detail where necessary.

[Intel® Acceleration Stack for FPGAs Quick Start Guide](https://www.intel.com/content/dam/altera-www/global/en_US/pdfs/literature/ug/ug-qs-ias-v1-1.pdf)

[OpenCL™ on Intel® PAC Quick Start Guide](https://www.intel.com/content/dam/altera-www/global/en_US/pdfs/literature/ug/ug-qs-ias-opencl-a10-v1-1.pdf)

[Installing the Intel® Distribution of OpenVINO™ toolkit for Linux*](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html)

(Optional): Install NTFS support for transferring large installers if already downloaded on another machine.
```sh
sudo yum -y install epel-release
```
```sh
sudo yum -y install ntfs-3g
```

## Install Intel® PAC and the Intel® Programmable Acceleration Card Stack

1. Download version 1.2 of the Acceleration Stack for Runtime from the [Intel FPGA Acceleration Hub](https://www.altera.com/solutions/acceleration-hub/downloads.html). 
This downloads as `a10_gx_pac_ias_1_2_pv_rte_installer.tar.gz`. Let it download to `~/Downloads`.

2. Create a new directory to install to:
```sh
mkdir -p ~/tools/intelrtestack
```

3. Untar and launch the installer:
```sh
cd ~/Downloads
```
```sh
tar xf a10_gx_pac_ias_1_2_pv_rte_installer.tar.gz
```
```sh
cd a10_gx_pac_ias_1_2_pv_rte_installer
```
```sh
./setup.sh
```

4. Select **Y** to install OPAE and accept license and when asked, specify `/home/<user>/tools/intelrtestack` as the absolute install path. During the installation there should be a message stating the directory already exists as it was created in the first command above.  Select **Y** to install to this directory. If this message is not seen, it suggests that there was a typo when entering the install location.

5. Tools are installed to the following directories:
   * OpenCL™ Run-time Environment: `~/tools/intelrtestack/opencl_rte/aclrte-linux64`
   * Intel® Acceleration Stack for FPGAs: `~/tools/intelrtestack/a10_gx_pac_ias_1_2_pv`
  
7. Check the version of the FPGA Interface Manager firmware on the PAC board.
```sh
sudo fpgainfo fme
```

8. If the reported `Pr Interface Id` is not `69528db6-eb31-577a-8c36-68f9faa081f6` then follow the instructions in section 4 of the [Intel® Acceleration Stack for FPGAs Quick Start Guide](https://www.intel.com/content/dam/altera-www/global/en_US/pdfs/literature/ug/ug-qs-ias-v1-2.pdf) to update the FME.

9. Run the built in self-test to verify operation of the Acceleration Stack and Intel® PAC in a non-virtualized environment.
```sh
sudo sh -c "echo 20 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages"
```
```sh
source ~/tools/intelrtestack/init_env.sh
```
```sh
sudo fpgabist $OPAE_PLATFORM_ROOT/hw/samples/nlb_mode_3/bin/nlb_mode_3.gbs
```

## Verify the Intel® Acceleration Stack for FPGAs OpenCL™ BSP

1. Remove any previous FCD files that may be from previous installations of hardware in the `/opt/Intel/OpenCL/Boards/` directory:
```sh
cd /opt/Intel/OpenCL/Boards
sudo rm -rf *.fcd
```

2. Install `lsb_release` on your system if you are using CentOS:
```sh
sudo yum install redhat-lsb-core
``` 

3. Create an initialization script `~/init_openvino.sh` with the following content that can be run upon opening a new terminal or rebooting. This will source the script ran above as well as setting up the OpenCL™ environment.
```sh
source $HOME/tools/intelrtestack/init_env.sh
```
```sh
export CL_CONTEXT_COMPILER_MODE_ALTERA=3
```
```sh
export CL_CONTEXT_COMPILER_MODE_INTELFPGA=3
```
```sh
export INTELFPGAOCLSDKROOT="/opt/altera/aocl-pro-rte/aclrte-linux64"
```
```sh
export ALTERAOCLSDKROOT="$INTELFPGAOCLSDKROOT"
```
```sh
export AOCL_BOARD_PACKAGE_ROOT="$OPAE_PLATFORM_ROOT/opencl/opencl_bsp"
```
```sh
$AOCL_BOARD_PACKAGE_ROOT/linux64/libexec/setup_permissions.sh
```
```sh
source $INTELFPGAOCLSDKROOT/init_opencl.sh
```

4. Source the script:
```sh
source ~/init_openvino.sh
```

5. Some of the settings made in the child scripts need a reboot to take effect.  Reboot the machine and source the script again. Note that this script should be sourced each time a new terminal is opened for use with the Intel® Acceleration Stack for FPGAs and Intel® Distribution of OpenVINO™ toolkit.
```sh
source ~/init_openvino.sh
```

6. Install the OpenCL™ driver:
```sh
cd ~
```
```sh
sudo -E ./tools/intelrtestack/opencl_rte/aclrte-linux64/bin/aocl install
```
Select **Y** when asked to install the BSP. Note that the following warning can be safely ignored.
```sh
WARNING: install not implemented.  Please refer to DCP Quick Start User Guide.
```

7. Program the Intel® PAC board with a pre-compiled `.aocx` file (OpenCL™ based FPGA bitstream).
```sh
cd $OPAE_PLATFORM_ROOT/opencl
```
```sh 
aocl program acl0 hello_world.aocx
```

8. Build and run the Hello World application:
```sh
sudo tar xf exm_opencl_hello_world_x64_linux.tgz
```
```sh
sudo chmod -R a+w hello_world
```
```sh
cd hello_world
```
```sh
make
```
```sh
cp ../hello_world.aocx ./bin
```
```sh
./bin/host
```

## Add Intel® Distribution of OpenVINO™ toolkit with FPGA Support to Environment Variables

1. To run the Intel® Distribution of OpenVINO™ toolkit, add the last four commands to the `~/init_openvino.sh` script. The previous content is shown as well.
```sh
source $HOME/tools/intelrtestack/init_env.sh
export CL_CONTEXT_COMPILER_MODE_ALTERA=3
export CL_CONTEXT_COMPILER_MODE_INTELFPGA=3
export INTELFPGAOCLSDKROOT="/opt/altera/aocl-pro-rte/aclrte-linux64"
export ALTERAOCLSDKROOT="$INTELFPGAOCLSDKROOT"
export AOCL_BOARD_PACKAGE_ROOT="$OPAE_PLATFORM_ROOT/opencl/opencl_bsp"
$AOCL_BOARD_PACKAGE_ROOT/linux64/libexec/setup_permissions.sh
source $INTELFPGAOCLSDKROOT/init_opencl.sh
export IE_INSTALL="/opt/intel/openvino/deployment_tools"
source $IE_INSTALL/../bin/setupvars.sh
export PATH="$PATH:$HOME/inference_engine_samples_build/intel64/Release"
alias mo="python3.6 $IE_INSTALL/model_optimizer/mo.py"
```
For Ubuntu systems, it is recommended to use python3.5 above instead of python3.6.

2. Source the script
```sh
source ~/init_openvino.sh
```

## Program a Bitstream

The bitstream you program should correspond to the topology you want to deploy. In this section, you program a SqueezeNet bitstream and deploy the classification sample with a SqueezeNet model.

> **IMPORTANT**: Only use bitstreams from the installed version of the Intel® Distribution of OpenVINO™ toolkit. Bitstreams from older versions of the Intel® Distribution of OpenVINO™ toolkit are incompatible with later versions. For example, you cannot use the `1-0-1_RC_FP16_Generic` bitstream, when the Intel® Distribution of OpenVINO™ toolkit supports the `2-0-1_RC_FP16_Generic bitstream`.

There are different folders for each FPGA card type which were downloaded in the Intel® Distribution of OpenVINO™ toolkit package. 
For the Intel® Programmable Acceleration Card with Intel® Arria® 10 FPGA GX, the pre-trained bitstreams are in the `/opt/intel/openvino/bitstreams/a10_dcp_bitstreams` directory. This example uses a SqueezeNet bitstream with low precision for the classification sample.

Program the bitstream for Intel® Programmable Acceleration Card with Intel® Arria® 10 FPGA GX.
```sh
aocl program acl0 /opt/intel/openvino/bitstreams/a10_dcp_bitstreams/2020-3_RC_FP11_InceptionV1_SqueezeNet_VGG_YoloV3.aocx
```

## Use the Intel® Distribution of OpenVINO™ toolkit

1. Run inference with the Intel® Distribution of OpenVINO™ toolkit independent of the demo scripts using the SqueezeNet model that was download by the scripts. For convenience, copy the necessary files to a local directory. If the workstation has been rebooted or a new terminal is opened, source the script above first.
```sh
mkdir ~/openvino_test
```
```sh
cd ~/openvino_test
```
```sh
cp ~/openvino_models/models/public/squeezenet1.1/squeezenet1.1.* .
```
```sh
cp ~/openvino_models/ir/public/squeezenet1.1/FP16/squeezenet1.1.labels .
```

2. Note that the `squeezenet1.1.labels` file contains the classes used by ImageNet and is included here so that the inference results show text rather than classification numbers.  Convert the model with the [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).  Note that the command below uses the alias defined in the script above and is not referred to in other documentation.
```sh
mo --input_model squeezenet1.1.caffemodel
```

3. Now run Inference on the CPU using one of the built in Inference Engine samples:
```sh
classification_sample_async -m squeezenet1.1.xml -i $IE_INSTALL/demo/car.png
```

4. Add the `-d` option to run on FPGA:
```sh
classification_sample_async -m squeezenet1.1.xml -i $IE_INSTALL/demo/car.png -d HETERO:FPGA,CPU
```

Congratulations, You are done with the Intel® Distribution of OpenVINO™ toolkit installation for FPGA. To learn more about how the Intel® Distribution of OpenVINO™ toolkit works, the Hello World tutorial and are other resources are provided below.

## Hello World Face Detection Tutorial

Use the  [Intel® Distribution of OpenVINO™ toolkit with FPGA Hello World Face Detection Exercise](https://github.com/fritzboyle/openvino-with-fpga-hello-world-face-detection) to learn more about how the software and hardware work together.

## Additional Resources

Intel® Distribution of OpenVINO™ toolkit home page: [https://software.intel.com/en-us/openvino-toolkit](https://software.intel.com/en-us/openvino-toolkit)

Intel® Distribution of OpenVINO™ toolkit documentation: [https://docs.openvinotoolkit.org](https://docs.openvinotoolkit.org)

Inference Engine FPGA plugin documentation: [https://docs.openvinotoolkit.org/latest/_docs_IE_DG_supported_plugins_FPGA.html](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_supported_plugins_FPGA.html)
