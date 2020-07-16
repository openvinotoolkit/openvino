# Configuration Guide for the Intel® Distribution of OpenVINO™ toolkit 2018R5 and the Intel® Vision Accelerator Design with an Intel® Arria® 10 FPGA (IEI's Mustang-F100-A10) on Linux* {#openvino_docs_install_guides_VisionAcceleratorFPGA_Configure_2018R5}

> **NOTES:**
>  * For a first-time installation, use all steps.
>  * Use steps 1 and 2 only after receiving a new FPGA card.
>  * Repeat steps 2-5 when installing a new version of the Intel® Distribution of OpenVINO™ toolkit.
>  * Use steps 3-5 when a Neural Network topology used by an Intel® Distribution of OpenVINO™ toolkit application changes.

## 1. Configure and Install the Intel® Vision Accelerator Design with an Intel® Arria® 10 FPGA

1. Download `fpga_support_files.tgz` from the [Intel Registration Center](http://registrationcenter-download.intel.com/akdlm/irc_nas/12954/fpga_support_files.tgz). The files in this `.tgz` archive are required to ensure your FPGA card and the Intel® Distribution of OpenVINO™ toolkit work correctly.

2. Go to the directory where you downloaded the `fpga_support_files.tgz` archive.

3. Unpack the `.tgz` file:
```sh
tar -xvzf fpga_support_files.tgz
```
A directory named `fpga_support_files` is created.

4. Go to the `fpga_support_files` directory:
```sh
cd fpga_support_files
```
	
5. Source `setup_env.sh` to set your environment variables:
```sh
source /home/<user>/Downloads/fpga_support_files/setup_env.sh
```
	
6. Configure the FPGA Driver Blacklist:
```sh
sudo mv config/blacklist-altera-cvp.conf /etc/modprobe.d
```

7. Switch to superuser:
```sh
sudo su
```
	
8. Use the `setup_env.sh` script from `fpga_support_files.tgz` to set your environment variables:
```sh
source /home/<user>/Downloads/fpga_support_files/setup_env.sh
```
	
9. Change directory to `Downloads/fpga_support_files/`:
```sh
cd /home/<user>/Downloads/fpga_support_files/
```
	
10. Run the FPGA dependencies script, which allows OpenCL to support Ubuntu* and recent kernels:
```sh
./install_openvino_fpga_dependencies.sh
```

11. When asked, select the FPGA card, Intel® GPU, and Intel® Movidius™ Neural Compute Stick, then you can install the correct dependencies.

12. If you installed the 4.14 kernel as part of the installation script, you will need to reboot the machine and select the new kernel in the Ubuntu (grub) boot menu. You will also need to rerun `setup_env.sh` to set up your environmental variables again.
		
13. Install OpenCL™ devices. Enter **Y** when prompted to install:
```sh
aocl install
```
	
14. Reboot the machine:
```sh
reboot
```
	
15. Use the `setup_env.sh` script from `fpga_support_files.tgz` to set your environment variables:
```sh
source /home/<user>/Downloads/fpga_support_files/setup_env.sh
```
	
16. Run `aocl diagnose`:
```sh
aocl diagnose
```
Your screen displays `DIAGNOSTIC_PASSED`.

## 2. Set Up the Intel® Vision Accelerator Design with Intel® Arria® 10 FPGA for 2018R5

For the 2018R5 release, the Intel® Distribution of OpenVINO™ toolkit introduced a new board support package (BSP) `a10_1150_sg1` for the Intel® Vision Accelerator Design with an Intel® Arria®  10 FPGA, which is included into the `fpga_support_files.tgz` archive. To program the bitstreams for the Intel® Distribution of OpenVINO™ toolkit R5, you need to program the BSP into the board using the USB blaster.

> **NOTE**: These steps apply only if you update to the Intel® Distribution of OpenVINO™ toolkit R5. Otherwise, you can skip them.

1. Go to the `config` folder of the `fpga_support_files` directory where the `a10_1150_sg1` is located:
```sh
cd /home/<user>/Downloads/fpga_support_files/config/
```

2. Copy the `a10_1150_sg1` folder to the `board` directory:
```sh
sudo cp -rf a10_1150_sg1 /opt/altera/aocl-pro-rte/aclrte-linux64/board/
```
	
3. Convert the BSP files from DOS to UNIX:
```sh
sudo chmod +x a10_1150_sg1
find a10_1150_sg1 -type f -print0 | xargs -0 dos2unix
```
	
4. Set up the USB Blaster:
		
    1. Connect the cable between the board and the host system. Use the letter codes in the diagram below for the connection points:
				
    2. Connect the B end of the cable to point B on the board.

    3. Connect the F end of the cable to point F on the FPGA download cable.
				
    4. From point F end of the cable to point F on the FPGA download cable, the connection is as shown:
![](../img/VisionAcceleratorJTAG.png)

5. Source the `setup_env.sh` script from the `fpga_support_files` to set up the environment variables:
```sh
source /home/<user>/Downloads/fpga_support_files/setup_env.sh
```
	
6. Update the Intel® FPGA Download Cable rules to program the board without root permissions and to flash the initialization bitstreams so that the Intel® FPGA Download Cable can communicate with the board: 
```sh
sudo cp config/51-usbblaster.rules /etc/udev/rules.d
```
	
7. Load the USB rules:
```sh
sudo udevadm control --reload-rules && udevadm trigger
```
	
8. Unplug and re-plug the Intel® FPGA Download Cable to enable JTAG connection.

9. Run `jtagconfig` to ensure that your Intel FPGA Download Cable driver is ready to use:
```sh
jtagconfig
```
Your output is similar to:
```sh
1) USB-Blaster [1-6]
02E660DD   10AX115H1(.|E2|ES)/10AX115H2/.. 
```

10. Download [Intel® Quartus® Prime Software Lite Edition 17.1](http://fpgasoftware.intel.com/17.1/?edition=lite). Install the Intel® Quartus® Prime Software Lite to the `/home/<user>/intelFPGA/17.1` directory.
> **NOTE**: You will need the complete the Intel® Quartus® Prime Software Lite version when you want to program the `boardtest_1ddr_top.aocx` into the flash for permanent availability.
	
11. Export the Intel® Quartus® Prime Software Lite environment variable:
```sh
export QUARTUS_ROOTDIR=/home/<user>/intelFPGA/17.1/quartus
```
	
12. Use `jtagconfig` to slow the clock:
```sh
jtagconfig --setparam 1 JtagClock 6M
```
	
13. (OPTIONAL) Confirm the clock is set to 6M:
```sh
jtagconfig --getparam 1 JtagClock
```
You should see the following:
```sh
6M
```

14. Go to `/opt/altera/aocl-pro-rte/aclrte-linux64/board/a10_1150_sg1/bringup`, where `boardtest_1ddr_top.aocx `is located:
```sh
cd /opt/altera/aocl-pro-rte/aclrte-linux64/board/a10_1150_sg1/bringup
```
	
15. Program the `boardtest_1ddr_top.aocx` file to the flash to be made permanently available even after power cycle:
```sh
aocl flash acl0 boardtest_1ddr_top.aocx
```
> **NOTE**: You will need the USB Blaster for this.

16. Reboot the host system.

17. Check if the host system recognizes the Intel® Vision Accelerator Design with Intel® Arria® 10 FPGA board. Confirm you can detect the PCIe card:
```sh
lspci | grep -i Altera
```
Your output is similar to:
```sh
01:00.0 Processing accelerators: Altera Corporation Device 2494 (rev 01)
```

18. Source the `setup_env.sh` script from the `fpga_support_files` directory to setup the environment variables:
```sh
source /home/<user>/Downloads/fpga_support_file/setup_env.sh
```
	
19. Uninstall the previous BSP before installing the OpenCL drivers for the R5 BSP:
```sh
aocl uninstall /opt/altera/aocl-pro-rte/aclrte-linux64/board/<BSP_package>/
```
	
20. Export and source the environment script:
```sh
export AOCL_BOARD_PACKAGE_ROOT=/opt/altera/aocl-pro-rte/aclrte-linux64/board/a10_1150_sg1
```
```sh
source /opt/altera/aocl-pro-rte/aclrte-linux64/init_opencl.sh
```

21. Install OpenCL™ devices:
```sh
aocl install
```
	
22. Run the `diagnose` command:
```sh
aocl diagnose
```
You should see `DIAGNOSTIC_PASSED` before proceeding to the next steps.
	
## 3. Program a Bitstream

The bitstream you program should correspond to the topology you want to deploy. In this section, you program a SqueezeNet bitstream and deploy the classification sample with a SqueezeNet model that you used the Model Optimizer to convert in the steps before.

> **IMPORTANT**: Only use bitstreams from the installed version of the Intel® Distribution of OpenVINO™ toolkit. Bitstreams from older versions of the Intel® Distribution of OpenVINO™ toolkit are incompatible with later versions of the Intel® Distribution of OpenVINO™ toolkit. For example, you cannot use the `1-0-1_A10DK_FP16_Generic` bitstream, when the Intel® Distribution of OpenVINO™ toolkit supports the `2-0-1_A10DK_FP16_Generic` bitstream.

Depending on how many bitstreams you selected, there are different folders for each FPGA card type which were downloaded in the Intel® Distribution of OpenVINO™ toolkit package:

1. For the Intel® Vision Accelerator Design with Intel® Arria® 10 FPGA the pre-trained bistreams are in `/opt/intel/openvino/bitstreams/a10_vision_design_bitstreams`. This example uses a SqueezeNet bitstream with low precision for the classification sample.

2. Rerun the environment setup script:
```sh
source /home/<user>/Downloads/fpga_support_files/setup_env.sh
```
	
3. Change to your home directory:
```sh
cd /home/<user>
```
	
4. Program the bitstream for the Intel® Vision Accelerator Design with Intel® Arria® 10 FPGA:
```sh
aocl program acl0 /opt/intel/openvino/bitstreams/a10_vision_design_bitstreams/5-0_PL1_FP11_SqueezeNet.aocx
```
			
### Optional Steps to Flash the FPGA Card

> **NOTE**:
>	- To avoid having to reprogram the board after a power down, a bitstream will be programmed to permanent memory on the Intel® Vision Accelerator Design with Intel® Arria® 10 FPGA. This will take about 20 minutes.
>	- The following steps 1-5 need to be done only once for a new Intel® Arria 10 FPGA card.

1. Plug in the micro USB cable to the card and your host system.

2. Run `jtagconfig` to ensure that the cable is properly inserted:
```sh
jtagconfig
```
	
3. Use `jtagconfig` to slow the clock:
```sh
jtagconfig --setparam 1 JtagClock 6M
```
	
4. Store the Intel® Vision Accelerator Design with Intel® Arria® 10 FPGA bistream on the board:
```sh
aocl flash acl0 /opt/intel/openvino/bitstreams/a10_vision_design_bitstreams/5-0_PL1_FP11_SqueezeNet.aocx
```
Your output is similar to:
```sh
USB-BlasterII [1-14]
02E660DD 10AX115H1(.|E2|ES)/10AX115H2/..
020A40DD 5M(1270ZF324|2210Z)/EPM2210
```

## 4. Setup a Neural Network Model for FPGA

In this section, you will create an FP16 model suitable for hardware accelerators. For more information, see the [FPGA plugin](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_supported_plugins_FPGA.html) section in the Inference Engine Developer Guide.


1. Create a directory for the FP16 SqueezeNet Model:
```sh
mkdir /home/<user>/squeezenet1.1_FP16
```
	
2. Go to `/home/<user>/squeezenet1.1_FP16`:
```sh
cd /home/<user>/squeezenet1.1_FP16
```
	
3. Use the Model Optimizer to convert an FP16 SqueezeNet Caffe* model into an optimized Intermediate Representation (IR):
```sh
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model /home/<user>/openvino_models/FP32/classification/squeezenet/1.1/caffe/squeezenet1.1.caffemodel --data_type FP16 --output_dir .
```
	
4. The `squeezenet1.1.labels` file contains the classes `ImageNet` uses. This file is included so that the inference results show text instead of classification numbers. Copy `squeezenet1.1.labels` to the your optimized model location:
```sh
cp /home/<user>/openvino_models/ir/squeezenet1.1/FP32/squeezenet1.1.labels  .
```
	
5. Copy a sample image to the release directory. You will use this with your optimized model:
```sh
sudo cp /opt/intel/openvino/deployment_tools/demo/car.png  ~/inference_engine_samples/intel64/Release
```
	
## 5. Run a Sample Application

1. Go to the samples directory
```sh
cd /home/<user>/inference_engine_samples/intel64/Release
```

2. Use an Inference Engine sample to run a sample application on the CPU:
```sh
./classification_sample_async -i car.png -m ~/openvino_models/ir/squeezenet1.1/FP32/squeezenet1.1.xml
```
Note the CPU throughput in Frames Per Second (FPS). This tells you how quickly the inference is done on the hardware. Now run the inference using the FPGA.

3. Add the `-d` option to target the FPGA:
```sh
./classification_sample_async -i car.png -m ~/squeezenet1.1_FP16/squeezenet1.1.xml -d HETERO:FPGA,CPU
```
The throughput on FPGA is listed and may show a lower FPS. This is due to the initialization time. To account for that, the next step increases the iterations to get a better sense of the speed the FPGA can run inference at.

4. Use `-ni` to increase the number of iterations, This option reduces the initialization impact:
```sh
./classification_sample_async -i car.png -m ~/squeezenet1.1_FP16/squeezenet1.1.xml -d HETERO:FPGA,CPU -ni 100
```

Congratulations, you are done with the Intel® Distribution of OpenVINO™ toolkit installation for FPGA. To learn more about how the Intel® Distribution of OpenVINO™ toolkit works, the Hello World tutorial and are other resources are provided below.

## Hello World Face Detection Tutorial

Use the [Intel® Distribution of OpenVINO™ toolkit with FPGA Hello World Face Detection Exercise](https://github.com/fritzboyle/openvino-with-fpga-hello-world-face-detection) to learn more about how the software and hardware work together.

## Additional Resources

Intel® Distribution of OpenVINO™ toolkit home page: [https://software.intel.com/en-us/openvino-toolkit](https://software.intel.com/en-us/openvino-toolkit)

Intel® Distribution of OpenVINO™ toolkit documentation: [https://docs.openvinotoolkit.org/](https://docs.openvinotoolkit.org/)

Inference Engine FPGA plugin documentation: [https://docs.openvinotoolkit.org/latest/_docs_IE_DG_supported_plugins_FPGA.html](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_supported_plugins_FPGA.html)
