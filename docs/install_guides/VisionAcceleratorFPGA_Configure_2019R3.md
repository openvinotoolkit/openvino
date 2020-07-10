# Configuration Guide for the Intel® Distribution of OpenVINO™ toolkit 2019R3 and the Intel® Vision Accelerator Design with an Intel® Arria® 10 FPGA SG1 and SG2 (IEI's Mustang-F100-A10) on Linux* {#openvino_docs_install_guides_VisionAcceleratorFPGA_Configure_2019R3}

## 1. Configure and Set Up the Intel® Vision Accelerator Design with an Intel® Arria® 10 FPGA

1. Download [Intel® Quartus® Prime Programmer and Tools Standard Edition 18.1](http://fpgasoftware.intel.com/18.1/?edition=standard&platform=linux&download_manager=direct#tabs-4). Install the Intel® Quartus® Prime Programmer and Tools Software to the `/home/<user>/intelFPGA/18.1` directory.

2. Download `fpga_support_files.tgz` from the [Intel Registration Center](http://registrationcenter-download.intel.com/akdlm/irc_nas/12954/fpga_support_files.tgz) to the `~/Downloads` directory. The files in this `.tgz` archive are required to ensure your FPGA card and the Intel® Distribution of OpenVINO™ toolkit work correctly.

3. Go to the directory where you downloaded the `fpga_support_files.tgz` archive.

4. Unpack the `.tgz` file:
```sh
tar -xvzf fpga_support_files.tgz
```
A directory named `fpga_support_files` is created.

5. Switch to superuser:
```sh
sudo su
```
	
6. Change directory to `Downloads/fpga_support_files/`:
```sh
cd /home/<user>/Downloads/fpga_support_files/
```

7. Copy the USB Blaster Rules file:
```sh
cp config/51-usbblaster.rules /etc/udev/rules.d
udevadm control --reload-rules
udevadm trigger
```

8. Copy aocl fixes for latest kernels:
```sh
cp fixes/Command.pm /opt/altera/aocl-pro-rte/aclrte-linux64/share/lib/perl/acl/
cp config/blacklist-altera-cvp.conf /etc/modprobe.d/
```

9. Copy flash files so we don't need a full Quartus installation:
```sh
cp -r config/aocl_flash/linux64/* /home/<user>/intelFPGA/18.1/qprogrammer/linux64
```

10. Unpack the BSP for your appropriate Intel® Vision Accelerator Design with an Intel® Arria® 10 FPGA SG1 or SG2:
```sh
cd /opt/intel/openvino/bitstreams/a10_vision_design_sg<#>_bitstreams/BSP/
tar -xvzf a10_1150_sg<#>_r3.tgz
chmod -R 755 /opt/intel/openvino/bitstreams/a10_vision_design_sg<#>_bitstreams
```
> **NOTE**: If you do not know which version of the board you have, please refer to the product label on the fan cover side or by the product SKU: Mustang-F100-A10-R10 => SG1; Mustang-F100-A10E-R10 => SG2

11. Create an initialization script `/home/<user>/init_openvino.sh` with the following content that can be run upon opening a new terminal or rebooting. This will setup your proper environment variables.
```sh
export AOCL_BOARD_PACKAGE_ROOT=/opt/intel/openvino/bitstreams/a10_vision_design_sg<#>_bitstreams/BSP/a10_1150_sg<#>
export QUARTUS_ROOTDIR=/home/<user>/intelFPGA/18.1/qprogrammer
export PATH=$PATH:/opt/altera/aocl-pro-rte/aclrte-linux64/bin:/opt/altera/aocl-pro-rte/aclrte-linux64/host/linux64/bin:/home/<user>/intelFPGA/18.1/qprogrammer/bin
export INTELFPGAOCLSDKROOT=/opt/altera/aocl-pro-rte/aclrte-linux64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$AOCL_BOARD_PACKAGE_ROOT/linux64/lib
export CL_CONTEXT_COMPILER_MODE_INTELFPGA=3
source /opt/altera/aocl-pro-rte/aclrte-linux64/init_opencl.sh
source /opt/intel/openvino/bin/setupvars.sh 
```

12. Source the script.
```sh
source /home/<user>/init_openvino.sh
```

13. Uninstall any previous BSP before installing the OpenCL BSP for the 2019R3 BSP:
```sh
aocl uninstall /opt/altera/aocl-pro-rte/aclrte-linux64/board/<BSP_package>/
```

14. Set up the USB Blaster:
		
    1. Connect the cable between the board and the host system. Use the letter codes in the diagram below for the connection points:
				
    2. Connect the B end of the cable to point B on the board.

    3. Connect the F end of the cable to point F on the FPGA download cable.
				
    4. From point F end of the cable to point F on the FPGA download cable, the connection is as shown:
![](../img/VisionAcceleratorJTAG.png)

15. Run `jtagconfig` to ensure that your Intel FPGA Download Cable driver is ready to use:
```sh
jtagconfig
```
Your output is similar to:
```sh
1) USB-Blaster [1-6]
02E660DD   10AX115H1(.|E2|ES)/10AX115H2/.. 
```

16. Use `jtagconfig` to slow the clock. The message "No parameter named JtagClock" can be safely ignored.
```sh
jtagconfig --setparam 1 JtagClock 6M
```
	
17. (OPTIONAL) Confirm the clock is set to 6M:
```sh
jtagconfig --getparam 1 JtagClock
```
You should see the following:
```sh
6M
```

18. Go to `/opt/intel/openvino/bitstreams/a10_vision_design_sg<#>_bitstreams/BSP/a10_1150_sg<#>/bringup`, where `sg<#>_boardtest_2ddr_base.sof`is located:
```sh
cd /opt/intel/openvino/bitstreams/a10_vision_design_sg<#>_bitstreams/BSP/a10_1150_sg<#>/bringup
```
	
19. Program the new sof file to the board:
```sh
quartus_pgm -c 1 -m JTAG -o "p;sg<#>_boardtest_2ddr_base.sof"
```

20. Soft reboot:
```sh
reboot
```

21. Source the environment variable script you made.
```sh
sudo su
source /home/<user>/init_openvino.sh
```

22. Install OpenCL™ devices. Enter **Y** when prompted to install:
```sh
aocl install
```
	
23. Reboot the machine:
```sh
reboot
```
	
24. Source the environment variable script you made.
```sh
sudo su
source /home/<user>/init_openvino.sh
```

25. Run `aocl diagnose`:
```sh
aocl diagnose
```
Your screen displays `DIAGNOSTIC_PASSED`.

26. Use `jtagconfig` to slow the clock. The message "No parameter named JtagClock" can be safely ignored.
```sh
jtagconfig --setparam 1 JtagClock 6M
```
	
27. Go to `/opt/intel/openvino/bitstreams/a10_vision_design_sg<#>_bitstreams/`, where `2019R3_PV_PL<#>_FP11_InceptionV1_SqueezeNet.aocx `is located:
```sh
cd /opt/intel/openvino/bitstreams/a10_vision_design_sg<#>_bitstreams/
```
	
28. Program the `2019R3_PV_PL<#>_FP11_InceptionV1_SqueezeNet.aocx` file to the flash to be made permanently available even after power cycle:
```sh
aocl flash acl0 2019R3_PV_PL<#>_FP11_InceptionV1_SqueezeNet.aocx
```
> **NOTE**: You will need the USB Blaster for this.

29. Hard reboot the host system including powering off.

30. Source the environment variable script you made.
```sh
sudo su
source /home/<user>/init_openvino.sh
```

31. Check if the host system recognizes the Intel® Vision Accelerator Design with Intel® Arria® 10 FPGA board. Confirm you can detect the PCIe card:
```sh
lspci | grep -i Altera
```
Your output is similar to:
```sh
01:00.0 Processing accelerators: Altera Corporation Device 2494 (rev 01)
```

32. Run `aocl diagnose`:
```sh
aocl diagnose
```
You should see `DIAGNOSTIC_PASSED` before proceeding to the next steps.

## 2. Program a Bitstream

The bitstream you program should correspond to the topology you want to deploy. In this section, you program a SqueezeNet bitstream and deploy the classification sample with a SqueezeNet model that you used the Model Optimizer to convert in the steps before.

> **IMPORTANT**: Only use bitstreams from the installed version of the Intel® Distribution of OpenVINO™ toolkit. Bitstreams from older versions of the Intel® Distribution of OpenVINO™ toolkit are incompatible with later versions of the Intel® Distribution of OpenVINO™ toolkit. For example, you cannot use the `1-0-1_A10DK_FP16_Generic` bitstream, when the Intel® Distribution of OpenVINO™ toolkit supports the `2-0-1_A10DK_FP16_Generic` bitstream.

Depending on how many bitstreams you selected, there are different folders for each FPGA card type which were downloaded in the Intel® Distribution of OpenVINO™ toolkit package:

1. For the Intel® Vision Accelerator Design with Intel® Arria® 10 FPGA SG1 or SG2, the pre-trained bistreams are in `/opt/intel/openvino/bitstreams/a10_vision_design_sg<#>_bitstreams/`. This example uses a SqueezeNet bitstream with low precision for the classification sample.

2. Source the environment variable script you made.
```sh
source /home/<user>/init_openvino.sh
```
	
3. Change to your home directory:
```sh
cd /home/<user>
```
	
4. Program the bitstream for the Intel® Vision Accelerator Design with Intel® Arria® 10 FPGA SG1 or SG2:
```sh
aocl program acl0 /opt/intel/openvino/bitstreams/a10_vision_design_sg<#>_bitstreams/2019R3_PV_PL<#>_FP11_InceptionV1_SqueezeNet.aocx
```
			
### Steps to Flash the FPGA Card

> **NOTE**:
>	- To avoid having to reprogram the board after a power down, a bitstream will be programmed to permanent memory on the Intel® Vision Accelerator Design with Intel® Arria® 10 FPGA. This will take about 20 minutes.
>	- The steps can be followed above in this guide to do this.


## 3. Setup a Neural Network Model for FPGA

In this section, you will create an FP16 model suitable for hardware accelerators. For more information, see the [FPGA plugin](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_supported_plugins_FPGA.html) section in the Inference Engine Developer Guide.


1. Create a directory for the FP16 SqueezeNet Model:
```sh
mkdir ~/squeezenet1.1_FP16
```
	
2. Go to `~/squeezenet1.1_FP16`:
```sh
cd ~/squeezenet1.1_FP16
```

3. Use the Model Optimizer to convert the FP32 SqueezeNet Caffe* model into an FP16 optimized Intermediate Representation (IR). The model files were downloaded when you ran the the Image Classification verification script while [installing the Intel® Distribution of OpenVINO™ toolkit for Linux* with FPGA Support](installing-openvino-linux-fpga.md). To convert, run the Model Optimizer script with the following arguments:	
```sh
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model ~/openvino_models/models/FP16/public/squeezenet1.1/squeezenet1.1.caffemodel --data_type FP16 --output_dir .
```
	
4. The `squeezenet1.1.labels` file contains the classes `ImageNet` uses. This file is included so that the inference results show text instead of classification numbers. Copy `squeezenet1.1.labels` to the your optimized model location:
```sh
cp ~/openvino_models/ir/FP16/public/squeezenet1.1/squeezenet1.1.labels  .
```
	
5. Copy a sample image to the release directory. You will use this with your optimized model:
```sh
cp /opt/intel/openvino/deployment_tools/demo/car.png ~/inference_engine_samples_build/intel64/Release
```
	
## 4. Run a Sample Application

1. Go to the samples directory
```sh
cd ~/inference_engine_samples_build/intel64/Release
```

2. Use an Inference Engine sample to run a sample application on the CPU:
```sh
./classification_sample_async -i car.png -m ~/openvino_models/ir/FP16/public/squeezenet1.1/squeezenet1.1.xml
```
Note the CPU throughput in Frames Per Second (FPS). This tells you how quickly the inference is done on the hardware. Now run the inference using the FPGA.

3. Add the `-d` option to target the FPGA:
```sh
./classification_sample_async -i car.png -m ~/openvino_models/ir/FP16/public/squeezenet1.1/squeezenet1.1.xml -d HETERO:FPGA,CPU
```
The throughput on FPGA is listed and may show a lower FPS. This may be due to the initialization time. To account for that, increase the number of iterations or batch size when deploying to get a better sense of the speed the FPGA can run inference at.

Congratulations, you are done with the Intel® Distribution of OpenVINO™ toolkit installation for FPGA. To learn more about how the Intel® Distribution of OpenVINO™ toolkit works, the Hello World tutorial and are other resources are provided below.

## Hello World Face Detection Tutorial

Use the [Intel® Distribution of OpenVINO™ toolkit with FPGA Hello World Face Detection Exercise](https://github.com/fritzboyle/openvino-with-fpga-hello-world-face-detection) to learn more about how the software and hardware work together.

## Additional Resources

Intel® Distribution of OpenVINO™ toolkit home page: [https://software.intel.com/en-us/openvino-toolkit](https://software.intel.com/en-us/openvino-toolkit)

Intel® Distribution of OpenVINO™ toolkit documentation: [https://docs.openvinotoolkit.org/](https://docs.openvinotoolkit.org/)

Inference Engine FPGA plugin documentation: [https://docs.openvinotoolkit.org/latest/_docs_IE_DG_supported_plugins_FPGA.html](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_supported_plugins_FPGA.html)
