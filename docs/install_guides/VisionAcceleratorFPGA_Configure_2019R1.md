# Configuration Guide for the Intel® Distribution of OpenVINO™ toolkit 2019R1 and the Intel® Vision Accelerator Design with an Intel® Arria® 10 FPGA (IEI's Mustang-F100-A10) on Linux* {#openvino_docs_install_guides_VisionAcceleratorFPGA_Configure_2019R1}

> **NOTES:**
>  * For a first-time installation, use all steps.
>  * Use step 1 only after receiving a new FPGA card.
>  * Repeat steps 2-4 when installing a new version of the Intel® Distribution of OpenVINO™ toolkit.
>  * Use steps 3-4 when a Neural Network topology used by an Intel® Distribution of OpenVINO™ toolkit application changes.

## 1. Configure and Set Up the Intel® Vision Accelerator Design with an Intel® Arria® 10 FPGA

For the 2019R1.x releases, the Intel® Distribution of OpenVINO™ toolkit introduced a new board support package (BSP) `a10_1150_sg1` for the Intel® Vision Accelerator Design with an Intel® Arria®  10 FPGA, which is included in the `fpga_support_files.tgz` archive below. To program the bitstreams for the Intel® Distribution of OpenVINO™ toolkit 2019R1.x, you need to program the BSP into the board using the USB blaster.

1. Download [Intel® Quartus® Prime Programmer and Tools Standard Edition 18.1](http://fpgasoftware.intel.com/18.1/?edition=standard&platform=linux&download_manager=direct#tabs-4). Install the Intel® Quartus® Prime Programmer and Tools Software to the `/home/<user>/intelFPGA/18.1` directory.

2. Download `fpga_support_files.tgz` from the [Intel Registration Center](http://registrationcenter-download.intel.com/akdlm/irc_nas/12954/fpga_support_files.tgz) to the `~/Downloads` directory. The files in this `.tgz` archive are required to ensure your FPGA card and the Intel® Distribution of OpenVINO™ toolkit work correctly.

3. Go to the directory where you downloaded the `fpga_support_files.tgz` archive.

4. Unpack the `.tgz` file:
```sh
tar -xvzf fpga_support_files.tgz
```
A directory named `fpga_support_files` is created.

5. Go to the `fpga_support_files` directory:
```sh
cd fpga_support_files
```
	
6. Switch to superuser:
```sh
sudo su
```
	
7. Use the `setup_env.sh` script from `fpga_support_files.tgz` to set your environment variables:
```sh
source /home/<user>/Downloads/fpga_support_files/setup_env.sh
```

8. Uninstall any previous BSP before installing the OpenCL BSP for the 2019R1.x BSP:
```sh
aocl uninstall /opt/altera/aocl-pro-rte/aclrte-linux64/board/<BSP_package>/
```

9. Change directory to `Downloads/fpga_support_files/`:
```sh
cd /home/<user>/Downloads/fpga_support_files/
```
	
10. Run the FPGA dependencies script, which allows OpenCL to support Ubuntu* and recent kernels:
```sh
./install_openvino_fpga_dependencies.sh
```

11. When asked, select the appropriate hardware accelerators you plan to use so it installs the correct dependencies.

12. If you installed the 4.14 kernel as part of the installation script, you will need to reboot the machine and select the new kernel in the Ubuntu (grub) boot menu. You will also need to rerun `setup_env.sh` to set up your environmental variables again.
		
13. Export the Intel® Quartus® Prime Programmer environment variable:
```sh
export QUARTUS_ROOTDIR=/home/<user>/intelFPGA/18.1/qprogrammer
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

18. Go to `/opt/altera/aocl-pro-rte/aclrte-linux64/board/a10_1150_sg1/bringup`, where `sg1_boardtest_2ddr_base.sof`is located:
```sh
cd /opt/altera/aocl-pro-rte/aclrte-linux64/board/a10_1150_sg1/bringup
```
	
19. Program the new sof file to the board:
```sh
quartus_pgm -c 1 -m JTAG -o "p;sg1_boardtest_2ddr_base.sof"
```

20. Soft reboot:
```sh
sudo reboot
```

21. Open up a new terminal and restore sudo access and the environment variables:
```sh
sudo su
source /home/<user>/Downloads/fpga_support_files/setup_env.sh
```

22. Install OpenCL™ devices. Enter **Y** when prompted to install:
```sh
aocl install
```
	
23. Reboot the machine:
```sh
reboot
```
	
24. Open up a new terminal and restore sudo access and the environment variables:
```sh
sudo su
source /home/<user>/Downloads/fpga_support_files/setup_env.sh
export QUARTUS_ROOTDIR=/home/<user>/intelFPGA/18.1/qprogrammer
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
	
27. Go to `/opt/intel/openvino/bitstreams/a10_vision_design_bitstreams/`, where `2019R1_PL1_FP11_ResNet_SqueezeNet_VGG.aocx `is located:
```sh
cd /opt/intel/openvino/bitstreams/a10_vision_design_bitstreams/
```
	
28. Program the `2019R1_PL1_FP11_ResNet_SqueezeNet_VGG.aocx` file to the flash to be made permanently available even after power cycle:
```sh
aocl flash acl0 2019R1_PL1_FP11_ResNet_SqueezeNet_VGG.aocx
```
> **NOTE**: You will need the USB Blaster for this.

29. Hard reboot the host system including powering off.

30. Now Soft reboot the host system to ensure the new PCIe device is seen properly
```sh
reboot
```

31. Open up a new terminal and restore sudo access and the environment variables:
```sh
sudo su
source /home/<user>/Downloads/fpga_support_files/setup_env.sh
```

32. Check if the host system recognizes the Intel® Vision Accelerator Design with Intel® Arria® 10 FPGA board. Confirm you can detect the PCIe card:
```sh
lspci | grep -i Altera
```
Your output is similar to:
```sh
01:00.0 Processing accelerators: Altera Corporation Device 2494 (rev 01)
```

33. Run `aocl diagnose`:
```sh
aocl diagnose
```
You should see `DIAGNOSTIC_PASSED` before proceeding to the next steps.

## 2. Program a Bitstream

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
aocl program acl0 /opt/intel/openvino/bitstreams/a10_vision_design_bitstreams/2019R1_PL1_FP11_ResNet_SqueezeNet_VGG.aocx
```
			
### Steps to Flash the FPGA Card

> **NOTE**:
>	- To avoid having to reprogram the board after a power down, a bitstream will be programmed to permanent memory on the Intel® Vision Accelerator Design with Intel® Arria® 10 FPGA. This will take about 20 minutes.
>	- The steps can be followed in the [Configure and Setup the Intel® Vision Accelerator Design with an Intel® Arria® 10 FPGA](#1-configure-and-setup-the-intel-vision-accelerator-design-with-an-intel-arria-10-fpga) section of this guide from steps 14-18 and 28-36.


## 3. Setup a Neural Network Model for FPGA

In this section, you will create an FP16 model suitable for hardware accelerators. For more information, see the [FPGA plugin](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_supported_plugins_FPGA.html) section in the Inference Engine Developer Guide.


1. Create a directory for the FP16 SqueezeNet Model:
```sh
mkdir /home/<user>/squeezenet1.1_FP16
```
	
2. Go to `/home/<user>/squeezenet1.1_FP16`:
```sh
cd /home/<user>/squeezenet1.1_FP16
```

3. Use the Model Optimizer to convert the FP32 SqueezeNet Caffe* model into an FP16 optimized Intermediate Representation (IR). The model files were downloaded when you ran the the Image Classification verification script while [installing the Intel® Distribution of OpenVINO™ toolkit for Linux* with FPGA Support](installing-openvino-linux-fpga.md). To convert, run the Model Optimizer script with the following arguments:	
```sh
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model /home/<user>/openvino_models/models/FP32/classification/squeezenet/1.1/caffe/squeezenet1.1.caffemodel --data_type FP16 --output_dir .
```
	
4. The `squeezenet1.1.labels` file contains the classes `ImageNet` uses. This file is included so that the inference results show text instead of classification numbers. Copy `squeezenet1.1.labels` to the your optimized model location:
```sh
cp /home/<user>/openvino_models/ir/FP32/classification/squeezenet/1.1/caffe/squeezenet1.1.labels  .
```
	
5. Copy a sample image to the release directory. You will use this with your optimized model:
```sh
sudo cp /opt/intel/openvino/deployment_tools/demo/car.png  ~/inference_engine_samples_build/intel64/Release
```
	
## 4. Run a Sample Application

1. Go to the samples directory
```sh
cd /home/<user>/inference_engine_samples_build/intel64/Release
```

2. Use an Inference Engine sample to run a sample application on the CPU:
```sh
./classification_sample_async -i car.png -m ~/openvino_models/ir/FP32/classification/squeezenet/1.1/caffe/squeezenet1.1.xml
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

Congratulations, you are done with the Intel® Distribution of OpenVINO™ toolkit installation for FPGA. 

## Additional Resources

Intel® Distribution of OpenVINO™ toolkit home page: [https://software.intel.com/en-us/openvino-toolkit](https://software.intel.com/en-us/openvino-toolkit)

Intel® Distribution of OpenVINO™ toolkit documentation: [https://docs.openvinotoolkit.org/](https://docs.openvinotoolkit.org/)

Inference Engine FPGA plugin documentation: [https://docs.openvinotoolkit.org/latest/_docs_IE_DG_supported_plugins_FPGA.html](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_supported_plugins_FPGA.html)
