# Configuration Guide for the Intel® Distribution of OpenVINO™ toolkit 2020.3 and the Intel® Vision Accelerator Design with an Intel® Arria® 10 FPGA SG2 (IEI's Mustang-F100-A10) on Linux* {#openvino_docs_install_guides_VisionAcceleratorFPGA_Configure}

> **NOTE**: Intel® Arria® 10 FPGA (Mustang-F100-A10) Speed Grade 1 is not available in the OpenVINO 2020.3 package. If you use Intel® Vision Accelerator Design with an Intel® Arria 10 FPGA (Mustang-F100-A10) Speed Grade 1, we recommend continuing to use the [Intel® Distribution of OpenVINO™ toolkit 2020.1](https://docs.openvinotoolkit.org/2020.1/_docs_install_guides_VisionAcceleratorFPGA_Configure.html) release.
For previous versions, see [Configuration Guide for OpenVINO 2019R3](https://docs.openvinotoolkit.org/2019_R3.1/_docs_install_guides_VisionAcceleratorFPGA_Configure_2019R3.html), [Configuration Guide for OpenVINO 2019R1](https://docs.openvinotoolkit.org/2019_R3.1/_docs_install_guides_VisionAcceleratorFPGA_Configure_2019R1.html), [Configuration Guide for OpenVINO 2018R5](https://docs.openvinotoolkit.org/2019_R3.1/_docs_install_guides_VisionAcceleratorFPGA_Configure_2018R5.html).

## 1. Configure and Set Up the Intel® Vision Accelerator Design with an Intel® Arria® 10 FPGA

1. Download [Intel® Quartus® Prime Programmer and Tools Standard Edition 18.1](http://fpgasoftware.intel.com/18.1/?edition=standard&platform=linux&download_manager=direct#tabs-4). Install the Intel® Quartus® Prime Programmer and Tools Software to the `/home/<user>/intelFPGA/18.1` directory.

2. Download the [fpga_install.sh](https://docs.openvinotoolkit.org/downloads/2020/2/fpga_install.sh) script to the `/home/<user>` directory.

   a. Switch to superuser:
```sh
sudo su
```
   b. Use the `fpga_install.sh` script from `/home/<user>` to install your FPGA card (default is SG2).
```sh
source /home/<user>/fpga_install.sh
```
   c. To know more about the fpga_install options, invoke the script with `-h` command.
```sh
source /home/<user>/fpga_install.sh -h
```   
   d. Follow the `fpga_install.sh` script prompts to finish installing your FPGA card.
   
   e. After reboot launch the script again with same options as in step 2.b.
   
   f. The `fpga_install.sh` script creates an initialization script `/home/<user>/init_openvino.sh` that should be used to setup proper environment variables.
   
   g. To test if FPGA card was installed succesfully run `aocl diagnose`:
```sh
aocl diagnose
```
You should see `DIAGNOSTIC_PASSED` before proceeding to the next steps.

   h. If you prefer to install the FPGA card manually, follow the steps 3-17 in this section and [Steps to Flash the FPGA Card](#steps-to-flash-the-fpga-card), otherwise you can skip to "Program a Bitstream".

3. Check if /etc/udev/rules.d/51-usbblaster.rules file exists and content matches with 3.b, if it does skip to next step.

   a. Switch to superuser:
```sh
sudo su
```

   b. Create a file named /etc/udev/rules.d/51-usbblaster.rules and add the following lines to it (Red Hat Enterprise 5 and above):
```sh
# Intel FPGA Download Cable
SUBSYSTEM=="usb", ATTR{idVendor}=="09fb", ATTR{idProduct}=="6001", MODE="0666"
SUBSYSTEM=="usb", ATTR{idVendor}=="09fb", ATTR{idProduct}=="6002", MODE="0666" 
SUBSYSTEM=="usb", ATTR{idVendor}=="09fb", ATTR{idProduct}=="6003", MODE="0666"   

# Intel FPGA Download Cable II
SUBSYSTEM=="usb", ATTR{idVendor}=="09fb", ATTR{idProduct}=="6010", MODE="0666"
SUBSYSTEM=="usb", ATTR{idVendor}=="09fb", ATTR{idProduct}=="6810", MODE="0666"
```
>  **CAUTION**: Do not add extra line breaks to the .rules file.

   c. Reload udev rules without reboot:
```sh
udevadm control --reload-rules
udevadm trigger
```

   d. You can exit superuser if you wish.


4. Unpack the BSP for your Intel® Vision Accelerator Design with an Intel® Arria® 10 FPGA SG2:
> **NOTE**: If you installed OpenVINO™ as root you will need to switch to superuser
```sh
cd /opt/intel/openvino/bitstreams/a10_vision_design_sg2_bitstreams/BSP/
sudo su
tar -xvzf a10_1150_sg2_r4.1.tgz
chmod -R 755 /opt/intel/openvino/bitstreams/a10_vision_design_sg2_bitstreams
```
> **NOTE**: If you do not know which version of the board you have, please refer to the product label on the fan cover side or by the product SKU: Mustang-F100-A10E-R10 => SG2

5. Create an initialization script `/home/<user>/init_openvino.sh` with the following content that can be run upon opening a new terminal or rebooting. This will setup your proper environment variables.
```sh
export IOCL_BOARD_PACKAGE_ROOT=/opt/intel/openvino/bitstreams/a10_vision_design_sg2_bitstreams/BSP/a10_1150_sg2
export AOCL_BOARD_PACKAGE_ROOT=/opt/intel/openvino/bitstreams/a10_vision_design_sg2_bitstreams/BSP/a10_1150_sg2
export QUARTUS_DIR=/home/<user>/intelFPGA/18.1/qprogrammer
export QUARTUS_ROOTDIR=/home/<user>/intelFPGA/18.1/qprogrammer
export INTELFPGAOCLSDKROOT=/opt/altera/aocl-pro-rte/aclrte-linux64
source $INTELFPGAOCLSDKROOT/init_opencl.sh
export PATH=$PATH:$INTELFPGAOCLSDKROOT/host/linux64/bin:$QUARTUS_ROOTDIR/bin
export CL_CONTEXT_COMPILER_MODE_INTELFPGA=3
source /opt/intel/openvino/bin/setupvars.sh 
```

6. Source the script. (This assumes you already have installed the Intel® FPGA Runtime Environment for OpenCL Linux x86-64 Pro Edition 19.1)
```sh
source /home/<user>/init_openvino.sh
```

7. Uninstall any previous BSP before installing the OpenCL BSP for the 2020.3 BSP. Enter **Y** when prompted to uninstall (Enter sudo credentials when prompted):
```sh
aocl uninstall
```

8. Install the new BSP. Enter **Y** when prompted to install (Enter sudo credentials when prompted):
```sh
aocl install
```

9. Set up the USB Blaster:
		
    1. Connect the cable between the board and the host system. Use the letter codes in the diagram below for the connection points:
				
    2. Connect the B end of the cable to point B on the board.

    3. Connect the F end of the cable to point F on the FPGA download cable.
				
    4. From point F end of the cable to point F on the FPGA download cable, the connection is as shown:
![](../img/VisionAcceleratorJTAG.png)

10. Run `jtagconfig` to ensure that your Intel FPGA Download Cable driver is ready to use:
```sh
jtagconfig
```
Your output is similar to:
```sh
1) USB-Blaster [1-6]
02E660DD   10AX115H1(.|E2|ES)/10AX115H2/.. 
```
or:
```sh
1) USB-BlasterII [3-3]                        
  02E660DD   10AX115H1(.|E2|ES)/10AX115H2/..
```

11. Use `jtagconfig` to slow the clock. The message "No parameter named JtagClock" can be safely ignored.
```sh
jtagconfig --setparam 1 JtagClock 6M
```
	
12. (OPTIONAL) Confirm the clock is set to 6M:
```sh
jtagconfig --getparam 1 JtagClock
```
You should see the following:
```sh
6M
```

13. Go to `/opt/intel/openvino/bitstreams/a10_vision_design_sg2_bitstreams/BSP/a10_1150_sg2/bringup`, where `sg2_boardtest_2ddr_base.sof`is located:
```sh
cd /opt/intel/openvino/bitstreams/a10_vision_design_sg2_bitstreams/BSP/a10_1150_sg2/bringup
```
	
14. Program the new sof file to the board:
```sh
quartus_pgm -c 1 -m JTAG -o "p;sg2_boardtest_2ddr_base.sof"
```

15. Soft reboot:
```sh
reboot
```

16. Source the environment variable script you made.
```sh
source /home/<user>/init_openvino.sh
```

17. Run `aocl diagnose`:
```sh
aocl diagnose
```
Your screen displays `DIAGNOSTIC_PASSED`.

> **NOTE**: at this point if you do not want to flash the FPGA Card you can go to "Program a Bitstream"

### <a name="steps-to-flash-the-fpga-card"></a>Steps to Flash the FPGA Card

> **NOTE**:
>	- To avoid having to reprogram the board after a power down, a bitstream will be programmed to permanent memory on the Intel® Vision Accelerator Design with Intel® Arria® 10 FPGA. This will take about 20 minutes.
>	- The steps can be followed below in this guide to do this.

18. Use `jtagconfig` to slow the clock. The message "No parameter named JtagClock" can be safely ignored.
```sh
jtagconfig --setparam 1 JtagClock 6M
```

19. Check if $QUARTUS_ROOTDIR/linux64/perl/bin exists
```sh
ls $QUARTUS_ROOTDIR/linux64/perl/bin
```

20. If you see message "ls: cannot access /home/<user>/intelFPGA/18.1/qprogrammer/linux64/perl/bin: No such file or directory" create perl/bin directory and a symbolic link to perl
```sh
mkdir -p $QUARTUS_ROOTDIR/linux64/perl/bin
ln -s /usr/bin/perl $QUARTUS_ROOTDIR/linux64/perl/bin/perl
```

21. If you see message "perl" go to the next step
	
22. Go to `/opt/intel/openvino/bitstreams/a10_vision_design_sg2_bitstreams/BSP/a10_1150_sg2/bringup`, where `sg2_boardtest_2ddr_top.aocx` is located:
```sh
cd /opt/intel/openvino/bitstreams/a10_vision_design_sg2_bitstreams/BSP/a10_1150_sg2/bringup
```
	
23. Program the `sg2_boardtest_2ddr_top.aocx` file to the flash to be made permanently available even after power cycle:
```sh
sudo su
aocl flash acl0 sg2_boardtest_2ddr_top.aocx
```
> **NOTE**: You will need the USB Blaster for this.

24. Hard reboot the host system including powering off.

25. Source the environment variable script you made.
```sh
source /home/<user>/init_openvino.sh
```

26. Check if the host system recognizes the Intel® Vision Accelerator Design with Intel® Arria® 10 FPGA board. Confirm you can detect the PCIe card:
```sh
lspci | grep -i Altera
```
Your output is similar to:
```sh
01:00.0 Processing accelerators: Altera Corporation Device 2494 (rev 01)
```

27. Run `aocl diagnose`:
```sh
aocl diagnose
```
You should see `DIAGNOSTIC_PASSED` before proceeding to the next steps.

## 2. Program a Bitstream

The bitstream you program should correspond to the topology you want to deploy. In this section, you program a SqueezeNet bitstream and deploy the classification sample with a SqueezeNet model that you used the Model Optimizer to convert in the steps before.

> **IMPORTANT**: Only use bitstreams from the installed version of the Intel® Distribution of OpenVINO™ toolkit. Bitstreams from older versions of the Intel® Distribution of OpenVINO™ toolkit are incompatible with later versions of the Intel® Distribution of OpenVINO™ toolkit. For example, you cannot use the `2019R4_PL2_FP11_AlexNet_GoogleNet_Generic` bitstream, when the Intel® Distribution of OpenVINO™ toolkit supports the `2020-2_PL2_FP11_AlexNet_GoogleNet_Generic` bitstream.

Depending on how many bitstreams you selected, there are different folders for each FPGA card type which were downloaded in the Intel® Distribution of OpenVINO™ toolkit package:

1. For the Intel® Vision Accelerator Design with Intel® Arria® 10 FPGA SG2, the pre-trained bitstreams are in `/opt/intel/openvino/bitstreams/a10_vision_design_sg2_bitstreams/`. This example uses a SqueezeNet bitstream with low precision for the classification sample.

2. Source the environment variable script you made.
```sh
source /home/<user>/init_openvino.sh
```
	
3. Change to your home directory:
```sh
cd /home/<user>
```
	
4. Program the bitstream for the Intel® Vision Accelerator Design with Intel® Arria® 10 FPGA SG2:
```sh
aocl program acl0 /opt/intel/openvino/bitstreams/a10_vision_design_sg2_bitstreams/2020-3_PL2_FP16_SqueezeNet_TinyYolo_VGG.aocx
```

## 3. Set up a Sample Neural Network Model for FPGA

> **NOTE**: The SqueezeNet Caffe* model was already downloaded and converted to an FP16 IR when you ran the Image Classification Verification Script while [installing the Intel® Distribution of OpenVINO™ toolkit for Linux* with FPGA Support](installing-openvino-linux-fpga.md). Read this section only if you want to convert the model manually, otherwise skip and go to the next section to run the Image Classification sample application.

In this section, you will create an FP16 model suitable for hardware accelerators. For more information, see the [FPGA plugin](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_supported_plugins_FPGA.html) section in the Inference Engine Developer Guide.


1. Create a directory for the FP16 SqueezeNet Model:
```sh
mkdir ~/squeezenet1.1_FP16
```
	
2. Go to `~/squeezenet1.1_FP16`:
```sh
cd ~/squeezenet1.1_FP16
```

3. Use the Model Optimizer to convert the FP16 SqueezeNet Caffe* model into an FP16 optimized Intermediate Representation (IR). The model files were downloaded when you ran the the Image Classification verification script while [installing the Intel® Distribution of OpenVINO™ toolkit for Linux* with FPGA Support](installing-openvino-linux-fpga.md). To convert, run the Model Optimizer script with the following arguments:	
```sh
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model ~/openvino_models/models/public/squeezenet1.1/squeezenet1.1.caffemodel --data_type FP16 --output_dir .
```
	
4. The `squeezenet1.1.labels` file contains the classes `ImageNet` uses. This file is included so that the inference results show text instead of classification numbers. Copy `squeezenet1.1.labels` to the your optimized model location:
```sh
cp /opt/intel/openvino/deployment_tools/demo/squeezenet1.1.labels  .
```
	
5. Copy a sample image to the release directory. You will use this with your optimized model:
```sh
cp /opt/intel/openvino/deployment_tools/demo/car.png ~/inference_engine_samples_build/intel64/Release
```
	
## 4. Run the Image Classification Sample Application

In this section you will run the Image Classification sample application, with the Caffe* Squeezenet1.1 model on your Intel® Vision Accelerator Design with an Intel® Arria® 10 FPGA. 

Image Classification sample application binary file was automatically built and the FP16 model IR files are created when you ran the Image Classification Verification Script while [installing the Intel® Distribution of OpenVINO™ toolkit for Windows* with FPGA Support](installing-openvino-windows-fpga.md):
* Compiled sample Application binaries are located in the `~/inference_engine_samples_build/intel64/Release` directory.
* Generated IR files are in the `~/openvino_models/ir/public/squeezenet1.1/FP16/` directory.


1. Go to the samples directory
```sh
cd ~/inference_engine_samples_build/intel64/Release
```

2. Use an Inference Engine sample to run a sample inference on the CPU:
```sh
./classification_sample_async -i car.png -m ~/openvino_models/ir/public/squeezenet1.1/FP16/squeezenet1.1.xml
```
Note the CPU throughput in Frames Per Second (FPS). This tells you how quickly the inference is done on the hardware. Now run the inference using the FPGA.

3. Add the `-d` option to target the FPGA:
```sh
./classification_sample_async -i car.png -m ~/openvino_models/ir/public/squeezenet1.1/FP16/squeezenet1.1.xml -d HETERO:FPGA,CPU
```
The throughput on FPGA is listed and may show a lower FPS. This may be due to the initialization time. To account for that, increase the number of iterations or batch size when deploying to get a better sense of the speed the FPGA can run inference at.

Congratulations, you are done with the Intel® Distribution of OpenVINO™ toolkit installation for FPGA. To learn more about how the Intel® Distribution of OpenVINO™ toolkit works, the Hello World tutorial and are other resources are provided below.

## Hello World Face Detection Tutorial

Use the [Intel® Distribution of OpenVINO™ toolkit with FPGA Hello World Face Detection Exercise](https://github.com/fritzboyle/openvino-with-fpga-hello-world-face-detection) to learn more about how the software and hardware work together.

## Additional Resources

Intel® Distribution of OpenVINO™ toolkit home page: [https://software.intel.com/en-us/openvino-toolkit](https://software.intel.com/en-us/openvino-toolkit)

Intel® Distribution of OpenVINO™ toolkit documentation: [https://docs.openvinotoolkit.org/](https://docs.openvinotoolkit.org/)

Inference Engine FPGA plugin documentation: [https://docs.openvinotoolkit.org/latest/_docs_IE_DG_supported_plugins_FPGA.html](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_supported_plugins_FPGA.html)
