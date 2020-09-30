# Configuration Guide for the Intel® Distribution of OpenVINO™ toolkit and the Intel® Vision Accelerator Design with an Intel® Arria® 10 FPGA SG2 (IEI's Mustang-F100-A10) on Windows* {#openvino_docs_install_guides_VisionAcceleratorFPGA_Configure_Windows}

> **NOTE**: Intel® Arria® 10 FPGA (Mustang-F100-A10) Speed Grade 1 is not available in the OpenVINO 2020.3 package.

## 1. Configure and Set Up the Intel® Vision Accelerator Design with an Intel® Arria® 10 FPGA

1. Download [Intel® Quartus® Prime Programmer and Tools Standard Edition 18.1](http://fpgasoftware.intel.com/18.1/?edition=standard&platform=windows&download_manager=direct#tabs-4). Install the Intel® Quartus® Prime Programmer and Tools Software to the `C:\intelFPGA\18.1` directory.

2. Download [OpenSSL](http://slproweb.com/download/Win64OpenSSL_Light-1_1_1f.exe). Install the OpenSSL and add the `<install location>\bin` path to your system `PATH` variable.

3. Unpack the BSP for your Intel® Vision Accelerator Design with an Intel® Arria® 10 FPGA SG2: 
Extract `Intel_vision_accel_win_driver_1.2_SG2.zip` from `C:\Program Files (x86)\IntelSWTools\openvino\a10_vision_design_sg2_bitstreams\BSP` to `C:\intelFPGA\19.2\aclrte-windows64\board`
5. Open an admin command prompt.
6. Setup your environment variables:
```sh
set INTELFPGAOCLSDKROOT=C:\intelFPGA\19.2\aclrte-windows64
set AOCL_BOARD_PACKAGE_ROOT=%INTELFPGAOCLSDKROOT%\board\a10_1150_sg2
set IOCL_BOARD_PACKAGE_ROOT=%INTELFPGAOCLSDKROOT%\board\a10_1150_sg2
C:\intelFPGA\19.2\aclrte-windows64\init_opencl.bat
"C:\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat"
```
7. Uninstall any previous BSP before installing the OpenCL BSP for the 2020.3 BSP. Enter **Y** when prompted to uninstall:
```sh
aocl uninstall
```
8. Install the new BSP. Enter **Y** when prompted to install
```sh
aocl install
```
9. Run `aocl diagnose`:
```sh
aocl diagnose
```
Your screen displays `DIAGNOSTIC_PASSED`.

## 2. Program a Bitstream

The bitstream you program should correspond to the topology you want to deploy. In this section, you program a SqueezeNet bitstream and deploy the classification sample with a SqueezeNet model that you used the Model Optimizer to convert in the steps before.

> **IMPORTANT**: Only use bitstreams from the installed version of the Intel® Distribution of OpenVINO™ toolkit. Bitstreams from older versions of the Intel® Distribution of OpenVINO™ toolkit are incompatible with later versions of the Intel® Distribution of OpenVINO™ toolkit. For example, you cannot use the `2019R4_PL2_FP11_AlexNet_GoogleNet_Generic` bitstream, when the Intel® Distribution of OpenVINO™ toolkit supports the `2020-3_PL2_FP11_AlexNet_GoogleNet_Generic` bitstream.

Depending on how many bitstreams you selected, there are different folders for each FPGA card type which were downloaded in the Intel® Distribution of OpenVINO™ toolkit package:

1. For the Intel® Vision Accelerator Design with Intel® Arria® 10 FPGA SG2, the pre-trained bitstreams are in `C:\Program Files (x86)\IntelSWTools\openvino\a10_vision_design_sg2_bitstreams`. This example uses a SqueezeNet bitstream with low precision for the classification sample.

2. Program the bitstream for the Intel® Vision Accelerator Design with Intel® Arria® 10 FPGA SG2:
```sh
aocl program acl0 "C:\Program Files (x86)\IntelSWTools\openvino\a10_vision_design_sg2_bitstreams/2020-3_PL2_FP16_SqueezeNet_TinyYolo_VGG.aocx"
```

## 3. Set up a Sample Neural Network Model for FPGA

> **NOTE**: The SqueezeNet Caffe* model was already downloaded and converted to an FP16 IR when you ran the Image Classification Verification Script while [installing the Intel® Distribution of OpenVINO™ toolkit for Windows* with FPGA Support](installing-openvino-windows-fpga.md). Read this section only if you want to convert the model manually, otherwise skip and go to the next section to run the Image Classification sample application.

In this section, you will prepare a sample FP16 model suitable for hardware accelerators. For more information, see the [FPGA plugin](../IE_DG/supported_plugins/FPGA.html) section in the Inference Engine Developer Guide.

1. Create a directory for the FP16 SqueezeNet Model:
```sh
mkdir %HOMEPATH%\squeezenet1.1_FP16
```
	
2. Go to `%HOMEPATH%\squeezenet1.1_FP16`:
```sh
cd %HOMEPATH%\squeezenet1.1_FP16
```

3. Use the Model Optimizer to convert the FP16 SqueezeNet Caffe* model into an FP16 optimized Intermediate Representation (IR). The model files were downloaded when you ran the the Image Classification verification script while [installing the Intel® Distribution of OpenVINO™ toolkit for Windows* with FPGA Support](installing-openvino-windows-fpga.md). To convert, run the Model Optimizer script with the following arguments:	
```sh
python "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\model_optimizer\mo.py" --input_model %HOMEPATH%\Documents\Intel\OpenVINO\openvino_models\models\public\squeezenet1.1\squeezenet1.1.caffemodel --data_type FP16 --output_dir .
```
	
4. The `squeezenet1.1.labels` file contains the classes `ImageNet` uses. This file is included so that the inference results show text instead of classification numbers. Copy `squeezenet1.1.labels` to the your optimized model location:
```sh
xcopy "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\demo\squeezenet1.1.labels" .
```
	
5. Copy a sample image to the release directory. You will use this with your optimized model:
```sh
xcopy "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\demo\car.png" %HOMEPATH%\Documents\Intel\OpenVINO\inference_engine_samples_build\intel64\Release
```

## 4. Run the Image Classification Sample Application

In this section you will run the Image Classification sample application, with the Caffe* Squeezenet1.1 model on your Intel® Vision Accelerator Design with an Intel® Arria® 10 FPGA. 

Image Classification sample application binary file was automatically built and the FP16 model IR files are created when you ran the Image Classification Verification Script while [installing the Intel® Distribution of OpenVINO™ toolkit for Windows* with FPGA Support](installing-openvino-windows-fpga.md):
* Compiled sample Application binaries are located in the `%HOMEPATH%\Documents\Intel\OpenVINO\inference_engine_samples_build\intel64\Release` folder.
* Generated IR files are in the `%HOMEPATH%\Documents\Intel\OpenVINO\openvino_models\ir\public\squeezenet1.1\FP16` folder.

1. Go to the samples directory
```sh
cd %HOMEPATH%\Documents\Intel\OpenVINO\inference_engine_samples_build\intel64\Release
```

2. Use an Inference Engine sample to run a sample inference on the CPU:
```sh
classification_sample_async -i car.png -m %HOMEPATH%\Documents\Intel\OpenVINO\openvino_models\ir\public\squeezenet1.1\FP16\squeezenet1.1.xml
```
Note the CPU throughput in Frames Per Second (FPS). This tells you how quickly the inference is done on the hardware. Now run the inference using the FPGA.

3. Add the `-d` option to target the FPGA:
```sh
classification_sample_async -i car.png -m %HOMEPATH%\Documents\Intel\OpenVINO\openvino_models\ir\public\squeezenet1.1\FP16\squeezenet1.1.xml -d HETERO:FPGA,CPU
```
The throughput on FPGA is listed and may show a lower FPS. This may be due to the initialization time. To account for that, increase the number of iterations or batch size when deploying to get a better sense of the speed the FPGA can run inference at.

Congratulations, you are done with the Intel® Distribution of OpenVINO™ toolkit installation for FPGA. To learn more about how the Intel® Distribution of OpenVINO™ toolkit works, try the other resources that are provided below.

## Additional Resources

* Intel® Distribution of OpenVINO™ toolkit home page: [https://software.intel.com/en-us/openvino-toolkit](https://software.intel.com/en-us/openvino-toolkit).
* Intel® Distribution of OpenVINO™ toolkit documentation: [https://docs.openvinotoolkit.org](https://docs.openvinotoolkit.org).
* [Inference Engine FPGA plugin documentation](../IE_DG/supported_plugins/FPGA.md).
