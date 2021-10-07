# Getting Started with Demo Scripts and Demo Applications {#openvino_get_started_demos_samples}

## Demo Scripts

A set of demo scripts in the `openvino_2021/deployment_tools/demo` directory give you a starting point for learning the OpenVINO workflow. These scripts automatically perform the workflow steps to demonstrate running inference pipelines for different scenarios. The demo steps let you see how to: 
* Compile several samples from the source files delivered as part of the OpenVINO toolkit.
* Download trained models.
* Convert the models to IR (Intermediated Representation format used by OpenVINO™) with Model Optimizer.
* Perform pipeline steps and see the output on the console.

This guide assumes you completed all installation and configuration steps. If you have not yet installed and configured the toolkit:

@sphinxdirective
.. tab:: Linux
See [Install Intel® Distribution of OpenVINO™ toolkit for Linux*](../install_guides/installing-openvino-linux.md)
.. tab:: Windows
See [Install Intel® Distribution of OpenVINO™ toolkit for Windows*](../install_guides/installing-openvino-windows.md)
.. tab:: macOS
See [Install Intel® Distribution of OpenVINO™ toolkit for macOS*](../install_guides/installing-openvino-macos.md)
@endsphinxdirective

The demo scripts can run inference on any [supported target device](https://software.intel.com/en-us/openvino-toolkit/hardware). Although the default inference device (i.e., processor) is the CPU, you can add the `-d` parameter to specify a different inference device. The general command to run a demo script is as follows:

@sphinxdirective
.. tab:: Linux
.. code-block:: sh
cd /opt/intel/openvino_2021/deployment_tools/demo/
#If you installed in a location other than /opt/intel, substitute that path.
./<script_name> -d [CPU, GPU, MYRIAD, HDDL]
.. tab:: Windows
.. code-block:: bat
cd "C:\Program Files (x86)\Intel\openvino_2021\deployment_tools\demo"
#If you installed in a location other than the default, substitute that path.
.\<script_name> -d [CPU, GPU, MYRIAD, HDDL]
.. tab:: macOS
.. code-block:: sh
TBD
@endsphinxdirective

Before running the demo applications on Intel® Processor Graphics or on an Intel® Neural Compute Stick 2 device, you must complete additional configuration steps. 

@sphinxdirective
.. tab:: Linux
For details, see the following sections in the [installation instructions](../install_guides/installing-openvino-linux.md):
* Steps for Intel® Processor Graphics (GPU) 
* Steps for Intel® Neural Compute Stick 2
.. tab:: Windows
For details, see the following sections in the [installation instructions](../install_guides/installing-openvino-windows.md):
* Additional Installation Steps for Intel® Processor Graphics (GPU)
* Additional Installation Steps for Intel® Vision Accelerator Design with Intel® Movidius™ VPUs
.. tab:: macOS
Intel® Processor Graphics (GPU) and Intel® Neural Compute Stick 2 processors are not compatible with macOS.
@endsphinxdirective

The following sections describe each demo script.

### Image Classification Demo Script
The `demo_squeezenet_download_convert_run` script illustrates the image classification pipeline.

The script: 
1. Downloads a SqueezeNet model. 
2. Runs the Model Optimizer to convert the model to the IR format used by OpenVINO™.
3. Builds the Image Classification Sample Async application.
4. Runs the compiled sample with the `car.png` image located in the `demo` directory.

@sphinxdirective
.. raw:: html
    <div class="collapsible-section">
@endsphinxdirective
**Click for an example of running the Image Classification demo script**

To preview the image that the script will classify:

@sphinxdirective
.. tab:: Linux
.. code-block:: sh
   cd ${INTEL_OPENVINO_DIR}/deployment_tools/demo
   eog car.png
.. tab:: Windows
.. code-block:: bat
   TBD
.. tab:: macOS
.. code-block:: sh
   TBD
@endsphinxdirective

To run the script and perform inference on a CPU:

@sphinxdirective
.. tab:: Linux
.. code-block:: sh
   ./demo_squeezenet_download_convert_run.sh
.. tab:: Windows
.. code-block:: bat
   .\demo_squeezenet_download_convert_run.bat
.. tab:: macOS
.. code-block:: sh
   TBD
@endsphinxdirective

When the script completes, you see the label and confidence for the top 10 categories:

@sphinxdirective
.. tab:: Linux
.. code-block:: sh
Top 10 results:

Image /opt/intel/openvino_2021/deployment_tools/demo/car.png

classid probability label
------- ----------- -----
817     0.8363345   sports car, sport car
511     0.0946488   convertible
479     0.0419131   car wheel
751     0.0091071   racer, race car, racing car
436     0.0068161   beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon
656     0.0037564   minivan
586     0.0025741   half track
717     0.0016069   pickup, pickup truck
864     0.0012027   tow truck, tow car, wrecker
581     0.0005882   grille, radiator grille


total inference time: 2.6642941
Average running time of one iteration: 2.6642941 ms

Throughput: 375.3339402 FPS

[ INFO ] Execution successful

.. tab:: Windows
.. code-block:: bat
Top 10 results:

Image C:\Program Files (x86)\Intel\openvino_2021\deployment_tools\demo\car.png

classid probability label
------- ----------- -----
817     0.8363345   sports car, sport car
511     0.0946488   convertible
479     0.0419131   car wheel
751     0.0091071   racer, race car, racing car
436     0.0068161   beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon
656     0.0037564   minivan
586     0.0025741   half track
717     0.0016069   pickup, pickup truck
864     0.0012027   tow truck, tow car, wrecker
581     0.0005882   grille, radiator grille


total inference time: 2.6642941
Average running time of one iteration: 2.6642941 ms

Throughput: 375.3339402 FPS

[ INFO ] Execution successful

.. tab:: macOS
.. code-block:: sh
   TBD
@endsphinxdirective
@sphinxdirective
.. raw:: html
    </div>
@endsphinxdirective

### Inference Pipeline Demo Script
The `demo_security_barrier_camera` application uses vehicle recognition in which vehicle attributes build on each other to narrow in on a specific attribute.

The script:
1. Downloads three pre-trained models, already converted to IR format.
2. Builds the Security Barrier Camera Demo application.
3. Runs the application with the three models and the `car_1.bmp` image from the `demo` directory to show an inference pipeline.

This application:

1. Gets the boundaries an object identified as a vehicle with the first model.
2. Uses the vehicle identification as input to the second model, which identifies specific vehicle attributes, including the license plate.
3. Uses the the license plate as input to the third model, which recognizes specific characters in the license plate.

@sphinxdirective
.. raw:: html
    <div class="collapsible-section">
@endsphinxdirective
**Click for an example of Running the Pipeline demo script</strong></summary>**
    
To run the script performing inference on Intel® Processor Graphics (only on CPU for macOS):
@sphinxdirective
.. tab:: Linux
.. code-block:: sh
./demo_security_barrier_camera.sh -d GPU
.. tab:: Windows
.. code-block:: bat
.\demo_security_barrier_camera.bat -d GPU
.. tab:: macOS
.. code-block:: sh
./demo_security_barrier_camera.sh -d CPU
@endsphinxdirective

When the verification script is complete, you see an image that displays the resulting frame with detections rendered as bounding boxes and overlaid text:

@sphinxdirective
.. tab:: Linux
![](../img/inference_pipeline_script_lnx.png)
.. tab:: Windows
![](../img/inference_pipeline_script_win.png)
.. tab:: macOS
![](../img/inference_pipeline_script_mac.png)
@endsphinxdirective

@sphinxdirective
.. raw:: html
    </div>
@endsphinxdirective

### Benchmark Demo Script
The `demo_benchmark_app` script illustrates how to use the Benchmark Application to estimate deep learning inference performance on supported devices. 

The script: 
1. Downloads a SqueezeNet model.
2. Runs the Model Optimizer to convert the model to IR format.
3. Builds the Inference Engine Benchmark tool.
4. Runs the tool with the `car.png` image located in the `demo` directory.

@sphinxdirective
.. raw:: html
    <div class="collapsible-section">
@endsphinxdirective
**Click for an example of running the Benchmark demo script**

To run the script that performs measures inference performance:
@sphinxdirective
.. tab:: Linux
.. code-block:: sh
./demo_benchmark_app.sh
.. tab:: Windows
.. code-block:: bat
.\demo_benchmark_app.bat
.. tab:: macOS
.. code-block:: sh
./demo_benchmark_app.sh
@endsphinxdirective

When the verification script is complete, you see the performance counters, resulting latency, and throughput values displayed on the screen.
@sphinxdirective
.. raw:: html
    </div>
@endsphinxdirective

## <a name="using-sample-application"></a>Demo/Sample Applications

This section guides you through a simplified workflow for the Intel® Distribution of OpenVINO™ toolkit using code samples and demo applications.
You will perform the following steps:
1. <a href="#download-models">Use the Model Downloader to download suitable models.</a>
2. <a href="#convert-models-to-intermediate-representation">Convert the models with the Model Optimizer.</a> 
3. <a href="download-media">Download media files to run inference on.</a>
4. <a href="run-image-classification">Run inference on the sample and see the results:</a>
    - <a href="run-image-classification">Image Classification Code Sample</a>
    - <a href="run-security-barrier">Security Barrier Camera Demo application</a>

### Build Samples and Demos

If you have already built the demos and samples, you can skip this section. The build will take about 5-10 minutes, depending on your system.

To build OpenVINO samples:
@sphinxdirective
.. tab:: Linux
Go to the [Inference Engine Samples page](../IE_DG/Samples_Overview.md) and see the "Build the Sample Applications on Linux*" section.
.. tab:: Windows
Go to the [Inference Engine Samples page](../IE_DG/Samples_Overview.md) and see the "Build the Sample Applications on Microsoft Windows* OS" section.
.. tab:: macOS
Go to the [Inference Engine Samples page](../IE_DG/Samples_Overview.md) and see the "Build the Sample Applications on Linux*" section. You can use or adapt the Linux steps for macOS. 
@endsphinxdirective

To build OpenVINO demos:
@sphinxdirective
.. tab:: Linux
Go to the [Open Model Zoo Demos page](../IE_DG/Samples_Overview.md) and see the "Build the Demo Applications on Linux*" section.
.. tab:: Windows
Go to the [Open Model Zoo Demos page](../IE_DG/Samples_Overview.md) and see the "Build the Demo Applications on Microsoft Windows* OS" section.
.. tab:: macOS
Go to the [Open Model Zoo Demos page](../IE_DG/Samples_Overview.md) and see the "Build the Demo Applications on macOS*" section. You can use the requirements from "To build OpenVINO samples" above and adapt the Linux build steps for macOS.
@endsphinxdirective

### <a name="download-models"></a> Step 1: Download the Models

You must have a model that is specific for your inference task. Example model types are:
- Classification (AlexNet, GoogleNet, SqueezeNet, others): Detects one type of element in an image.
- Object Detection (SSD, YOLO): Draws bounding boxes around multiple types of objects in an image.
- Custom: Often based on SSD

Options to find a model suitable for the OpenVINO™ toolkit:
- Download public or Intel pre-trained models from the [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo) using the [Model Downloader tool](@ref omz_tools_downloader)
- Download from GitHub*, Caffe* Zoo, TensorFlow* Zoo, etc.
- Train your own model with machine learning tools
        
This guide uses the OpenVINO™ Model Downloader to get pre-trained models. You can use one of the following commands to find a model:

* **List the models available in the downloader**: 
@sphinxdirective
.. tab:: Linux
.. code-block:: sh
   cd /opt/intel/openvino_2021/deployment_tools/tools/model_downloader/
   python3 info_dumper.py --print_all
.. tab:: Windows
.. code-block:: bat
   cd <INSTALL_DIR>\deployment_tools\tools\model_downloader\
   python3 info_dumper.py --print_all
.. tab:: macOS
.. code-block:: sh
   cd <INSTALL_DIR>/openvino_2021/deployment_tools/tools/model_downloader/
   python3 info_dumper.py --print_all
@endsphinxdirective

* **Use `grep` to list models that have a specific name pattern**: 
```
python3 info_dumper.py --print_all | grep <model_name>
```

Use the Model Downloader to download the models to a models directory. This guide uses `<models_dir>` and `<models_name>' as placeholders for the models directory and model name:
@sphinxdirective
.. tab:: Linux
Always run the Model Downloader with the `sudo` command. 
.. code-block:: sh
sudo python3 ./downloader.py --name <model_name> --output_dir <models_dir>
.. tab:: Windows
.. code-block:: bat
python3 .\downloader.py --name <model_name> --output_dir <models_dir>
.. tab:: macOS
Always run the Model Downloader with the `sudo` command. 
.. code-block:: sh
sudo python3 ./downloader.py --name <model_name> --output_dir <models_dir>
@endsphinxdirective

Download the following models to run the Image Classification Sample and Security Barrier Camera Demo applications:

|Model Name                                     | Code Sample or Demo App                  |
|-----------------------------------------------|------------------------------------------|
|`squeezenet1.1`                                | Image Classification Sample              |
|`vehicle-license-plate-detection-barrier-0106` | Security Barrier Camera Demo             |
|`vehicle-attributes-recognition-barrier-0039`  | Security Barrier Camera Demo             |
|`license-plate-recognition-barrier-0001`       | Security Barrier Camera Demo             |

@sphinxdirective
.. raw:: html
    <div class="collapsible-section">
@endsphinxdirective
**Click for an example of downloading the SqueezeNet Caffe* model**
To download the SqueezeNet 1.1 Caffe* model to the `models` folder:

@sphinxdirective
.. tab:: Linux
.. code-block:: sh
sudo python3 ./downloader.py --name squeezenet1.1 --output_dir ~/models
.. tab:: Windows
.. code-block:: bat
python .\downloader.py --name squeezenet1.1 --output_dir C:\Users\<username>\Documents\models
.. tab:: macOS
.. code-block:: sh
sudo python3 ./downloader.py --name squeezenet1.1 --output_dir ~/models
@endsphinxdirective

Your screen looks similar to this after the download and shows the paths of downloaded files:
@sphinxdirective
.. tab:: Linux
.. code-block:: sh
###############|| Downloading models ||###############

========= Downloading /home/username/models/public/squeezenet1.1/squeezenet1.1.prototxt

========= Downloading /home/username/models/public/squeezenet1.1/squeezenet1.1.caffemodel
... 100%, 4834 KB, 3157 KB/s, 1 seconds passed

###############|| Post processing ||###############

========= Replacing text in /home/username/models/public/squeezenet1.1/squeezenet1.1.prototxt =========
.. tab:: Windows
.. code-block:: bat
################|| Downloading models ||################

========== Downloading C:\Users\username\Documents\models\public\squeezenet1.1\squeezenet1.1.prototxt
... 100%, 9 KB, ? KB/s, 0 seconds passed

========== Downloading C:\Users\username\Documents\models\public\squeezenet1.1\squeezenet1.1.caffemodel
... 100%, 4834 KB, 571 KB/s, 8 seconds passed

################|| Post-processing ||################

========== Replacing text in C:\Users\username\Documents\models\public\squeezenet1.1\squeezenet1.1.prototxt

@endsphinxdirective

@sphinxdirective
.. raw:: html
    </div>
@endsphinxdirective

@sphinxdirective
.. raw:: html
    <div class="collapsible-section">
@endsphinxdirective
**Click for an example of downloading models for the Security Barrier Camera Demo application**

To download all three pre-trained models in FP16 precision to the `models` folder in your home folder:   

@sphinxdirective
.. tab:: Linux
.. code-block:: sh
python ./downloader.py --name vehicle-license-plate-detection-barrier-0106,vehicle-attributes-recognition-barrier-0039,license-plate-recognition-barrier-0001 --output_dir ~/models --precisions FP16
.. tab:: Windows
.. code-block:: bat
python .\downloader.py --name vehicle-license-plate-detection-barrier-0106,vehicle-attributes-recognition-barrier-0039,license-plate-recognition-barrier-0001 --output_dir C:\Users\<username>\Documents\models --precisions FP16
.. tab:: macOS
.. code-block:: sh
python ./downloader.py --name vehicle-license-plate-detection-barrier-0106,vehicle-attributes-recognition-barrier-0039,license-plate-recognition-barrier-0001 --output_dir ~/models --precisions FP16
@endsphinxdirective

Your screen looks similar to this after the download:

@sphinxdirective
.. tab:: Linux
.. code-block:: sh
################|| Downloading models ||################

========== Downloading /home/username/models/intel/vehicle-license-plate-detection-barrier-0106/FP16/vehicle-license-plate-detection-barrier-0106.xml
... 100%, 204 KB, 183949 KB/s, 0 seconds passed

========== Downloading /home/username/models/intel/vehicle-license-plate-detection-barrier-0106/FP16/vehicle-license-plate-detection-barrier-0106.bin
... 100%, 1256 KB, 3948 KB/s, 0 seconds passed

========== Downloading /home/username/models/intel/vehicle-attributes-recognition-barrier-0039/FP16/vehicle-attributes-recognition-barrier-0039.xml
... 100%, 32 KB, 133398 KB/s, 0 seconds passed

========== Downloading /home/username/models/intel/vehicle-attributes-recognition-barrier-0039/FP16/vehicle-attributes-recognition-barrier-0039.bin
... 100%, 1222 KB, 3167 KB/s, 0 seconds passed

========== Downloading /home/username/models/intel/license-plate-recognition-barrier-0001/FP16/license-plate-recognition-barrier-0001.xml
... 100%, 47 KB, 85357 KB/s, 0 seconds passed

========== Downloading /home/username/models/intel/license-plate-recognition-barrier-0001/FP16/license-plate-recognition-barrier-0001.bin
... 100%, 2378 KB, 5333 KB/s, 0 seconds passed

################|| Post-processing ||################
.. tab:: Windows
.. code-block:: bat
################|| Downloading models ||################

========== Downloading C:\Users\username\Documents\models\intel\vehicle-license-plate-detection-barrier-0106\FP16\vehicle-license-plate-detection-barrier-0106.xml
... 100%, 207 KB, 13810 KB/s, 0 seconds passed

========== Downloading C:\Users\username\Documents\models\intel\vehicle-license-plate-detection-barrier-0106\FP16\vehicle-license-plate-detection-barrier-0106.bin
... 100%, 1256 KB, 70 KB/s, 17 seconds passed

========== Downloading C:\Users\username\Documents\models\intel\vehicle-attributes-recognition-barrier-0039\FP16\vehicle-attributes-recognition-barrier-0039.xml
... 100%, 32 KB, ? KB/s, 0 seconds passed

========== Downloading C:\Users\username\Documents\models\intel\vehicle-attributes-recognition-barrier-0039\FP16\vehicle-attributes-recognition-barrier-0039.bin
... 100%, 1222 KB, 277 KB/s, 4 seconds passed

========== Downloading C:\Users\username\Documents\models\intel\license-plate-recognition-barrier-0001\FP16\license-plate-recognition-barrier-0001.xml
... 100%, 47 KB, ? KB/s, 0 seconds passed

========== Downloading C:\Users\username\Documents\models\intel\license-plate-recognition-barrier-0001\FP16\license-plate-recognition-barrier-0001.bin
... 100%, 2378 KB, 120 KB/s, 19 seconds passed

################|| Post-processing ||################
.. tab:: macOS
.. code-block:: sh
   TBD
@endsphinxdirective

@sphinxdirective
.. raw:: html
    </div>
@endsphinxdirective

### Step 2: <a name="convert-models-to-intermediate-representation">Convert the Model with Model Optimizer</a>

In this step, your trained models are ready to run through the Model Optimizer to convert them to the IR (Intermediate Representation) format. For most model types, this is required before using the Inference Engine with the model.

Models in the IR format always include an `.xml` and `.bin` file and may also include other files such as `.json` or `.mapping`. Make sure you have these files together in a single directory so the Inference Engine can find them.

REQUIRED: `model_name.xml` 
REQUIRED: `model_name.bin` 
OPTIONAL: `model_name.json`,  `model_name.mapping`, etc.

This tutorial uses the public SqueezeNet 1.1 Caffe* model to run the Image Classification Sample. See the example in the Download Models section of this page to learn how to download this model.

The SqueezeNet1.1 model is downloaded in the Caffe* format. You must use the Model Optimizer to convert the model to IR. The `vehicle-license-plate-detection-barrier-0106`, `vehicle-attributes-recognition-barrier-0039`, and `license-plate-recognition-barrier-0001` models are downloaded in IR format. You don't need to use the Model Optimizer on them because they are Intel models that have previously been converted. Public models will need converting with Model Optimizer.

Create an `<ir_dir>` directory to contain the model's Intermediate Representation (IR).

@sphinxdirective
.. tab:: Linux
.. code-block:: sh
mkdir ~/ir
.. tab:: Windows
.. code-block:: bat
mkdir C:\Users\<username>\Documents\ir
.. tab:: macOS
.. code-block:: sh
mkdir ~/ir
@endsphinxdirective

The Inference Engine can perform inference on different precision formats, such as FP32, FP16, or INT8. To generate an IR with a specific precision, run the Model Optimizer with the appropriate `--data_type` option.

Generic Model Optimizer script:

@sphinxdirective
.. tab:: Linux
.. code-block:: sh
cd /opt/intel/openvino/deployment_tools/model_optimizer
python3 ./mo.py --input_model <model_dir>/<model_file> --data_type <model_precision> --output_dir <ir_dir>
.. tab:: Windows
.. code-block:: bat
cd <INSTALL_DIR>\deployment_tools\model_optimizer
python .\mo.py --input_model <model_dir>\<model_file> --data_type <model_precision> --output_dir <ir_dir>
.. tab:: macOS
.. code-block:: sh
cd /opt/intel/openvino/deployment_tools/model_optimizer
python3 ./mo.py --input_model <model_dir>/<model_file> --data_type <model_precision> --output_dir <ir_dir>
@endsphinxdirective

IR files produced by the script are written to the <ir_dir> directory.

The command with most placeholders filled in and FP16 precision:

@sphinxdirective
.. tab:: Linux
.. code-block:: sh
cd /opt/intel/openvino/deployment_tools/model_optimizer
python3 ./mo.py --input_model ~/models/public/squeezenet1.1/squeezenet1.1.caffemodel --data_type FP16 --output_dir ~/ir
.. tab:: Windows
.. code-block:: bat
cd <INSTALL_DIR>\deployment_tools\model_optimizer
python .\mo.py --input_model C:\Users\<username>\Documents\models\public\squeezenet1.1\squeezenet1.1.caffemodel --data_type FP16 --output_dir C:\Users\<username>\Documents\ir
.. tab:: macOS
.. code-block:: sh
cd /opt/intel/openvino/deployment_tools/model_optimizer
python3 ./mo.py --input_model ~/models/public/squeezenet1.1/squeezenet1.1.caffemodel --data_type FP16 --output_dir ~/ir
@endsphinxdirective

### <a name="download-media"></a> Step 3: Download a Video or Still Photo as Media

Many sources are available from which you can download video media to use the code samples and demo applications. Possibilities include: 
- https://pexels.com
- https://images.google.com

As an alternative, the Intel® Distribution of OpenVINO™ toolkit includes several sample images and videos that you can use for running code samples and demo applications:
@sphinxdirective
.. tab:: Linux
* `/opt/intel/openvino_2021/deployment_tools/demo/car.png`
* `/opt/intel/openvino_2021/deployment_tools/demo/car_1.bmp`
* [Sample images and video](https://storage.openvinotoolkit.org/data/test_data/)
* [Sample videos](https://github.com/intel-iot-devkit/sample-videos)
.. tab:: Windows
.. code-block:: bat
* <INSTALL_DIR>\deployment_tools\demo\car.png
* <INSTALL_DIR>\deployment_tools\demo\car_1.bmp
* [Sample images and video](https://storage.openvinotoolkit.org/data/test_data/)
* [Sample videos](https://github.com/intel-iot-devkit/sample-videos)
.. tab:: macOS
.. code-block:: sh
* `<INSTALL_DIR>/openvino_2021/deployment_tools/demo/car.png`
* `<INSTALL_DIR>/intel/openvino_2021/deployment_tools/demo/car_1.bmp`
* [Sample images and video](https://storage.openvinotoolkit.org/data/test_data/)
* [Sample videos](https://github.com/intel-iot-devkit/sample-videos)
@endsphinxdirective

### <a name="run-image-classification"></a>Step 4: Run Inference on the Sample

#### Run the Image Classification Code Sample

To run the **Image Classification** code sample with an input image on the IR model: 

1. Set up the OpenVINO environment variables:
@sphinxdirective
.. tab:: Linux
.. code-block:: sh
   source /opt/intel/openvino/bin/setupvars.sh
.. tab:: Windows
.. code-block:: bat
   <INSTALL_DIR>\openvino\bin\setupvars.bat
.. tab:: macOS
.. code-block:: sh
   source <INSTALL_DIR>/openvino/bin/setupvars.sh
@endsphinxdirective

2. Go to the code samples release directory created when you built the samples earlier:
@sphinxdirective
.. tab:: Linux
.. code-block:: sh
   cd ~/inference_engine_cpp_samples_build/intel64/Release
.. tab:: Windows
.. code-block:: bat
   cd C:\Users\<username>\Documents\Intel\OpenVINO\inference_engine_samples_build\intel64\Release
.. tab:: macOS
.. code-block:: sh
   cd ~/inference_engine_cpp_samples_build/intel64/Release
@endsphinxdirective

3. Run the code sample executable, specifying the input media file, the IR for your model, and a target device for performing inference:

@sphinxdirective
.. tab:: Linux
.. code-block:: sh
   classification_sample_async -i <path_to_media> -m <path_to_model> -d <target_device>
.. tab:: Windows
.. code-block:: bat
   classification_sample_async.exe -i <path_to_media> -m <path_to_model> -d <target_device>
.. tab:: macOS
.. code-block:: sh
   classification_sample_async -i <path_to_media> -m <path_to_model> -d <target_device>
@endsphinxdirective

@sphinxdirective
.. raw:: html
    <div class="collapsible-content”>
@endsphinxdirective
**Click for examples of running the Image Classification code sample on different devices**

The following commands run the Image Classification Code Sample using the `car.png` file from the `demo` directory as an input image, the model in IR format from the `ir` directory, and on different hardware devices:

   **CPU:**  
@sphinxdirective
.. tab:: Linux
.. code-block:: sh
   ./classification_sample_async -i /opt/intel/openvino/deployment_tools/demo/car.png -m ~/models/public/squeezenet1.1/ir/squeezenet1.1.xml -d CPU
.. tab:: Windows
.. code-block:: bat
   .\classification_sample_async -i <INSTALL_DIR>\deployment_tools\demo\car.png -m C:\Users\<username>\Documents\models\public\squeezenet1.1\ir\squeezenet1.1.xml -d CPU
.. tab:: macOS
.. code-block:: sh
   ./classification_sample_async -i /opt/intel/openvino/deployment_tools/demo/car.png -m ~/models/public/squeezenet1.1/ir/squeezenet1.1.xml -d CPU
@endsphinxdirective

   **GPU:**
   >**NOTE**: Running inference on Intel® Processor Graphics (GPU) requires 
    [additional hardware configuration steps](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html#additional-GPU-steps), as described earlier on this page.
    
@sphinxdirective
.. tab:: Linux
.. code-block:: sh
   ./classification_sample -i /opt/intel/openvino/deployment_tools/demo/car.png -m ~/models/public/squeezenet1.1/ir/squeezenet1.1.xml -d GPU
.. tab:: Windows
.. code-block:: bat
   .\classification_sample_async -i <INSTALL_DIR>\deployment_tools\demo\car.png -m C:\Users\<username>\Documents\models\public\squeezenet1.1\ir\squeezenet1.1.xml -d GPU
@endsphinxdirective
   
   **MYRIAD:** 

   >**NOTE**: Running inference on VPU devices (Intel® Movidius™ Neural Compute 
   Stick or Intel® Neural Compute Stick 2) with the MYRIAD plugin requires 
    [additional hardware configuration steps](inference-engine/README.md#optional-additional-installation-steps-for-the-intel-movidius-neural-compute-stick-and-neural-compute-stick-2), as described earlier on this page.
@sphinxdirective
.. tab:: Linux
.. code-block:: sh
   ./classification_sample -i /opt/intel/openvino/deployment_tools/demo/car.png -m ~/models/public/squeezenet1.1/ir/squeezenet1.1.xml -d MYRIAD
.. tab:: Windows
.. code-block:: bat
   .\classification_sample_async -i <INSTALL_DIR>\deployment_tools\demo\car.png -m C:\Users\<username>\Documents\models\public\squeezenet1.1\ir\squeezenet1.1.xml -d MYRIAD
@endsphinxdirective

When the sample application is complete, you see the label and confidence for the top 10 categories on the display. Below is a sample output with inference results on CPU:    

@sphinxdirective
.. tab:: Linux
.. code-block:: sh
Top 10 results:

Image /opt/intel/deployment-tools/demo/car.png

classid probability label
------- ----------- -----
817     0.8363345   sports car, sport car
511     0.0946488   convertible
479     0.0419131   car wheel
751     0.0091071   racer, race car, racing car
436     0.0068161   beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon
656     0.0037564   minivan
586     0.0025741   half track
717     0.0016069   pickup, pickup truck
864     0.0012027   tow truck, tow car, wrecker
581     0.0005882   grille, radiator grille


total inference time: 2.6642941
Average running time of one iteration: 2.6642941 ms

Throughput: 375.3339402 FPS

[ INFO ] Execution successful
.. tab:: Windows
.. code-block:: bat
Top 10 results:

Image C:\Program Files (x86)\Intel\openvino_2021\deployment_tools\demo\car.png

classid probability label
------- ----------- -----
817     0.8364177   sports car, sport car
511     0.0945683   convertible
479     0.0419195   car wheel
751     0.0091233   racer, race car, racing car
436     0.0068038   beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon
656     0.0037315   minivan
586     0.0025940   half track
717     0.0016044   pickup, pickup truck
864     0.0012045   tow truck, tow car, wrecker
581     0.0005833   grille, radiator grille

[ INFO ] Execution successful
.. tab:: macOS
.. code-block:: sh
   TBD
@endsphinxdirective
@sphinxdirective
.. raw:: html
    </div>
@endsphinxdirective

#### <a name="run-security-barrier"></a>Run the Security Barrier Camera Demo Application

To run the **Security Barrier Camera Demo Application** using an input image on the prepared IR models:

1. Set up the OpenVINO environment variables:
@sphinxdirective
.. tab:: Linux
.. code-block:: sh
   source /opt/intel/openvino/bin/setupvars.sh
.. tab:: Windows
.. code-block:: bat
   <INSTALL_DIR>\bin\setupvars.bat
.. tab:: macOS
.. code-block:: sh
   TBD
@endsphinxdirective

2. Go to the demo application build directory:
@sphinxdirective
.. tab:: Linux
.. code-block:: sh
   cd ~/inference_engine_demos_build/intel64/Release
.. tab:: Windows
.. code-block:: bat
   cd C:\Users\<USER_ID>\Documents\Intel\OpenVINO\inference_engine_demos_build\intel64\Release
.. tab:: macOS
.. code-block:: sh
   TBD
@endsphinxdirective

3. Run the demo executable, specifying the input media file, list of model IRs, and a target device for performing inference:
@sphinxdirective
.. tab:: Linux
.. code-block:: sh
   ./security_barrier_camera_demo -i <path_to_media> -m <path_to_model>/vehicle-license-plate-detection-barrier-0106.xml -m_va <path_to_vehicle_attributes model>/vehicle-attributes-recognition-barrier-0039.xml -m_lpr <path_to_license_plate_recognition_model>/license-plate-recognition-barrier-0001.xml -d <target_device>
.. tab:: Windows
.. code-block:: bat
   .\security_barrier_camera_demo -i <path_to_media> -m <path_to_vehicle-license-plate-detection_model_xml> -m_va <path_to_vehicle_attributes_model_xml> -m_lpr <path_to_license_plate_recognition_model_xml> -d <target_device>
.. tab:: macOS
.. code-block:: sh
      ./security_barrier_camera_demo -i <path_to_media> -m <path_to_model>/vehicle-license-plate-detection-barrier-0106.xml -m_va <path_to_vehicle_attributes model>/vehicle-attributes-recognition-barrier-0039.xml -m_lpr <path_to_license_plate_recognition_model>/license-plate-recognition-barrier-0001.xml -d <target_device>
@endsphinxdirective

@sphinxdirective
.. raw:: html
    <div class="collapsible-content”>
@endsphinxdirective
**Click for examples of running the Security Barrier Camera demo application on different devices**

**CPU:**
@sphinxdirective
.. tab:: Linux
.. code-block:: sh
./security_barrier_camera_demo -i /opt/intel/openvino/deployment_tools/demo/car_1.bmp -m <path_to_model>/vehicle-license-plate-detection-barrier-0106.xml -m_va <path_to_model>/vehicle-attributes-recognition-barrier-0039.xml -m_lpr <path_to_model>/license-plate-recognition-barrier-0001.xml -d CPU
.. tab:: Windows
.. code-block:: bat
   .\security_barrier_camera_demo -i <INSTALL_DIR>\deployment_tools\demo\car_1.bmp -m C:\Users\username\Documents\models\intel\vehicle-license-plate-detection-barrier-0106\FP16\vehicle-license-plate-detection-barrier-0106.xml -m_va C:\Users\username\Documents\models\intel\vehicle-attributes-recognition-barrier-0039\FP16\vehicle-attributes-recognition-barrier-0039.xml -m_lpr C:\Users\username\Documents\models\intel\license-plate-recognition-barrier-0001\FP16\license-plate-recognition-barrier-0001.xml -d CPU
.. tab:: macOS
.. code-block:: sh
   TBD
@endsphinxdirective

**GPU:**
>**NOTE**: Running inference on Intel® Processor Graphics (GPU) requires 
    [additional hardware configuration steps](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html#additional-GPU-steps), as described earlier on this page.

@sphinxdirective
.. tab:: Linux
.. code-block:: sh
./security_barrier_camera_demo -i /opt/intel/openvino/deployment_tools/demo/car_1.bmp -m <path_to_model>/vehicle-license-plate-detection-barrier-0106.xml -m_va <path_to_model>/vehicle-attributes-recognition-barrier-0039.xml -m_lpr <path_to_model>/license-plate-recognition-barrier-0001.xml -d GPU
.. tab:: Windows
.. code-block:: bat
   .\security_barrier_camera_demo -i <INSTALL_DIR>\deployment_tools\demo\car_1.bmp -m <path_to_model>/vehicle-license-plate-detection-barrier-0106.xml -m_va <path_to_model>/vehicle-attributes-recognition-barrier-0039.xml -m_lpr <path_to_model>/license-plate-recognition-barrier-0001.xml -d GPU
.. tab:: macOS
.. code-block:: sh
   TBD
@endsphinxdirective

**MYRIAD:** 
   >**NOTE**: Running inference on VPU devices (Intel® Movidius™ Neural Compute 
   Stick or Intel® Neural Compute Stick 2) with the MYRIAD plugin requires 
    [additional hardware configuration steps](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html#additional-NCS-steps), as described earlier on this page.
    
@sphinxdirective
.. tab:: Linux
.. code-block:: sh
   ./classification_sample -i /opt/intel/deployment-tools/demo/car.png -m <ir_dir>/squeezenet1.1.xml -d MYRIAD
.. tab:: Windows
.. code-block:: bat
   .\security_barrier_camera_demo -i <INSTALL_DIR>\deployment_tools\demo\car_1.bmp -m <path_to_model>/vehicle-license-plate-detection-barrier-0106.xml -m_va <path_to_model>/vehicle-attributes-recognition-barrier-0039.xml -m_lpr <path_to_model>/license-plate-recognition-barrier-0001.xml -d MYRIAD
.. tab:: macOS
.. code-block:: sh
   TBD
@endsphinxdirective

@sphinxdirective
.. raw:: html
    </div>
@endsphinxdirective

## Exercises

The following exercises will guide you through using samples with gradually less specific help. As you move through each exercise, you will get a sense of how to use OpenVINO™ in more sophisticated use cases.

In these exercises, you will:
1. Convert and optimize a neural network model to work on Intel® hardware.
2. Run computer vision applications using optimized models and appropriate media.
   - During optimization with the DL Workbench™, a subset of ImageNet* and VOC* images are used.
   - When running samples, we'll use an image or video file located on this system.
> **NOTE**: Before starting these sample exercises, change directories into the samples directory:
@sphinxdirective
.. tab:: Linux
.. code-block:: sh
   cd ~/omz_demos_build/intel64/Release
.. tab:: Windows
.. code-block:: bat
   cd C:\Users\<USER_ID>\Documents\Intel\OpenVINO\inference_engine_demos_build\intel64\Release
.. tab:: macOS
.. code-block:: sh
   cd ~/omz_demos_build/intel64/Release
@endsphinxdirective

> **NOTE**: During this exercise you will move to multiple directories and occasionally copy files so that you don't have to specify full paths in commands. You are welcome to set up environment variables to make these tasks easier, but we leave that to you.

> **REMEMBER**: When using OpenVINO™ from the command line, you must set up your environment whenever you change users or launch a new terminal.
@sphinxdirective
.. tab:: Linux
.. code-block:: sh
   source /opt/intel/openvino/bin/setupvars.sh
.. tab:: Windows
.. code-block:: bat
   <INSTALL_DIR>\bin\setupvars.bat
.. tab:: macOS
.. code-block:: sh
   source <INSTALL_DIR>/openvino/bin/setupvars.sh
@endsphinxdirective

@sphinxdirective
.. raw:: html
    <div class="collapsible-section">
@endsphinxdirective
    
### Exercise 1: Run A Sample Application
Convert a model using the Model Optimizer, then use a sample application to load the model and run inference. In this section, you will convert an FP32 model suitable for running on a CPU.

**Prepare the Software Environment**

1. Set up the environment variables when logging in, changing users, or launching a new terminal. (Details above.)

2. Make a destination directory for the FP32 SqueezeNet* Model:
@sphinxdirective
.. tab:: Linux
.. code-block:: sh
mkdir ~/squeezenet1.1_FP32
cd ~/squeezenet1.1_FP32
.. tab:: Windows
.. code-block:: bat
mkdir C:\Users\<username>\Documents\squeezenet1.1_FP32
cd C:\Users\<username>\Documents\squeezenet1.1_FP32
.. tab:: macOS
.. code-block:: sh
mkdir ~/squeezenet1.1_FP32
cd ~/squeezenet1.1_FP32
@endsphinxdirective

*\*Convert and Optimize a Neural Network Model from Caffe*

Use the Model Optimizer to convert a SqueezeNet* Caffe* model into an optimized Intermediate Representation (IR):
@sphinxdirective
.. tab:: Linux
.. code-block:: sh
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model ~/models/public/squeezenet1.1/squeezenet1.1.caffemodel --data_type FP32 --output_dir .
.. tab:: Windows
.. code-block:: bat
python3 <INSTALL_DIR>\deployment_tools\model_optimizer\mo.py --input_model C:\Users\<username>\Documents\models\public\squeezenet1.1\squeezenet1.1.caffemodel --data_type FP32 --output_dir .
.. tab:: macOS
.. code-block:: sh
python3 <INSTALL_DIR>/openvino/deployment_tools/model_optimizer/mo.py --input_model ~/models/public/squeezenet1.1/squeezenet1.1.caffemodel --data_type FP32 --output_dir .
@endsphinxdirective

**Prepare the Data (Media) or Dataset**

> **NOTE**: In this case, we are using a single image.

1. Copy the labels file to the same location as the IR model.
@sphinxdirective
.. tab:: Linux
.. code-block:: sh
cp /opt/intel/openvino/deployment_tools/demo/squeezenet1.1.labels .
.. tab:: Windows
.. code-block:: bat
copy <INSTALL_DIR>\deployment_tools\demo\squeezenet1.1.labels .
.. tab:: macOS
.. code-block:: sh
cp <INSTALL_DIR>/openvino/deployment_tools/demo/squeezenet1.1.labels .
@endsphinxdirective

   - **Tip:** The labels file contains the classes used by this SqueezeNet* model. If it's in the same directory as the model, the inference results will show text in addition to confidence percentages.
  
2. Copy a sample image to the current directory. You will use this with your optimized model:
@sphinxdirective
.. tab:: Linux
.. code-block:: sh
sudo cp /opt/intel/openvino/deployment_tools/demo/car.png .
.. tab:: Windows
.. code-block:: bat
copy <INSTALL_DIR>\deployment_tools\demo\car.png .
.. tab:: macOS
.. code-block:: sh
cp <INSTALL_DIR>/openvino/deployment_tools/demo/car.png .
@endsphinxdirective

**Run the Sample Application**

Once your setup is complete, you're ready to run a sample application:
@sphinxdirective
.. tab:: Linux
.. code-block:: sh
~/inference_engine_samples_build/intel64/Release/classification_sample_async -i car.png -m ~/squeezenet1.1_FP32/squeezenet1.1.xml -d CPU
.. tab:: Windows
.. code-block:: bat
C:\Users\<username>\Documents\Intel\OpenVINO\inference_engine_samples_build\intel64\Release\classification_sample_async.exe -i car.png -m C:\Users\<username>\Documents\squeezenet1.1_FP32\squeezenet1.1.xml -d CPU
.. tab:: macOS
.. code-block:: sh
~/inference_engine_samples_build/intel64/Release/classification_sample_async -i car.png -m ~/squeezenet1.1_FP32/squeezenet1.1.xml -d CPU
@endsphinxdirective

> **NOTE**: You can usually see an application's help information (parameters, etc.) by using the -h option.
@sphinxdirective
.. tab:: Linux
.. code-block:: sh
   ~/inference_engine_samples_build/intel64/Release/classification_sample_async -h
.. tab:: Windows
.. code-block:: bat
   C:\Users\<username>\Documents\Intel\OpenVINO\inference_engine_samples_build\intel64\Release\classification_sample_async.exe -h
.. tab:: macOS
.. code-block:: sh
   ~/inference_engine_samples_build/intel64/Release/classification_sample_async -h
@endsphinxdirective
    
@sphinxdirective
.. raw:: html
    </div>
@endsphinxdirective
    
@sphinxdirective
.. raw:: html
    <div class="collapsible-section">
@endsphinxdirective
    
### Exercise 2: Human Pose Estimation

This demo detects people and draws stick figures to show limb positions. This model has already been converted for use with the Intel® Distribution of OpenVINO™ toolkit.

- Requires downloading the human-pose-estimation-0001 (ICV) Model
- Requires video or camera input

Example Syntax:

@sphinxdirective
.. tab:: Linux
.. code-block:: sh
   ./human_pose_estimation_demo -i path/to/video -m path/to/model/human-pose-estimation-0001.xml -d CPU
.. tab:: Windows
.. code-block:: bat
   .\human_pose-estimation_demo.exe -i path/to/video -m path\to\model\human-pose-estimation-0001.xml -d CPU
.. tab:: macOS
.. code-block:: sh
   ./human_pose_estimation_demo -i path/to/video -m path/to/model/human-pose-estimation-0001.xml -d CPU
@endsphinxdirective
    
**Steps to Run the Human Pose Demo:**

1. Set up the environment variables:
    
@sphinxdirective
.. tab:: Linux
.. code-block:: sh
   source /opt/intel/openvino/bin/setupvars.sh
.. tab:: Windows
.. code-block:: bat
   <INSTALL_DIR>\openvino\bin\setupvars.bat
.. tab:: macOS
.. code-block:: sh
   source <INSTALL_DIR>/openvino/bin/setupvars.sh
@endsphinxdirective    

2. Move to the Model Downloader Directory:
        
@sphinxdirective
.. tab:: Linux
.. code-block:: sh
   cd /opt/intel/openvino/deployment_tools/tools/model_downloader/
.. tab:: Windows
.. code-block:: bat
   cd <INSTALL_DIR>\deployment_tools\tools\model_downloader\
.. tab:: macOS
.. code-block:: sh
   cd <INSTALL_DIR>/openvino/deployment_tools/tools/model_downloader/
@endsphinxdirective  


3. Find a suitable model:
        
@sphinxdirective
.. tab:: Linux
.. code-block:: sh
   python3 info_dumper.py --print_all | grep pose
.. tab:: Windows
.. code-block:: bat
   python3 info_dumper.py --print_all | grep pose
.. tab:: macOS
.. code-block:: sh
   python3 info_dumper.py --print_all | grep pose
@endsphinxdirective  

**Note:** `info_dumper.py` is a script that can list details about every model available in the Intel® Model Zoo.

4. Download the model:
        
@sphinxdirective
.. tab:: Linux
.. code-block:: sh
   sudo ./downloader.py --name human-pose*
.. tab:: Windows
.. code-block:: bat
   python3 .\downloader.py --name human-pose*
.. tab:: macOS
.. code-block:: sh
   sudo ./downloader.py --name human-pose*
@endsphinxdirective

5. Move the model to a more convenient location:
        
@sphinxdirective
.. tab:: Linux
.. code-block:: sh
   mkdir ~/ir
   cp /opt/intel/openvino/deployment_tools/tools/model_downloader/intel/human-pose-estimation-0001/FP32/human-pose-estimation-0001* ~/ir/
.. tab:: Windows
.. code-block:: bat
   mkdir C:\Users\<username>\Documents\ir
   copy <INSTALL_DIR>\deployment_tools\tools\model_downloader\intel\human-pose-estimation-0001\FP32\human-pose-estimation-0001* C:\Users\<username>\Documents\ir
.. tab:: macOS
.. code-block:: sh
   mkdir ~/ir
   cp <INSTALL_DIR>/openvino/deployment_tools/tools/model_downloader/intel/human-pose-estimation-0001/FP32/human-pose-estimation-0001* ~/ir/
@endsphinxdirective

6. Download an appropriate video.
   Browse to the following URL and download the video:
   https://www.pexels.com/video/couple-dancing-on-sunset-background-2035509/
        	   
@sphinxdirective
.. tab:: Linux
   Rename the video for convenience:
.. code-block:: sh
   mv ~/Downloads/Pexels\ Videos\ 2035509.mp4 ~/Videos/hum-pose.mp4
.. tab:: Windows
   Rename the video for convenience:
.. code-block:: bat
   ren "C:\Users\<username>\Downloads\Pexels Videos 2035509.mp4" C:\Users\<username>\Videos\hum-pose.mp4
.. tab:: macOS
   Rename the video for convenience:
.. code-block:: sh
   mv ~/Downloads/Pexels\ Videos\ 2035509.mp4 ~/Videos/hum-pose.mp4
@endsphinxdirective

7. Run the sample:
        
@sphinxdirective
.. tab:: Linux
.. code-block:: sh
   cd ~/omz_demos_build/intel64/Release/
   ./human_pose_estimation_demo -i ~/Videos/hum-pose.mp4 -m ~/ir/human-pose-estimation-0001.xml -d CPU
.. tab:: Windows
.. code-block:: bat
   cd C:\Users\<username>\Documents\Intel\OpenVINO\inference_engine_demos_build\intel64\Release
   human_pose_estimation_demo.exe -i C:\Users\<username>\Videos\hum-pose.mp4 -m C:\Users\<username>\Documents\ir\human-pose-estimation-0001.xml -d CPU
.. tab:: macOS
.. code-block:: sh
   cd ~/omz_demos_build/intel64/Release/
   ./human_pose_estimation_demo -i ~/Videos/hum-pose.mp4 -m ~/ir/human-pose-estimation-0001.xml -d CPU
@endsphinxdirective

@sphinxdirective
.. raw:: html
    </div>
@endsphinxdirective

@sphinxdirective
.. raw:: html
    <div class="collapsible-section">
@endsphinxdirective

### Exercise 3: Interactive Face Detection
	
The Face Detection demo draws bounding boxes around faces, and optionally feeds the output of the primary model to additional models. The model has already been converted for use with OpenVINO™. 

This demo supports face detection, plus optional functions:

- Age-gender recognition
- Emotion recognition
- Head pose
- Facial landmark display

Example Syntax:
- interactive_face_detection_demo -i path/to/video -m path/to/face/model -d CPU

Steps:

1. Find and download an appropriate face detection model.  There are several available in the Intel® Model Zoo.
    - You can access the [Pretrained Models](https://software.intel.com/en-us/openvino-toolkit/documentation/pretrained-models) page from the OpenVINO™ documentation to review model options.
    - You may need to try out different models to find one that works, or that works best for your scenario.
2. Find and download a video that features faces.
3. Run the demo with just the face detection model.
4. **OPTIONAL:** Run the demo using additional models (age-gender, emotion recognition, head pose, etc.).
    Note that when you use multiple models, there is always a primary model that is used followed by a number of optional models that use the output from the initial model.

@sphinxdirective
.. raw:: html
    </div>
@endsphinxdirective

@sphinxdirective
.. raw:: html
    <div class="collapsible-section">
@endsphinxdirective
	
### Exercise 4: DL Streamer
	
The DL Streamer is a command-line tool and API for integrating OpenVINO into a media analytics pipeline.  It supports OpenVINO, GStreamer, Mosquitto, Kafka, and a variety of other technologies.

Follow the link below, read through the documentation, then do the tutorial.
	
[DL Streamer Documentation and Tutorial](DL_Streamer/README.md)

@sphinxdirective
.. raw:: html
    </div>
@endsphinxdirective

## Other Demos/Samples

For more samples and demos, you can visit the samples and demos pages below. You can review samples and demos by complexity or by usage.

[Samples](../IE_DG/Samples_Overview.md)

[Demos](@ref omz_demos)