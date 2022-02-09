# Get Started with Sample and Demo Applications {#openvino_docs_get_started_get_started_demos}

## Introduction

This section guides you through a simplified workflow for the Intel® Distribution of OpenVINO™ toolkit using code samples and demo applications.
You will perform the following steps:

1. <a href="#download-models">Use the Model Downloader to download suitable models.</a>
2. <a href="#convert-models-to-intermediate-representation">Convert the models with the Model Optimizer.</a> 
3. <a href="download-media">Download media files to run inference on.</a>
4. <a href="run-image-classification">Run inference on the sample and see the results:</a>
    - <a href="run-image-classification">Image Classification Code Sample</a>
    - <a href="run-security-barrier">Security Barrier Camera Demo application</a>

If you installed OpenVINO™ via `pip` you need to change commands listed below. Details are listed in one of [tutorials](https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/002-openvino-api/002-openvino-api.ipynb).

This guide assumes you completed all installation and configuration steps. If you have not yet installed and configured the toolkit:

@sphinxdirective
.. tab:: Linux

   See :doc:`Install Intel® Distribution of OpenVINO™ toolkit for Linux* <openvino_docs_install_guides_installing_openvino_linux>`

.. tab:: Windows

   See :doc:`Install Intel® Distribution of OpenVINO™ toolkit for Windows* <openvino_docs_install_guides_installing_openvino_windows>`

.. tab:: macOS

   See :doc:`Install Intel® Distribution of OpenVINO™ toolkit for macOS* <openvino_docs_install_guides_installing_openvino_macos>`
  
@endsphinxdirective

## Build Samples and Demos

If you have already built the demos and samples, you can skip this section. The build will take about 5-10 minutes, depending on your system.

To build OpenVINO samples:
@sphinxdirective
.. tab:: Linux

   Go to the :doc:`Inference Engine Samples page <openvino_docs_IE_DG_Samples_Overview>` and see the "Build the Sample Applications on Linux*" section.

.. tab:: Windows

   Go to the :doc:`Inference Engine Samples page <openvino_docs_IE_DG_Samples_Overview>` and see the "Build the Sample Applications on Microsoft Windows* OS" section.

.. tab:: macOS

   Go to the :doc:`Inference Engine Samples page <openvino_docs_IE_DG_Samples_Overview>` and see the "Build the Sample Applications on macOS*" section. 

@endsphinxdirective

To build OpenVINO demos:
@sphinxdirective
.. tab:: Linux

   Go to the :doc:`Open Model Zoo Demos page <omz_demos>` and see the "Build the Demo Applications on Linux*" section.

.. tab:: Windows

   Go to the :doc:`Open Model Zoo Demos page <omz_demos>` and see the "Build the Demo Applications on Microsoft Windows* OS" section.

.. tab:: macOS

   Go to the :doc:`Open Model Zoo Demos page <omz_demos>` and see the "Build the Demo Applications on Linux*" section. You can use the requirements from "To build OpenVINO samples" above and adapt the Linux build steps for macOS*.

@endsphinxdirective

## <a name="download-models"></a> Step 1: Download the Models

You must have a model that is specific for your inference task. Example model types are:

- Classification (AlexNet, GoogleNet, SqueezeNet, others): Detects one type of element in an image
- Object Detection (SSD, YOLO): Draws bounding boxes around multiple types of objects in an image
- Custom: Often based on SSD

Options to find a model suitable for the OpenVINO™ toolkit:

- Download public or Intel pre-trained models from the [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo) using the [Model Downloader tool](@ref omz_tools_downloader)
- Download from GitHub*, Caffe* Zoo, TensorFlow* Zoo, etc.
- Train your own model with machine learning tools
  
This guide uses the OpenVINO™ Model Downloader to get pre-trained models. You can use one of the following commands to find a model:

### List the models available in the downloader

@sphinxdirective
.. tab:: Linux

   .. code-block:: sh

      cd /opt/intel/openvino_2021/deployment_tools/tools/model_downloader/
      python3 info_dumper.py --print_all

.. tab:: Windows

   .. code-block:: bat

      cd <INSTALL_DIR>\deployment_tools\tools\model_downloader\
      python info_dumper.py --print_all

.. tab:: macOS

   .. code-block:: sh

      cd /opt/intel/openvino_2021/deployment_tools/tools/model_downloader/
      python3 info_dumper.py --print_all

@endsphinxdirective

### Use `grep` to list models that have a specific name pattern

``` sh
   python3 info_dumper.py --print_all | grep <model_name>
```

Use the Model Downloader to download the models to a models directory. This guide uses `<models_dir>` and `<models_name>` as placeholders for the models directory and model name:
@sphinxdirective
.. tab:: Linux

   Don't run downloader with `sudo`. It will further lead to complications
   .. code-block:: sh

      python3 downloader.py --name <model_name> --output_dir <models_dir>

.. tab:: Windows

   .. code-block:: bat

      python downloader.py --name <model_name> --output_dir <models_dir>

.. tab:: macOS

   Don't run downloader with `sudo`. It will further lead to complications
   .. code-block:: sh

      python3 downloader.py --name <model_name> --output_dir <models_dir>

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

   <div class="collapsible-section" data-title="Click for an example of downloading the SqueezeNet Caffe* model">

@endsphinxdirective

To download the SqueezeNet 1.1 Caffe* model to the `models` folder:

@sphinxdirective
.. tab:: Linux

   .. code-block:: sh

      python3 downloader.py --name squeezenet1.1 --output_dir ~/models

.. tab:: Windows

   .. code-block:: bat

      python downloader.py --name squeezenet1.1 --output_dir C:\Users\<USER_ID>\Documents\models

.. tab:: macOS

   .. code-block:: sh

      python3 downloader.py --name squeezenet1.1 --output_dir ~/models

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

.. tab:: macOS

   .. code-block:: sh

      ###############|| Downloading models ||###############

      ========= Downloading /Users/username/models/public/squeezenet1.1/squeezenet1.1.prototxt
      ... 100%, 9 KB, 44058 KB/s, 0 seconds passed

      ========= Downloading /Users/username/models/public/squeezenet1.1/squeezenet1.1.caffemodel
      ... 100%, 4834 KB, 4877 KB/s, 0 seconds passed

      ###############|| Post processing ||###############

      ========= Replacing text in /Users/username/models/public/squeezenet1.1/squeezenet1.1.prototxt =========

@endsphinxdirective

@sphinxdirective
.. raw:: html

   </div>

@endsphinxdirective

@sphinxdirective
.. raw:: html

   <div class="collapsible-section" data-title="Click for an example of downloading models for the Security Barrier Camera Demo application">

@endsphinxdirective

To download all three pre-trained models in FP16 precision to the `models` folder in your home folder:

@sphinxdirective
.. tab:: Linux

   .. code-block:: sh

      python3 downloader.py --name vehicle-license-plate-detection-barrier-0106,vehicle-attributes-recognition-barrier-0039,license-plate-recognition-barrier-0001 --output_dir ~/models --precisions FP16

.. tab:: Windows

   .. code-block:: bat

      python downloader.py --name vehicle-license-plate-detection-barrier-0106,vehicle-attributes-recognition-barrier-0039,license-plate-recognition-barrier-0001 --output_dir C:\Users\<USER_ID>\Documents\models --precisions FP16

.. tab:: macOS

   .. code-block:: sh

      python3 downloader.py --name vehicle-license-plate-detection-barrier-0106,vehicle-attributes-recognition-barrier-0039,license-plate-recognition-barrier-0001 --output_dir ~/models --precisions FP16

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

      ################|| Downloading models ||################

      ========== Downloading /Users/username/models/intel/vehicle-license-plate-detection-barrier-0106/FP16/vehicle-license-plate-detection-barrier-0106.xml
      ... 100%, 207 KB, 313926 KB/s, 0 seconds passed

      ========== Downloading /Users/username/models/intel/vehicle-license-plate-detection-barrier-0106/FP16/vehicle-license-plate-detection-barrier-0106.bin
      ... 100%, 1256 KB, 2552 KB/s, 0 seconds passed

      ========== Downloading /Users/username/models/intel/vehicle-attributes-recognition-barrier-0039/FP16/vehicle-attributes-recognition-barrier-0039.xml
      ... 100%, 32 KB, 172042 KB/s, 0 seconds passed

      ========== Downloading /Users/username/models/intel/vehicle-attributes-recognition-barrier-0039/FP16/vehicle-attributes-recognition-barrier-0039.bin
      ... 100%, 1222 KB, 2712 KB/s, 0 seconds passed

      ========== Downloading /Users/username/models/intel/license-plate-recognition-barrier-0001/FP16/license-plate-recognition-barrier-0001.xml
      ... 100%, 47 KB, 217130 KB/s, 0 seconds passed

      ========== Downloading /Users/username/models/intel/license-plate-recognition-barrier-0001/FP16/license-plate-recognition-barrier-0001.bin
      ... 100%, 2378 KB, 4222 KB/s, 0 seconds passed

      ################|| Post-processing ||################

@endsphinxdirective

@sphinxdirective
.. raw:: html

   </div>

@endsphinxdirective

## <a name="convert-models-to-intermediate-representation"></a>Step 2: Convert the Model with Model Optimizer

In this step, your trained models are ready to run through the Model Optimizer to convert them to the IR (Intermediate Representation) format. For most model types, this is required before using the Inference Engine with the model.

Models in the IR format always include an `.xml` and `.bin` file and may also include other files such as `.json` or `.mapping`. Make sure you have these files together in a single directory so the Inference Engine can find them.

REQUIRED: `model_name.xml`
REQUIRED: `model_name.bin`
OPTIONAL: `model_name.json`, `model_name.mapping`, etc.

This tutorial uses the public SqueezeNet 1.1 Caffe* model to run the Image Classification Sample. See the example in the Download Models section of this page to learn how to download this model.

The SqueezeNet1.1 model is downloaded in the Caffe* format. You must use the Model Optimizer to convert the model to IR. The `vehicle-license-plate-detection-barrier-0106`, `vehicle-attributes-recognition-barrier-0039`, and `license-plate-recognition-barrier-0001` models are downloaded in IR format. You don't need to use the Model Optimizer on them because they are Intel models that have previously been converted. Public models will need converting with Model Optimizer.

Create an `<ir_dir>` directory to contain the model's Intermediate Representation (IR).

@sphinxdirective
.. tab:: Linux

   .. code-block:: sh

      mkdir ~/ir

.. tab:: Windows

   .. code-block:: bat

      mkdir C:\Users\<USER_ID>\Documents\ir

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
      python3 mo.py --input_model <model_dir>/<model_file> --data_type <model_precision> --output_dir <ir_dir>

.. tab:: Windows

   .. code-block:: bat

      cd <INSTALL_DIR>\deployment_tools\model_optimizer
      python mo.py --input_model <model_dir>\<model_file> --data_type <model_precision> --output_dir <ir_dir>

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
      python3 mo.py --input_model ~/models/public/squeezenet1.1/squeezenet1.1.caffemodel --data_type FP16 --output_dir ~/ir

.. tab:: Windows

   .. code-block:: bat

      cd <INSTALL_DIR>\deployment_tools\model_optimizer
      python mo.py --input_model C:\Users\<USER_ID>\Documents\models\public\squeezenet1.1\squeezenet1.1.caffemodel --data_type FP16 --output_dir C:\Users\<USER_ID>\Documents\ir

.. tab:: macOS

   .. code-block:: sh

      cd /opt/intel/openvino/deployment_tools/model_optimizer
      python3 mo.py --input_model ~/models/public/squeezenet1.1/squeezenet1.1.caffemodel --data_type FP16 --output_dir ~/ir

@endsphinxdirective

## <a name="download-media"></a> Step 3: Download a Video or Still Photo as Media

Many sources are available from which you can download video media to use the code samples and demo applications. Possibilities include:

- [Pexels](https://pexels.com)
- [Google Images](https://images.google.com)

As an alternative, the Intel® Distribution of OpenVINO™ toolkit includes several sample images and videos that you can use for running code samples and demo applications:
@sphinxdirective
.. tab:: Linux

   - ``/opt/intel/openvino_2021/deployment_tools/demo/car.png``
   - ``/opt/intel/openvino_2021/deployment_tools/demo/car_1.bmp``
   - `Sample images and video <https://storage.openvinotoolkit.org/data/test_data/>`_
   - `Sample videos <https://github.com/intel-iot-devkit/sample-videos>`_

.. tab:: Windows

   - ``<INSTALL_DIR>\deployment_tools\demo\car.png``
   - ``<INSTALL_DIR>\deployment_tools\demo\car_1.bmp``
   - `Sample images and video <https://storage.openvinotoolkit.org/data/test_data/>`_
   - `Sample videos <https://github.com/intel-iot-devkit/sample-videos>`_

.. tab:: macOS

   - ``/opt/intel/openvino_2021/deployment_tools/demo/car.png``
   - ``/opt/intel/openvino_2021/deployment_tools/demo/car_1.bmp``
   - `Sample images and video <https://storage.openvinotoolkit.org/data/test_data/>`_
   - `Sample videos <https://github.com/intel-iot-devkit/sample-videos>`_

@endsphinxdirective

## <a name="run-image-classification"></a>Step 4: Run Inference on the Sample

### Run the Image Classification Code Sample

To run the **Image Classification** code sample with an input image using the IR model:

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

      source /opt/intel/openvino/bin/setupvars.sh

@endsphinxdirective

2. Go to the code samples release directory created when you built the samples earlier:
@sphinxdirective
.. tab:: Linux

   .. code-block:: sh

      cd ~/inference_engine_cpp_samples_build/intel64/Release

.. tab:: Windows

   .. code-block:: bat

      cd C:\Users\<USER_ID>\Documents\Intel\OpenVINO\inference_engine_samples_build\intel64\Release

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

   <div class="collapsible-section" data-title="Click for examples of running the Image Classification code sample on different devices">

@endsphinxdirective

The following commands run the Image Classification Code Sample using the `car.png` file from the `demo` directory as an input image, the model in IR format from the `ir` directory, and on different hardware devices:

   **CPU:**  
@sphinxdirective
.. tab:: Linux

   .. code-block:: sh

      ./classification_sample_async -i /opt/intel/openvino/deployment_tools/demo/car.png -m ~/models/public/squeezenet1.1/ir/squeezenet1.1.xml -d CPU

.. tab:: Windows

   .. code-block:: bat

      .\classification_sample_async.exe -i <INSTALL_DIR>\deployment_tools\demo\car.png -m C:\Users\<USER_ID>\Documents\models\public\squeezenet1.1\ir\squeezenet1.1.xml -d CPU

.. tab:: macOS

   .. code-block:: sh

      ./classification_sample_async -i /opt/intel/openvino/deployment_tools/demo/car.png -m ~/models/public/squeezenet1.1/ir/squeezenet1.1.xml -d CPU

@endsphinxdirective

   **GPU:**
   > **NOTE**: Running inference on Intel® Processor Graphics (GPU) requires 
    [additional hardware configuration steps](https://docs.openvino.ai/latest/_docs_install_guides_installing_openvino_linux.html#additional-GPU-steps), as described earlier on this page. Running on GPU is not compatible with macOS*.

@sphinxdirective
.. tab:: Linux

   .. code-block:: sh

      ./classification_sample -i /opt/intel/openvino/deployment_tools/demo/car.png -m ~/models/public/squeezenet1.1/ir/squeezenet1.1.xml -d GPU

.. tab:: Windows

   .. code-block:: bat

      .\classification_sample_async.exe -i <INSTALL_DIR>\deployment_tools\demo\car.png -m C:\Users\<USER_ID>\Documents\models\public\squeezenet1.1\ir\squeezenet1.1.xml -d GPU

@endsphinxdirective

   **MYRIAD:**
   > **NOTE**: Running inference on VPU devices (Intel® Movidius™ Neural Compute
   Stick or Intel® Neural Compute Stick 2) with the MYRIAD plugin requires 
    [additional hardware configuration steps](inference-engine/README.md#optional-additional-installation-steps-for-the-intel-movidius-neural-compute-stick-and-neural-compute-stick-2), as described earlier on this page.

@sphinxdirective
.. tab:: Linux

   .. code-block:: sh

      ./classification_sample -i /opt/intel/openvino/deployment_tools/demo/car.png -m ~/models/public/squeezenet1.1/ir/squeezenet1.1.xml -d MYRIAD

.. tab:: Windows

   .. code-block:: bat

      .\classification_sample_async.exe -i <INSTALL_DIR>\deployment_tools\demo\car.png -m C:\Users\<USER_ID>\Documents\models\public\squeezenet1.1\ir\squeezenet1.1.xml -d MYRIAD

.. tab:: macOS

   .. code-block:: sh

      ./classification_sample -i /opt/intel/openvino/deployment_tools/demo/car.png -m ~/models/public/squeezenet1.1/ir/squeezenet1.1.xml -d MYRIAD

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

      [ INFO ] Execution successful
      
      [ INFO ] This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool

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

      [ INFO ] Execution successful
      
      [ INFO ] This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool

.. tab:: macOS

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

      [ INFO ] Execution successful
      
      [ INFO ] This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool

@endsphinxdirective

@sphinxdirective
.. raw:: html

   </div>

@endsphinxdirective

### <a name="run-security-barrier"></a>Run the Security Barrier Camera Demo Application

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

      source /opt/intel/openvino/bin/setupvars.sh

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

      cd ~/inference_engine_demos_build/intel64/Release

@endsphinxdirective

3. Run the demo executable, specifying the input media file, list of model IRs, and a target device for performing inference:
@sphinxdirective
.. tab:: Linux

   .. code-block:: sh

      ./security_barrier_camera_demo -i <path_to_media> -m <path_to_model>/vehicle-license-plate-detection-barrier-0106.xml -m_va <path_to_vehicle_attributes model>/vehicle-attributes-recognition-barrier-0039.xml -m_lpr <path_to_license_plate_recognition_model>/license-plate-recognition-barrier-0001.xml -d <target_device>

.. tab:: Windows

   .. code-block:: bat

      .\security_barrier_camera_demo.exe -i <path_to_media> -m <path_to_vehicle-license-plate-detection_model_xml> -m_va <path_to_vehicle_attributes_model_xml> -m_lpr <path_to_license_plate_recognition_model_xml> -d <target_device>

.. tab:: macOS

   .. code-block:: sh

      ./security_barrier_camera_demo -i <path_to_media> -m <path_to_model>/vehicle-license-plate-detection-barrier-0106.xml -m_va <path_to_vehicle_attributes model>/vehicle-attributes-recognition-barrier-0039.xml -m_lpr <path_to_license_plate_recognition_model>/license-plate-recognition-barrier-0001.xml -d <target_device>

@endsphinxdirective

@sphinxdirective
.. raw:: html

   <div class="collapsible-section" data-title="Click for examples of running the Security Barrier Camera demo application on different devices">

@endsphinxdirective

**CPU:**
@sphinxdirective
.. tab:: Linux

   .. code-block:: sh

      ./security_barrier_camera_demo -i /opt/intel/openvino/deployment_tools/demo/car_1.bmp -m <path_to_model>/vehicle-license-plate-detection-barrier-0106.xml -m_va <path_to_model>/vehicle-attributes-recognition-barrier-0039.xml -m_lpr <path_to_model>/license-plate-recognition-barrier-0001.xml -d CPU

.. tab:: Windows

   .. code-block:: bat

      .\security_barrier_camera_demo.exe -i <INSTALL_DIR>\deployment_tools\demo\car_1.bmp -m C:\Users\username\Documents\models\intel\vehicle-license-plate-detection-barrier-0106\FP16\vehicle-license-plate-detection-barrier-0106.xml -m_va C:\Users\username\Documents\models\intel\vehicle-attributes-recognition-barrier-0039\FP16\vehicle-attributes-recognition-barrier-0039.xml -m_lpr C:\Users\username\Documents\models\intel\license-plate-recognition-barrier-0001\FP16\license-plate-recognition-barrier-0001.xml -d CPU

.. tab:: macOS

   .. code-block:: sh

      ./security_barrier_camera_demo -i /opt/intel/openvino/deployment_tools/demo/car_1.bmp -m <path_to_model>/vehicle-license-plate-detection-barrier-0106.xml -m_va <path_to_model>/vehicle-attributes-recognition-barrier-0039.xml -m_lpr <path_to_model>/license-plate-recognition-barrier-0001.xml -d CPU

@endsphinxdirective

**GPU:**
> **NOTE**: Running inference on Intel® Processor Graphics (GPU) requires [additional hardware configuration steps](https://docs.openvino.ai/latest/_docs_install_guides_installing_openvino_linux.html#additional-GPU-steps), as described earlier on this page. Running on GPU is not compatible with macOS*.

@sphinxdirective
.. tab:: Linux

   .. code-block:: sh

      ./security_barrier_camera_demo -i /opt/intel/openvino/deployment_tools/demo/car_1.bmp -m <path_to_model>/vehicle-license-plate-detection-barrier-0106.xml -m_va <path_to_model>/vehicle-attributes-recognition-barrier-0039.xml -m_lpr <path_to_model>/license-plate-recognition-barrier-0001.xml -d GPU

.. tab:: Windows

   .. code-block:: bat

      .\security_barrier_camera_demo.exe -i <INSTALL_DIR>\deployment_tools\demo\car_1.bmp -m <path_to_model>/vehicle-license-plate-detection-barrier-0106.xml -m_va <path_to_model>/vehicle-attributes-recognition-barrier-0039.xml -m_lpr <path_to_model>/license-plate-recognition-barrier-0001.xml -d GPU

.. tab:: macOS

   .. code-block:: sh

      ./security_barrier_camera_demo -i /opt/intel/openvino/deployment_tools/demo/car_1.bmp -m <path_to_model>/vehicle-license-plate-detection-barrier-0106.xml -m_va <path_to_model>/vehicle-attributes-recognition-barrier-0039.xml -m_lpr <path_to_model>/license-plate-recognition-barrier-0001.xml -d GPU

@endsphinxdirective

**MYRIAD:**
> **NOTE**: Running inference on VPU devices (Intel® Movidius™ Neural Compute Stick or Intel® Neural Compute Stick 2) with the MYRIAD plugin requires     [additional hardware configuration steps](https://docs.openvino.ai/latest/_docs_install_guides_installing_openvino_linux.html#additional-NCS-steps), as described earlier on this page.

@sphinxdirective
.. tab:: Linux

   .. code-block:: sh

      ./security_barrier_camera_demo -i /opt/intel/openvino/deployment_tools/demo/car_1.bmp -m <path_to_model>/vehicle-license-plate-detection-barrier-0106.xml -m_va <path_to_model>/vehicle-attributes-recognition-barrier-0039.xml -m_lpr <path_to_model>/license-plate-recognition-barrier-0001.xml -d MYRIAD

.. tab:: Windows

   .. code-block:: bat

      .\security_barrier_camera_demo.exe -i <INSTALL_DIR>\deployment_tools\demo\car_1.bmp -m <path_to_model>/vehicle-license-plate-detection-barrier-0106.xml -m_va <path_to_model>/vehicle-attributes-recognition-barrier-0039.xml -m_lpr <path_to_model>/license-plate-recognition-barrier-0001.xml -d MYRIAD

@endsphinxdirective

@sphinxdirective
.. raw:: html

   </div>

@endsphinxdirective

## Other Demos/Samples

For more samples and demos, you can visit the samples and demos pages below. You can review samples and demos by complexity or by usage, run the relevant application, and adapt the code for your use.

[Samples](../OV_Runtime_UG/Samples_Overview.md)

[Demos](@ref omz_demos)
