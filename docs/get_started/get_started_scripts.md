# Getting Started with Demo Scripts {#openvino_docs_get_started_get_started_scripts}

## Introduction

A set of demo scripts in the `openvino_2021/deployment_tools/demo` directory give you a starting point for learning the OpenVINO™ workflow. These scripts automatically perform the workflow steps to demonstrate running inference pipelines for different scenarios. The demo steps let you see how to: 
* Compile several samples from the source files delivered as part of the OpenVINO™ toolkit.
* Download trained models.
* Convert the models to IR (Intermediate Representation format used by OpenVINO™) with Model Optimizer.
* Perform pipeline steps and see the output on the console.

This guide assumes you completed all installation and configuration steps. If you have not yet installed and configured the toolkit:

@sphinxdirective
.. tab:: Linux
 
   See :doc:`Install Intel® Distribution of OpenVINO™ toolkit for Linux* <openvino_docs_install_guides_installing_openvino_linux>`
 
.. tab:: Windows
 
   See :doc:`Install Intel® Distribution of OpenVINO™ toolkit for Windows* <openvino_docs_install_guides_installing_openvino_windows>`
 
.. tab:: macOS
 
   See :doc:`Install Intel® Distribution of OpenVINO™ toolkit for macOS* <openvino_docs_install_guides_installing_openvino_macos>`
  
@endsphinxdirective

The demo scripts can run inference on any [supported target device](https://software.intel.com/en-us/openvino-toolkit/hardware). Although the default inference device (i.e., processor) is the CPU, you can add the `-d` parameter to specify a different inference device. The general command to run a demo script is as follows:

@sphinxdirective
.. tab:: Linux

   .. code-block:: sh

      #If you installed in a location other than /opt/intel, substitute that path.
      cd /opt/intel/openvino_2021/deployment_tools/demo/
      ./<script_name> -d [CPU, GPU, MYRIAD, HDDL]

.. tab:: Windows

   .. code-block:: sh

      rem If you installed in a location other than the default, substitute that path.
      cd "C:\Program Files (x86)\Intel\openvino_2021\deployment_tools\demo"
      .\<script_name> -d [CPU, GPU, MYRIAD, HDDL]

.. tab:: macOS

   .. code-block:: sh

      #If you installed in a location other than /opt/intel, substitute that path.
      cd /opt/intel/openvino_2021/deployment_tools/demo/
      ./<script_name> -d [CPU, MYRIAD]

@endsphinxdirective

Before running the demo applications on Intel® Processor Graphics or on an Intel® Neural Compute Stick 2 device, you must complete additional configuration steps. 

@sphinxdirective
.. tab:: Linux

   For details, see the following sections in the :doc:`installation instructions <openvino_docs_install_guides_installing_openvino_linux>`:
   
   * Steps for Intel® Processor Graphics (GPU) 
   * Steps for Intel® Neural Compute Stick 2

.. tab:: Windows

   For details, see the following sections in the :doc:`installation instructions <openvino_docs_install_guides_installing_openvino_windows>`:
   
   * Additional Installation Steps for Intel® Processor Graphics (GPU)
   * Additional Installation Steps for Intel® Vision Accelerator Design with Intel® Movidius™ VPUs

.. tab:: macOS

   For details, see the following sections in the :doc:`installation instructions <openvino_docs_install_guides_installing_openvino_macos>`:
   
   * Steps for Intel® Neural Compute Stick 2

@endsphinxdirective

The following sections describe each demo script.

## Image Classification Demo Script
The `demo_squeezenet_download_convert_run` script illustrates the image classification pipeline.

The script: 
1. Downloads a SqueezeNet model. 
2. Runs the Model Optimizer to convert the model to the IR format used by OpenVINO™.
3. Builds the Image Classification Sample Async application.
4. Runs the compiled sample with the `car.png` image located in the `demo` directory.

### Example of Running the Image Classification Demo Script

@sphinxdirective
.. raw:: html
    <div class="collapsible-section">

@endsphinxdirective
**Click for an example of running the Image Classification demo script**

To preview the image that the script will classify:

@sphinxdirective
.. tab:: Linux

   .. code-block:: sh

      cd /opt/intel/openvino_2021/deployment_tools/demo
      eog car.png

.. tab:: Windows

   .. code-block:: sh

      car.png

.. tab:: macOS

   .. code-block:: sh

      cd /opt/intel/openvino_2021/deployment_tools/demo
      open car.png

@endsphinxdirective

To run the script and perform inference on the CPU:

@sphinxdirective
.. tab:: Linux

   .. code-block:: sh

      ./demo_squeezenet_download_convert_run.sh

.. tab:: Windows

      .. code-block:: bat

         .\demo_squeezenet_download_convert_run.bat

.. tab:: macOS

   .. code-block:: sh

      ./demo_squeezenet_download_convert_run.sh

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
      436     0.0068161   beach wagon, station wagon, wagon, estate car, beach wagon, station wagon, wagon
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

      Image /Users/colin/intel/openvino_2021/deployment_tools/demo/car.png

      classid probability label
      ------- ----------- -----
      817     0.8363345   sports car, sport car
      511     0.0946488   convertible
      479     0.0419131   car wheel
      751     0.0091071   racer, race car, racing car
      436     0.0068161   beach wagon, station wagon, wagon, estate car, beach wagon, station wagon, wagon
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

## Inference Pipeline Demo Script
The `demo_security_barrier_camera` application uses vehicle recognition in which vehicle attributes build on each other to narrow in on a specific attribute.

The script:
1. Downloads three pre-trained models, already converted to IR format.
2. Builds the Security Barrier Camera Demo application.
3. Runs the application with the three models and the `car_1.bmp` image from the `demo` directory to show an inference pipeline.

This application:

1. Gets the boundaries an object identified as a vehicle with the first model.
2. Uses the vehicle identification as input to the second model, which identifies specific vehicle attributes, including the license plate.
3. Uses the license plate as input to the third model, which recognizes specific characters in the license plate.

### Example of Running the Pipeline Demo Script
@sphinxdirective
.. raw:: html

   <div class="collapsible-section">

@endsphinxdirective
**Click for an example of Running the Pipeline demo script**
    
To run the script performing inference on Intel® Processor Graphics:
@sphinxdirective
.. tab:: Linux

   .. code-block:: sh

      ./demo_security_barrier_camera.sh -d GPU

.. tab:: Windows

   .. code-block:: bat

      .\demo_security_barrier_camera.bat -d GPU

@endsphinxdirective

When the verification script is complete, you see an image that displays the resulting frame with detections rendered as bounding boxes and overlaid text:

@sphinxdirective
.. tab:: Linux

   .. image:: ../img/inference_pipeline_script_lnx.png

.. tab:: Windows

   .. image:: ../img/inference_pipeline_script_win.png

.. tab:: macOS

   .. image:: ../img/inference_pipeline_script_mac.png

@endsphinxdirective

@sphinxdirective
.. raw:: html

   </div>

@endsphinxdirective

## Benchmark Demo Script
The `demo_benchmark_app` script illustrates how to use the Benchmark Application to estimate deep learning inference performance on supported devices. 

The script:
1. Downloads a SqueezeNet model.
2. Runs the Model Optimizer to convert the model to IR format.
3. Builds the Inference Engine Benchmark tool.
4. Runs the tool with the `car.png` image located in the `demo` directory.

### Example of Running the Benchmark Demo Script

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

## Other Get Started Documents

For more get started documents, visit the pages below:

[Get Started with Sample and Demo Applications](get_started_demos.md)

[Get Started with Instructions](get_started_instructions.md)
