# GET STARTED {#get_started}

@sphinxdirective

.. meta::
   :description: Learn how to install Intel® Distribution of OpenVINO™ toolkit 
                 on Windows, macOS, and Linux operating systems, using various 
                 installation methods.

.. toctree::
   :maxdepth: 1
   :hidden:

   Install OpenVINO <openvino_docs_install_guides_overview>
   Additional Hardware Setup <openvino_docs_install_guides_configurations_header>
   Troubleshooting <openvino_docs_get_started_guide_troubleshooting>
   System Requirements <system_requirements>


.. raw:: html

   <link rel="stylesheet" type="text/css" href="_static/css/getstarted_style.css">

   <p id="GSG_introtext">Welcome to OpenVINO! This guide introduces installation and learning materials for Intel® Distribution of OpenVINO™ toolkit. The guide walks through the following steps:<br />
     <a href="https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F201-vision-monodepth%2F201-vision-monodepth.ipynb" >Quick Start Example</a>
     <a href="openvino_docs_install_guides_overview.html" >Install OpenVINO</a>
     <a href="#learn-openvino" >Learn OpenVINO</a>
   </p>
   <div style="clear:both;"> </div> 

.. _quick-start-example:

1. Quick Start Example (No Installation Required)
#################################################

.. image:: https://user-images.githubusercontent.com/15709723/127752390-f6aa371f-31b5-4846-84b9-18dd4f662406.gif
   :width: 400

Try out OpenVINO's capabilities with this quick start example that estimates depth in a scene using an OpenVINO monodepth model. `Run the example in a Jupyter Notebook inside your web browser <https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F201-vision-monodepth%2F201-vision-monodepth.ipynb>`__ to quickly see how to load a model, prepare an image, inference the image, and display the result.

.. _install-openvino-gsg:

2. Install OpenVINO
###################
   
See the :doc:`installation overview page <openvino_docs_install_guides_overview>` for options to install OpenVINO and set up a development environment on your device.
   
.. _get-started-tutorials:

3. Learn OpenVINO
#################
   
OpenVINO provides a wide array of examples and documentation showing how to work with models, run inference, and deploy applications. Step through the sections below to learn the basics of OpenVINO and explore its advanced optimization features. For further details, visit :doc:`OpenVINO documentation <documentation>`.
   
OpenVINO users of all experience levels can try `Intel® DevCloud <https://www.intel.com/content/www/us/en/developer/tools/devcloud/edge/overview.html>`__ , a free web-based service for developing, testing, and running OpenVINO applications on an online cluster of the latest Intel® hardware.

.. _openvino-basics:

OpenVINO Basics
+++++++++++++++

Learn the basics of working with models and inference in OpenVINO. Begin with “Hello World” Interactive Tutorials that show how to prepare models, run inference, and retrieve results using the OpenVINO API. Then, explore other examples from the Open Model Zoo and OpenVINO Code Samples that can be adapted for your own application.
   
.. _interactive-tutorials:

Interactive Tutorials - Jupyter Notebooks
-----------------------------------------

Start with :doc:`interactive Python tutorials <tutorials>` that show the basics of model inferencing, the OpenVINO API, how to convert models to OpenVINO format, and more.

* `Hello Image Classification <notebooks/001-hello-world-with-output.html>`__ - Load an image classification model in OpenVINO and use it to apply a label to an image
* `OpenVINO Runtime API Tutorial <notebooks/002-openvino-api-with-output.html>`__ - Learn the basic Python API for working with models in OpenVINO
* `Convert TensorFlow Models to OpenVINO <notebooks/101-tensorflow-classification-to-openvino-with-output.html>`__
* `Convert PyTorch Models to OpenVINO <notebooks/102-pytorch-onnx-to-openvino-with-output.html>`__

.. _code-samples:

OpenVINO Code Samples
---------------------

View :doc:`sample code <openvino_docs_OV_UG_Samples_Overview>` for various C++ and Python applications that can be used as a starting point for your own application. For C++ developers, step through the :doc:`Get Started with C++ Samples <openvino_docs_get_started_get_started_demos>` to learn how to build and run an image classification program that uses OpenVINO’s C++ API.
      
.. _integrate-openvino:

Integrate OpenVINO With Your Application
----------------------------------------

Learn how to :doc:`use the OpenVINO API to implement an inference pipeline <openvino_docs_OV_UG_Integrate_OV_with_your_application>` in your application.

.. _openvino-advanced-features:

OpenVINO Advanced Features
++++++++++++++++++++++++++

OpenVINO provides features to improve your model’s performance, optimize your runtime, maximize your application’s throughput on target hardware, and much more. Visit the links below to learn more about these features and how to use them.

Model Compression and Quantization
----------------------------------

Use OpenVINO’s model compression tools to reduce your model’s latency and memory footprint while maintaining good accuracy.

* Tutorial - `OpenVINO Post-Training Model Quantization <notebooks/111-yolov5-quantization-migration-with-output.html>`__
* Tutorial - `Quantization-Aware Training in TensorFlow with OpenVINO NNCF <notebooks/305-tensorflow-quantization-aware-training-with-output.html>`__
* Tutorial - `Quantization-Aware Training in PyTorch with NNCF <notebooks/302-pytorch-quantization-aware-training-with-output.html>`__
* :doc:`Model Optimization Guide <openvino_docs_model_optimization_guide>`

Automated Device Configuration
------------------------------

OpenVINO’s hardware device configuration options enable you to write an application once and deploy it anywhere with optimal performance.

* Increase application portability with :doc:`Automatic Device Selection (AUTO) <openvino_docs_OV_UG_supported_plugins_AUTO>`
* Perform parallel inference across processors with :doc:`Multi-Device Execution (MULTI) <openvino_docs_OV_UG_Running_on_multiple_devices>`
* Efficiently split inference between hardware cores with :doc:`Heterogeneous Execution (HETERO) <openvino_docs_OV_UG_Hetero_execution>`

Flexible Model and Pipeline Configuration
-----------------------------------------

Pipeline and model configuration features in OpenVINO Runtime allow you to easily optimize your application’s performance on any target hardware.

* :doc:`Automatic Batching <openvino_docs_OV_UG_Automatic_Batching>` performs on-the-fly grouping of inference requests to maximize utilization of the target hardware’s memory and processing cores.
* :doc:`Performance Hints <openvino_docs_OV_UG_Performance_Hints>` automatically adjust runtime parameters to prioritize for low latency or high throughput
* :doc:`Dynamic Shapes <openvino_docs_OV_UG_DynamicShapes>` reshapes models to accept arbitrarily-sized inputs, increasing flexibility for applications that encounter different data shapes
* :doc:`Benchmark Tool <openvino_inference_engine_tools_benchmark_tool_README>` characterizes model performance in various hardware and pipeline configurations
   
.. _additional-resources:

Additional Resources
====================

* `OpenVINO Success Stories <https://www.intel.com/content/www/us/en/internet-of-things/ai-in-production/success-stories.html>`__ - See how Intel partners have successfully used OpenVINO in production applications to solve real-world problems.
* :doc:`OpenVINO Supported Models <openvino_supported_models>` - Check which models OpenVINO supports on your hardware.
* :doc:`Performance Benchmarks <openvino_docs_performance_benchmarks>` - View results from benchmarking models with OpenVINO on Intel hardware.

@endsphinxdirective
