GET STARTED
===========


.. meta::
   :description: Learn how to install Intel® Distribution of OpenVINO™ toolkit
                 on Windows, macOS, and Linux operating systems, using various
                 installation methods.

.. toctree::
   :maxdepth: 1
   :hidden:

   Install OpenVINO <get-started/install-openvino>
   Learn OpenVINO <get-started/learn-openvino>
   System Requirements <./about-openvino/release-notes-openvino/system-requirements>


.. raw:: html

   <link rel="stylesheet" type="text/css" href="_static/css/getstarted_style.css">

   <p id="GSG_introtext">Welcome to OpenVINO! This guide introduces installation and learning materials for Intel® Distribution of OpenVINO™ toolkit. The guide walks through the following steps:<br />
     <a href="https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/vision-monodepth" >Quick Start Example</a>
     <a href="get-started/install-openvino.html" >Install OpenVINO</a>
     <a href="#learn-openvino" >Learn OpenVINO</a>
   </p>
   <div style="clear:both;"> </div>


For a quick reference, check out
`the Quick Start Guide [pdf] <https://docs.openvino.ai/2025/_static/download/OpenVINO_Quick_Start_Guide.pdf>`__


.. _quick-start-example:

1. Quick Start Example (No Installation Required)
#################################################

.. image:: https://user-images.githubusercontent.com/15709723/127752390-f6aa371f-31b5-4846-84b9-18dd4f662406.gif
   :width: 400

Try out OpenVINO's capabilities with this `quick start example <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/vision-monodepth>`__
that estimates depth in a scene using an OpenVINO monodepth model to quickly see how to load a model, prepare an image, inference the image, and display the result.

.. _install-openvino-gsg:

2. Install OpenVINO
###################

See the :doc:`installation overview page <get-started/install-openvino>` for options to install OpenVINO and set up a development environment on your device.

.. _get-started-learn-openvino/interactive-tutorials-python:

3. Learn OpenVINO
#################

OpenVINO provides a wide array of examples and documentation showing how to work with models, run inference, and deploy applications. Step through the sections below to learn the basics of OpenVINO and explore its advanced optimization features. For further details, visit :doc:`OpenVINO documentation <documentation>`.

.. _openvino-basics:

OpenVINO Basics
+++++++++++++++

Learn the basics of working with models and inference in OpenVINO. Begin with “Hello World” Interactive Tutorials that show how to prepare models, run inference, and retrieve results using the OpenVINO API. Then, explore OpenVINO Code Samples that can be adapted for your own application.

.. _interactive-learn-openvino/interactive-tutorials-python:

Interactive Tutorials - Jupyter Notebooks
-----------------------------------------

Start with :doc:`interactive Python <get-started/learn-openvino/interactive-tutorials-python>` that show the basics of model inference, the OpenVINO API, how to convert models to OpenVINO format, and more.

* `Hello Image Classification <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/hello-world>`__
  - Load an image classification model in OpenVINO and use it to apply a label to an image
* `OpenVINO Runtime API Tutorial <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/openvino-api>`__
  - Learn the basic Python API for working with models in OpenVINO
* `Convert TensorFlow Models to OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/tensorflow-classification-to-openvino>`__
* `Convert PyTorch Models to OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/pytorch-to-openvino/pytorch-onnx-to-openvino.ipynb>`__

.. _code-samples:

OpenVINO Code Samples
---------------------

View :doc:`sample code <get-started/learn-openvino/openvino-samples>` for various C++ and Python applications that can be used as a starting point for your own application. For C++ developers, step through the :doc:`Get Started with C++ Samples <get-started/learn-openvino/openvino-samples/get-started-demos>` to learn how to build and run an image classification program that uses OpenVINO’s C++ API.

.. _integrate-openvino:

Integrate OpenVINO With Your Application
----------------------------------------

Learn how to :doc:`use the OpenVINO API to implement an inference pipeline <openvino-workflow/running-inference>` in your application.

.. _openvino-advanced-features:

OpenVINO Advanced Features
++++++++++++++++++++++++++

OpenVINO provides features to improve your model’s performance, optimize your runtime, maximize your application’s throughput on target hardware, and much more. Visit the links below to learn more about these features and how to use them.

Model Compression and Quantization
----------------------------------

Use OpenVINO’s model compression tools to reduce your model’s latency and memory footprint while maintaining good accuracy.

* Tutorial - `Quantization-Aware Training in TensorFlow with OpenVINO NNCF <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/tensorflow-quantization-aware-training>`__
* Tutorial - `Quantization-Aware Training in PyTorch with NNCF <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/pytorch-quantization-aware-training>`__
* :doc:`Model Optimization Guide <openvino-workflow/model-optimization>`

Automated Device Configuration
------------------------------

OpenVINO’s hardware device configuration options enable you to write an application once and deploy it anywhere with optimal performance.

* Increase application portability and perform parallel inference across processors with :doc:`Automatic Device Selection (AUTO) <openvino-workflow/running-inference/inference-devices-and-modes/auto-device-selection>`
* Efficiently split inference between hardware cores with :doc:`Heterogeneous Execution (HETERO) <openvino-workflow/running-inference/inference-devices-and-modes/hetero-execution>`

Flexible Model and Pipeline Configuration
-----------------------------------------

Pipeline and model configuration features in OpenVINO Runtime allow you to easily optimize your application’s performance on any target hardware.

* :doc:`Automatic Batching <openvino-workflow/running-inference/inference-devices-and-modes/automatic-batching>` performs on-the-fly grouping of inference requests to maximize utilization of the target hardware’s memory and processing cores.
* :doc:`Performance Hints <openvino-workflow/running-inference/optimize-inference/high-level-performance-hints>` automatically adjust runtime parameters to prioritize for low latency or high throughput
* :doc:`Dynamic Shapes <openvino-workflow/running-inference/model-input-output/dynamic-shapes>` reshapes models to accept arbitrarily-sized inputs, increasing flexibility for applications that encounter different data shapes
* :doc:`Benchmark Tool <get-started/learn-openvino/openvino-samples/benchmark-tool>` characterizes model performance in various hardware and pipeline configurations

.. _additional-about-openvino/additional-resources:

Additional Resources
====================

* `OpenVINO Success Stories <https://www.intel.com/content/www/us/en/internet-of-things/ai-in-production/success-stories.html>`__ - See how Intel partners have successfully used OpenVINO in production applications to solve real-world problems.
* :doc:`Performance Benchmarks <about-openvino/performance-benchmarks>` - View results from benchmarking models with OpenVINO on Intel hardware.

