# Get Started {#get_started}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Install & Config
   
   Installing OpenVINO <openvino_docs_install_guides_overview>
   Additional Configurations <openvino_docs_install_guides_configurations_header>
   Uninstalling <openvino_docs_install_guides_uninstalling_openvino>
   Troubleshooting <openvino_docs_get_started_guide_troubleshooting>
   
.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Get Started Guides
   
   Interactive Tutorials (Python) <tutorials>
   Samples <openvino_docs_OV_UG_Samples_Overview>


@endsphinxdirective
 
@sphinxdirective
.. raw:: html

   <link rel="stylesheet" type="text/css" href="_static/css/getstarted_style.css">
   
   <p id="GSG_introtext">Welcome to OpenVINO! This guide introduces installation and learning materials for Intel® Distribution of OpenVINO™ toolkit. The guide walks through the following steps:<br />
     <a href="https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F201-vision-monodepth%2F201-vision-monodepth.ipynb" >Quick Start Example</a>
     <a href="openvino_docs_install_guides_overview.html" >Install OpenVINO</a>
     <a href="#learn-openvino" >Learn OpenVINO</a>
   </p>
   <div style="clear:both;"> </div> 
   
@endsphinxdirective

## <a name="quick-start-example"></a>1. Quick Start Example (No Installation Required)

<img src="https://user-images.githubusercontent.com/15709723/127752390-f6aa371f-31b5-4846-84b9-18dd4f662406.gif">

Try out OpenVINO's capabilities with this quick start example that estimates depth in a scene using an OpenVINO monodepth model. <a href="https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F201-vision-monodepth%2F201-vision-monodepth.ipynb">Run the example in a Jupyter Notebook inside your web browser</a> to quickly see how to load a model, prepare an image, inference the image, and display the result.

   
## <a name="install-openvino"></a>2. Install OpenVINO
   
Visit the [Install OpenVINO Overview page](./install_guides/installing-openvino-overview.md) to view options for installing OpenVINO and setting up a development environment on your device.
   
## <a name="get-started-tutorials"></a>3. Learn OpenVINO
   
OpenVINO provides a wide array of examples and documentation showing how to work with models, run inference, and deploy applications. Step through the sections below to learn the basics of OpenVINO and explore its advanced optimization features. Visit [OpenVINO’s documentation](./documentation.md) for further details on how to use its features and tools.
   
OpenVINO users of all experience levels can try <a href="https://www.intel.com/content/www/us/en/developer/tools/devcloud/edge/overview.html">Intel Dev Cloud</a>, a free web-based service for developing, testing, and running OpenVINO applications for free on an online cluster of the latest Intel hardware.


### <a name="openvino-basics"></a>OpenVINO Basics
Learn the basics of working with models and inference in OpenVINO. Begin with “Hello World” Interactive Tutorials that show how to prepare models, run inference, and retrieve results using the OpenVINO API. Then, explore other examples from the Open Model Zoo and OpenVINO Code Samples that can be adapted for your own application.
   

#### <a href="tutorials.html"><ins>Interactive Tutorials - Jupyter Notebooks</ins></a>
Start with interactive Python tutorials that show the basics of model inferencing, the OpenVINO API, how to convert models to OpenVINO format, and more.
* <a href="001-hello-world-with-output.html">Hello Image Classification</a> - Load an image classification model in OpenVINO and use it to apply a label to an image
* <a href="002-openvino-api-with-output.html">OpenVINO Runtime API Tutorial</a> - Learn the basic Python API for working with models in OpenVINO
* <a href="101-tensorflow-to-openvino-with-output.html">Convert TensorFlow Models to OpenVINO</a>
* <a href="102-pytorch-onnx-to-openvino-with-output.html">Convert PyTorch Models to OpenVINO</a>

#### <a href="openvino_docs_OV_UG_Samples_Overview.html"><ins>OpenVINO Code Samples</ins></a>
View sample code for various C++ and Python applications that can be used as a starting point for your own application. For C++ developers, step through the <a href="openvino_docs_get_started_get_started_demos.html">Basic OpenVINO Workflow</a> to learn how to build and run an image classification program that uses OpenVINO’s C++ API.
      
#### <a href="openvino_docs_OV_UG_Integrate_OV_with_your_application.html"><ins>Integrate OpenVINO With Your Application</ins></a>
Learn how to use the OpenVINO API to implement an inference pipeline in your application.


### <a name="openvino-advanced-features"></a>OpenVINO Advanced Features
OpenVINO provides features to improve your model’s performance, optimize your runtime, maximize your application’s throughput on target hardware, and much more. Visit the links below to learn more about these features and how to use them.

#### Model Compression and Quantization
Use OpenVINO’s model compression tools to reduce your model’s latency and memory footprint while maintaining good accuracy.
* Tutorial - <a href="111-detection-quantization-with-output.html">OpenVINO Post-Training Model Quantization</a>
* Tutorial - <a href="305-tensorflow-quantization-aware-training-with-output.html">Quantization-Aware Training in TensorFlow with OpenVINO NNCF</a>
* Tutorial - <a href="302-pytorch-quantization-aware-training-with-output.html">Quantization-Aware Training in PyTorch with NNCF</a>
* <a href="openvino_docs_model_optimization_guide.html">Model Optimization Guide</a>

#### Automated Device Configuration
OpenVINO’s hardware device configuration features enable you to write an application once and deploy it anywhere with optimal performance.
* Increase application portability with <a href="openvino_docs_OV_UG_supported_plugins_AUTO.html">Automatic Device Selection (AUTO)</a>
* Perform parallel inference across processors with <a href="openvino_docs_OV_UG_Running_on_multiple_devices.html">Multi-Device Execution (MULTI)</a>
* Efficiently split inference between hardware cores with <a href="openvino_docs_OV_UG_Hetero_execution.html">Heterogeneous Execution (HETERO)</a>

#### Flexible Model and Pipeline Configuration
Pipeline and model configuration features in OpenVINO Runtime allow you to easily optimize your application’s performance on any target hardware.
* <a href="openvino_docs_OV_UG_Automatic_Batching.html">Automatic Batching</a> performs on-the-fly grouping of inference requests to maximize utilization of the target hardware’s memory and processing cores.
* <a href="openvino_docs_OV_UG_Performance_Hints.html">Performance Hints</a> automatically adjust runtime parameters to prioritize for low latency or high throughput
* <a href="openvino_docs_OV_UG_DynamicShapes.html">Dynamic Shapes</a> reshapes models to accept arbitrarily-sized inputs, increasing flexibility for applications that encounter different data shapes
* <a href="openvino_inference_engine_tools_benchmark_tool_README.html">Benchmark Tool</a> characterizes model performance in various hardware and pipeline configurations
   
### <a name="additional-resources"></a>Additional Resources
* <a href="https://www.intel.com/content/www/us/en/internet-of-things/ai-in-production/success-stories.html">OpenVINO Success Stories</a> - See how Intel partners have successfully used OpenVINO in production applications to solve real-world problems.
* OpenVINO Supported Models (coming soon!) - Check which models OpenVINO supports on your hardware
* <a href="openvino_docs_performance_benchmarks.html">Performance Benchmarks</a> - View results from benchmarking models with OpenVINO on Intel hardware
