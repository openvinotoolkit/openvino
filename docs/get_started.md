# Get Started {#get_started}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:
   
   Installing OpenVINO <openvino_docs_install_guides_overview>
   Additional Configurations <openvino_docs_install_guides_configurations_header>
   Uninstalling <openvino_docs_install_guides_uninstalling_openvino>
   Troubleshooting <openvino_docs_get_started_guide_troubleshooting>
   
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

   
## <a name="install-openvino-gsg"></a>2. Install OpenVINO
   
See the [installation overview page](./install_guides/installing-openvino-overview.md) for options to install OpenVINO and set up a development environment on your device.
   
## <a name="get-started-tutorials"></a>3. Learn OpenVINO
   
OpenVINO provides a wide array of examples and documentation showing how to work with models, run inference, and deploy applications. Step through the sections below to learn the basics of OpenVINO and explore its advanced optimization features. For further details, visit [OpenVINO documentation](./documentation.md).
   
OpenVINO users of all experience levels can try [Intel® DevCloud](https://www.intel.com/content/www/us/en/developer/tools/devcloud/edge/overview.html), a free web-based service for developing, testing, and running OpenVINO applications on an online cluster of the latest Intel® hardware.


### <a name="openvino-basics"></a>OpenVINO Basics
Learn the basics of working with models and inference in OpenVINO. Begin with “Hello World” Interactive Tutorials that show how to prepare models, run inference, and retrieve results using the OpenVINO API. Then, explore other examples from the Open Model Zoo and OpenVINO Code Samples that can be adapted for your own application.
   

#### <a name="interactive-tutorials"></a>Interactive Tutorials - Jupyter Notebooks
   Start with <a href="tutorials.html">interactive Python tutorials</a> that show the basics of model inferencing, the OpenVINO API, how to convert models to OpenVINO format, and more.
* <a href="notebooks/001-hello-world-with-output.html">Hello Image Classification</a> - Load an image classification model in OpenVINO and use it to apply a label to an image
* <a href="notebooks/002-openvino-api-with-output.html">OpenVINO Runtime API Tutorial</a> - Learn the basic Python API for working with models in OpenVINO
* <a href="notebooks/101-tensorflow-to-openvino-with-output.html">Convert TensorFlow Models to OpenVINO</a>
* <a href="notebooks/102-pytorch-onnx-to-openvino-with-output.html">Convert PyTorch Models to OpenVINO</a>

#### <a name="code-samples"></a>OpenVINO Code Samples
View <a href="openvino_docs_OV_UG_Samples_Overview.html">sample code</a> for various C++ and Python applications that can be used as a starting point for your own application. For C++ developers, step through the [Get Started with C++ Samples](./get_started/get_started_demos.md) to learn how to build and run an image classification program that uses OpenVINO’s C++ API.
      
#### <a name="integrate-openvino"></a>Integrate OpenVINO With Your Application
Learn how to <a href="openvino_docs_OV_UG_Integrate_OV_with_your_application.html">use the OpenVINO API to implement an inference pipeline</a> in your application.


### <a name="openvino-advanced-features"></a>OpenVINO Advanced Features
OpenVINO provides features to improve your model’s performance, optimize your runtime, maximize your application’s throughput on target hardware, and much more. Visit the links below to learn more about these features and how to use them.

#### Model Compression and Quantization
Use OpenVINO’s model compression tools to reduce your model’s latency and memory footprint while maintaining good accuracy.
* Tutorial - <a href="notebooks/111-detection-quantization-with-output.html">OpenVINO Post-Training Model Quantization</a>
* Tutorial - <a href="notebooks/305-tensorflow-quantization-aware-training-with-output.html">Quantization-Aware Training in TensorFlow with OpenVINO NNCF</a>
* Tutorial - <a href="notebooks/302-pytorch-quantization-aware-training-with-output.html">Quantization-Aware Training in PyTorch with NNCF</a>
* <a href="notebooks/openvino_docs_model_optimization_guide.html">Model Optimization Guide</a>

#### Automated Device Configuration
OpenVINO’s hardware device configuration options enable you to write an application once and deploy it anywhere with optimal performance.
* Increase application portability with [Automatic Device Selection (AUTO)](./OV_Runtime_UG/auto_device_selection.md)
* Perform parallel inference across processors with [Multi-Device Execution (MULTI)](./OV_Runtime_UG/multi_device.md)
* Efficiently split inference between hardware cores with [Heterogeneous Execution (HETERO)](./OV_Runtime_UG/hetero_execution.md)

#### Flexible Model and Pipeline Configuration
Pipeline and model configuration features in OpenVINO Runtime allow you to easily optimize your application’s performance on any target hardware.
* [Automatic Batching](./OV_Runtime_UG/automatic_batching.md) performs on-the-fly grouping of inference requests to maximize utilization of the target hardware’s memory and processing cores.
* [Performance Hints](./OV_Runtime_UG/performance_hints.md) automatically adjust runtime parameters to prioritize for low latency or high throughput
* [Dynamic Shapes](./OV_Runtime_UG/ov_dynamic_shapes.md) reshapes models to accept arbitrarily-sized inputs, increasing flexibility for applications that encounter different data shapes
* [Benchmark Tool](../tools/benchmark_tool/README.md) characterizes model performance in various hardware and pipeline configurations
   
### <a name="additional-resources"></a>Additional Resources
* [OpenVINO Success Stories](https://www.intel.com/content/www/us/en/internet-of-things/ai-in-production/success-stories.html) - See how Intel partners have successfully used OpenVINO in production applications to solve real-world problems.
* [OpenVINO Supported Models](./resources/supported_models.md) - Check which models OpenVINO supports on your hardware.
* [Performance Benchmarks](./benchmarks/performance_benchmarks.md) - View results from benchmarking models with OpenVINO on Intel hardware.
