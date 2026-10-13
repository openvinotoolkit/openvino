---
sidebar_position: 1
sidebar_label: "GET STARTED"
description: "Learn how to install Intel® Distribution of OpenVINO™ toolkit on Windows, macOS, and Linux operating systems, using various installation methods."
---

# GET STARTED


<link rel="stylesheet" type="text/css" href="_static/css/getstarted_style.css"/>

<p id="GSG_introtext">Welcome to OpenVINO! This guide introduces installation and learning materials for Intel® Distribution of OpenVINO™ toolkit. The guide walks through the following steps:<br />
  <a href="https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/vision-monodepth" >Quick Start Example</a>
  <a href="get-started/install-openvino.html" >Install OpenVINO</a>
  <a href="#learn-openvino" >Learn OpenVINO</a></p>
<div style={{clear: 'both'}}> </div>

For a quick reference, check out
[the Quick Start Guide [pdf]](https://docs.openvino.ai/2026/_static/download/OpenVINO_Quick_Start_Guide.pdf)


## 1. Quick Start Example (No Installation Required) {#quick-start-example}

<div style={{width: '400px'}}>

![](https://user-images.githubusercontent.com/15709723/127752390-f6aa371f-31b5-4846-84b9-18dd4f662406.gif)

</div>

Try out OpenVINO's capabilities with this [quick start example](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/vision-monodepth)
that estimates depth in a scene using an OpenVINO monodepth model to quickly see how to load a model, prepare an image, inference the image, and display the result.


## 2. Install OpenVINO {#install-openvino-gsg}

See the [installation overview page](install-openvino/index.md) for options to install OpenVINO and set up a development environment on your device.

(get-started-learn-openvino/interactive-tutorials-python)=

## 3. Learn OpenVINO

OpenVINO provides a wide array of examples and documentation showing how to work with models, run inference, and deploy applications. Step through the sections below to learn the basics of OpenVINO and explore its advanced optimization features. For further details, visit [OpenVINO documentation](../documentation/index.md).


### OpenVINO Basics {#openvino-basics}

Learn the basics of working with models and inference in OpenVINO. Begin with “Hello World” Interactive Tutorials that show how to prepare models, run inference, and retrieve results using the OpenVINO API. Then, explore OpenVINO Code Samples that can be adapted for your own application.

(interactive-learn-openvino/interactive-tutorials-python)=

#### Interactive Tutorials - Jupyter Notebooks

Start with [interactive Python](learn-openvino/interactive-tutorials-python/index.md) that show the basics of model inference, the OpenVINO API, how to convert models to OpenVINO format, and more.

- [Hello Image Classification](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/hello-world)
  - Load an image classification model in OpenVINO and use it to apply a label to an image
- [OpenVINO Runtime API Tutorial](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/openvino-api)
  - Learn the basic Python API for working with models in OpenVINO
- [Convert TensorFlow Models to OpenVINO](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/tensorflow-classification-to-openvino)
- [Convert PyTorch Models to OpenVINO](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/pytorch-to-openvino/pytorch-onnx-to-openvino.ipynb)


#### OpenVINO Code Samples {#code-samples}

View [sample code](learn-openvino/openvino-samples/index.md) for various C++ and Python applications that can be used as a starting point for your own application. For C++ developers, step through the [Get Started with C++ Samples](learn-openvino/openvino-samples/get-started-demos.md) to learn how to build and run an image classification program that uses OpenVINO’s C++ API.


#### Integrate OpenVINO With Your Application {#integrate-openvino}

Learn how to [use the OpenVINO API to implement an inference pipeline](../openvino-workflow/running-inference/index.md) in your application.


### OpenVINO Advanced Features {#openvino-advanced-features}

OpenVINO provides features to improve your model’s performance, optimize your runtime, maximize your application’s throughput on target hardware, and much more. Visit the links below to learn more about these features and how to use them.

#### Model Compression and Quantization

Use OpenVINO’s model compression tools to reduce your model’s latency and memory footprint while maintaining good accuracy.

- Tutorial - [Quantization-Aware Training in PyTorch with NNCF](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/pytorch-quantization-aware-training)
- [Model Optimization Guide](../openvino-workflow/model-optimization/index.md)

#### Automated Device Configuration

OpenVINO’s hardware device configuration options enable you to write an application once and deploy it anywhere with optimal performance.

- Increase application portability and perform parallel inference across processors with [Automatic Device Selection (AUTO)](../openvino-workflow/running-inference/inference-devices-and-modes/auto-device-selection/index.md)
- Efficiently split inference between hardware cores with [Heterogeneous Execution (HETERO)](../openvino-workflow/running-inference/inference-devices-and-modes/hetero-execution.md)

#### Flexible Model and Pipeline Configuration

Pipeline and model configuration features in OpenVINO Runtime allow you to easily optimize your application’s performance on any target hardware.

- [Automatic Batching](../openvino-workflow/running-inference/inference-devices-and-modes/automatic-batching.md) performs on-the-fly grouping of inference requests to maximize utilization of the target hardware’s memory and processing cores.
- [Performance Hints](../openvino-workflow/running-inference/optimize-inference/high-level-performance-hints.md) automatically adjust runtime parameters to prioritize for low latency or high throughput
- [Dynamic Shapes](../openvino-workflow/running-inference/model-input-output/dynamic-shapes/index.md) reshapes models to accept arbitrarily-sized inputs, increasing flexibility for applications that encounter different data shapes
- [Benchmark Tool](learn-openvino/openvino-samples/benchmark-tool.md) characterizes model performance in various hardware and pipeline configurations

(additional-about-openvino/additional-resources)=

# Additional Resources

- [OpenVINO Success Stories](https://www.intel.com/content/www/us/en/internet-of-things/ai-in-production/success-stories.html) - See how Intel partners have successfully used OpenVINO in production applications to solve real-world problems.
- [Performance Benchmarks](../about-openvino/performance-benchmarks/index.md) - View results from benchmarking models with OpenVINO on Intel hardware.
