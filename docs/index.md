# OpenVINO™ Toolkit Overview {#index}

## Introduction

OpenVINO™ toolkit is a comprehensive toolkit for quickly developing applications and solutions that solve a variety of tasks including emulation of human vision, automatic speech recognition, natural language processing, recommendation systems, and many others. Based on latest generations of artificial neural networks, including Convolutional Neural Networks (CNNs), recurrent and attention-based networks, the toolkit extends computer vision and non-vision workloads across Intel® hardware, maximizing performance. It accelerates applications with high-performance, AI and deep learning inference deployed from edge to cloud.

OpenVINO™ toolkit:

- Enables CNN-based deep learning inference on the edge
- Supports heterogeneous execution across an Intel® CPU, Intel® Integrated Graphics,  Intel® Neural Compute Stick 2 and Intel® Vision Accelerator Design with Intel® Movidius™ VPUs
- Speeds time-to-market via an easy-to-use library of computer vision functions and pre-optimized kernels
- Includes optimized calls for computer vision standards, including OpenCV\* and OpenCL™

## OpenVINO™ Toolkit Workflow

The following diagram illustrates the typical OpenVINO™ workflow (click to see the full-size image):
![](img/OpenVINO-diagram.png)

### Model Preparation, Conversion and Optimization

You can use your framework of choice to prepare and train a deep learning model or just download a pre-trained model from the Open Model Zoo. The Open Model Zoo includes deep learning solutions to a variety of vision problems, including object recognition, face recognition, pose estimation, text detection, and action recognition, at a range of measured complexities.
Several of these pre-trained models are used also in the [code samples](IE_DG/Samples_Overview.md) and [application demos](@ref omz_demos_README). To download models from the Open Model Zoo, the [Model Downloader](@ref omz_tools_downloader_README) tool is used.

One of the core component of the OpenVINO™ toolkit is the [Model Optimizer](MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md) a cross-platform command-line
tool that converts a trained neural network from its source framework to an open-source, nGraph-compatible [Intermediate Representation (IR)](MO_DG/IR_and_opsets.md) for use in inference operations. The Model Optimizer imports models trained in popular frameworks such as Caffe*, TensorFlow*, MXNet*, Kaldi*, and ONNX* and performs a few optimizations to remove excess layers and group operations when possible into simpler, faster graphs.
![](img/OV-diagram-step2.png)

If your neural network model contains layers that are not in the list of known layers for supported frameworks, you can adjust the conversion and optimization process through use of  [Custom Layers](HOWTO/Custom_Layers_Guide.md).

Run the [Accuracy Checker utility](@ref omz_tools_accuracy_checker) either against source topologies or against the output representation to evaluate the accuracy of inference. The Accuracy Checker is also part of the [Deep Learning Workbench](@ref workbench_docs_Workbench_DG_Introduction), an integrated web-based performance analysis studio.

Use the [Post-training Optimization Tool](@ref pot_README) to accelerate the inference of a deep learning model by quantizing it to INT8.

Useful documents for model optimization:
* [Model Optimizer Developer Guide](MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
* [Intermediate Representation and Opsets](MO_DG/IR_and_opsets.md)
* [Custom Layers Guide](HOWTO/Custom_Layers_Guide.md)
* [Accuracy Checker utility](@ref omz_tools_accuracy_checker)
* [Post-training Optimization Tool](@ref pot_README)
* [Deep Learning Workbench](@ref workbench_docs_Workbench_DG_Introduction)
* [Model Downloader](@ref omz_tools_downloader) utility
* [Intel's Pretrained Models (Open Model Zoo)](@ref omz_models_group_intel)
* [Public Pretrained Models (Open Model Zoo)](@ref omz_models_group_public)

### Running and Tuning Inference
The other core component of OpenVINO™ is the [Inference Engine](IE_DG/Deep_Learning_Inference_Engine_DevGuide.md), which manages the loading and compiling of the optimized neural network model, runs inference operations on input data, and outputs the results. Inference Engine can execute synchronously or asynchronously, and its plugin architecture manages the appropriate compilations for execution on multiple Intel® devices, including both workhorse CPUs and specialized graphics and video processing platforms (see below, Packaging and Deployment).

You can use OpenVINO™ Tuning Utilities with the Inference Engine to trial and test inference on your model. The Benchmark utility uses an input model to run iterative tests for throughput or latency measures, and the [Cross Check Utility](../inference-engine/tools/cross_check_tool/README.md) compares performance of differently configured inferences. 

For a full browser-based studio integrating these other key tuning utilities, try the [Deep Learning Workbench](@ref workbench_docs_Workbench_DG_Introduction).
![](img/OV-diagram-step3.png)

OpenVINO™ toolkit includes a set of [inference code samples](IE_DG/Samples_Overview.md) and [application demos](@ref omz_demos) showing how inference is run and output processed for use in retail environments, classrooms, smart camera applications, and other solutions.

OpenVINO also makes use of open-source and Intel™ tools for traditional graphics processing and performance management. Intel® Media SDK supports accelerated rich-media processing, including transcoding. OpenVINO™ optimizes calls to the rich OpenCV and OpenVX libraries for processing computer vision workloads. And the new DL Streamer integration further accelerates video pipelining and performance.

Useful documents for inference tuning:
* [Inference Engine Developer Guide](IE_DG/Deep_Learning_Inference_Engine_DevGuide.md)
* [Inference Engine API References](./api_references.html)
* [Inference Code Samples](IE_DG/Samples_Overview.md)
* [Application Demos](@ref omz_demos)
* [Low Precision Optimization Guide] (@ref pot_docs_LowPrecisionOptimizationGuide)
* [Deep Learning Workbench Guide](@ref workbench_docs_Workbench_DG_Introduction)
* [Intel Media SDK](https://github.com/Intel-Media-SDK/MediaSDK)
* [DL Streamer Samples](@ref gst_samples_README)
* [OpenCV](https://docs.opencv.org/master/)
* [OpenVX](https://software.intel.com/en-us/openvino-ovx-guide)

### Packaging and Deployment
The Intel Distribution of OpenVINO™ toolkit outputs optimized inference runtimes for the following devices:
* Intel® CPUs
* Intel® Processor Graphics
* Intel® Neural Compute Stick 2
* Intel® Vision Accelerator Design with Intel® Movidius™ VPUs

The Inference Engine's plug-in architecture can be extended to meet other specialized needs. [Deployment Manager](./install_guides/deployment-manager-tool.md) is a Python* command-line tool that assembles the tuned model, IR files, your application, and required dependencies into a runtime package for your target device. It outputs packages for CPU, GPU, and VPU on Linux* and Windows*, and Neural Compute Stick-optimized packages with Linux.

* [Inference Engine Integration Workflow](IE_DG/Integrate_with_customer_application_new_API.md)
* [Inference Engine API References](./api_references.html)
* [Inference Engine Plug-in Developer Guide](./ie_plugin_api/index.html)
* [Deployment Manager Guide](./install_guides/deployment-manager-tool.md)


## OpenVINO™ Toolkit Components 

Intel® Distribution of OpenVINO™ toolkit includes the following components:

- [Deep Learning Model Optimizer](MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md): A cross-platform command-line tool for importing models and preparing them for optimal execution with the Inference Engine. The Model Optimizer imports, converts, and optimizes models, which were trained in popular frameworks, such as Caffe*, TensorFlow*, MXNet*, Kaldi*, and ONNX*.
- [Deep Learning Inference Engine](IE_DG/Deep_Learning_Inference_Engine_DevGuide.md): A unified API to allow high performance inference on many hardware types including Intel® CPU, Intel® Integrated Graphics, Intel® Neural Compute Stick 2, Intel® Vision Accelerator Design with Intel® Movidius™ vision processing unit (VPU).
- [Inference Engine Samples](IE_DG/Samples_Overview.md): A set of simple console applications demonstrating how to use the Inference Engine in your applications.
- [Deep Learning Workbench](@ref workbench_docs_Workbench_DG_Introduction): A web-based graphical environment that allows you to easily use various sophisticated OpenVINO™ toolkit components.
- [Post-training Optimization Tool](@ref pot_README): A tool to calibrate a model and then execute it in the INT8 precision.
- Additional Tools: A set of tools to work with your models including [Benchmark App](../inference-engine/tools/benchmark_tool/README.md), [Cross Check Tool](../inference-engine/tools/cross_check_tool/README.md), [Compile tool](../inference-engine/tools/compile_tool/README.md).
- [Open Model Zoo](@ref omz_models_group_intel)     
    - [Demos](@ref omz_demos): Console applications that provide robust application templates to help you implement specific deep learning scenarios.
    - Additional Tools: A set of tools to work with your models including [Accuracy Checker Utility](@ref omz_tools_accuracy_checker) and [Model Downloader](@ref omz_tools_downloader).
    - [Documentation for Pretrained Models](@ref omz_models_group_intel): Documentation for pre-trained models that are available in the [Open Model Zoo repository](https://github.com/openvinotoolkit/open_model_zoo).
- Deep Learning Streamer (DL Streamer): Streaming analytics framework, based on GStreamer, for constructing graphs of media analytics components. DL Streamer can be installed by the Intel® Distribution of OpenVINO™ toolkit installer. Its open-source version is available on [GitHub](https://github.com/openvinotoolkit/dlstreamer_gst). For the DL Streamer documentation, see:
    - [DL Streamer Samples](@ref gst_samples_README)
    - [API Reference](https://openvinotoolkit.github.io/dlstreamer_gst/)
    - [Elements](https://github.com/openvinotoolkit/dlstreamer_gst/wiki/Elements)
    - [Tutorial](https://github.com/openvinotoolkit/dlstreamer_gst/wiki/DL-Streamer-Tutorial)
- [OpenCV](https://docs.opencv.org/master/) : OpenCV* community version compiled for Intel® hardware
- [Intel® Media SDK](https://software.intel.com/en-us/media-sdk) (in Intel® Distribution of OpenVINO™ toolkit for Linux only)

OpenVINO™ Toolkit opensource version is available on [GitHub](https://github.com/openvinotoolkit/openvino). For building the Inference Engine from the source code, see the <a href="https://github.com/openvinotoolkit/openvino/wiki/BuildingCode">build instructions</a>.
