# Inference Engine Developer Guide {#openvino_docs_IE_DG_Deep_Learning_Inference_Engine_DevGuide}

## Introduction to the OpenVINO™ Toolkit

The OpenVINO™ toolkit is a comprehensive toolkit that you can use to develop and deploy vision-oriented solutions on
Intel® platforms. Vision-oriented means the solutions use images or videos to perform specific tasks.
A few of the solutions use cases include autonomous navigation, digital surveillance cameras, robotics,
and mixed-reality headsets.

The OpenVINO™ toolkit:

* Enables CNN-based deep learning inference on the edge
* Supports heterogeneous execution across an Intel&reg; CPU, Intel&reg; Integrated Graphics, Intel&reg; Movidius&trade; Neural Compute Stick and Intel&reg; Neural Compute Stick 2
* Speeds time-to-market via an easy-to-use library of computer vision functions and pre-optimized kernels
* Includes optimized calls for computer vision standards including OpenCV\*, OpenCL&trade;, and OpenVX\*

The OpenVINO™ toolkit includes the following components:

* Intel® Deep Learning Deployment Toolkit (Intel® DLDT)
    - [Deep Learning Model Optimizer](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md) — A cross-platform command-line tool for importing models and
    preparing them for optimal execution with the Deep Learning Inference Engine. The Model Optimizer supports converting Caffe*,
    TensorFlow*, MXNet*, Kaldi*, ONNX* models.
    - [Deep Learning Inference Engine](inference_engine_intro.md) — A unified API to allow high performance inference on many hardware types
    including Intel® CPU, Intel® Processor Graphics, Intel® FPGA, Intel® Neural Compute Stick 2.
    - [nGraph](nGraph_Flow.md) — graph representation and manipulation engine which is used to represent a model inside Inference Engine and allows the run-time model construction without using Model Optimizer.
* [OpenCV](https://docs.opencv.org/) — OpenCV* community version compiled for Intel® hardware.
Includes PVL libraries for computer vision.
* Drivers and runtimes for OpenCL™ version 2.1
* [Intel® Media SDK](https://software.intel.com/en-us/media-sdk)
* [OpenVX*](https://software.intel.com/en-us/cvsdk-ovx-guide) — Intel's implementation of OpenVX*
optimized for running on Intel® hardware (CPU, GPU, IPU).
* [Demos and samples](Samples_Overview.md).


This Guide provides overview of the Inference Engine describing the typical workflow for performing
inference of a pre-trained and optimized deep learning model and a set of sample applications.

> **NOTES:** 
> - Before you perform inference with the Inference Engine, your models should be converted to the Inference Engine format using the Model Optimizer or built directly in run-time using nGraph API. To learn about how to use Model Optimizer, refer to the [Model Optimizer Developer Guide](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md). To learn about the pre-trained and optimized models delivered with the OpenVINO™ toolkit, refer to [Pre-Trained Models](@ref omz_models_intel_index).
> - [Intel® System Studio](https://software.intel.com/en-us/system-studio) is an all-in-one, cross-platform tool suite, purpose-built to simplify system bring-up and improve system and IoT device application performance on Intel® platforms. If you are using the Intel® Distribution of OpenVINO™ with Intel® System Studio, go to [Get Started with Intel® System Studio](https://software.intel.com/en-us/articles/get-started-with-openvino-and-intel-system-studio-2019).


## Table of Contents

* [Introduction to Intel® Deep Learning Deployment Toolkit](Introduction.md)

* [Inference Engine API Changes History](API_Changes.md)

* [Introduction to Inference Engine](inference_engine_intro.md)

* [Introduction to nGraph Flow](nGraph_Flow.md)

* [Understanding Inference Engine Memory Primitives](Memory_primitives.md)

* [Introduction to Inference Engine Device Query API](InferenceEngine_QueryAPI.md)

* [Adding Your Own Layers to the Inference Engine](Extensibility_DG/Intro.md)

* [Integrating Inference Engine in Your Application](Integrate_with_customer_application_new_API.md)

* [Migration from Inference Engine Plugin API to Core API](Migration_CoreAPI.md)

* [Introduction to Performance Topics](Intro_to_Performance.md)

* [Inference Engine Python API Overview](../../inference-engine/ie_bridges/python/docs/api_overview.md)

* [Using Dynamic Batching feature](DynamicBatching.md)

* [Using Static Shape Infer feature](ShapeInference.md)

* [Using Low-Precision 8-bit Integer Inference](Int8Inference.md)

* [Using Bfloat16 Inference](Bfloat16Inference.md)

* Utilities to Validate Your Converted Model
    * [Using Cross Check Tool for Per-Layer Comparison Between Plugins](../../inference-engine/tools/cross_check_tool/README.md)

* [Supported Devices](supported_plugins/Supported_Devices.md)
    * [GPU](supported_plugins/CL_DNN.md)
    * [CPU](supported_plugins/CPU.md)
    * [FPGA](supported_plugins/FPGA.md)
    * [VPU](supported_plugins/VPU.md)
      * [MYRIAD](supported_plugins/MYRIAD.md)
      * [HDDL](supported_plugins/HDDL.md)
    * [Heterogeneous execution](supported_plugins/HETERO.md)
    * [GNA](supported_plugins/GNA.md)
    * **NEW!** [MULTI](supported_plugins/MULTI.md)

* [Pre-Trained Models](@ref omz_models_intel_index)

* [Known Issues](Known_Issues_Limitations.md)

**Typical Next Step:** [Introduction to Intel® Deep Learning Deployment Toolkit](Introduction.md)
