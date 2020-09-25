# OpenVINO™ Toolkit Documentation {#index}

## Introduction to OpenVINO™ Toolkit

OpenVINO™ toolkit quickly deploys applications and solutions that emulate human vision. Based on Convolutional Neural Networks (CNNs), the toolkit extends computer vision (CV) workloads across Intel® hardware, maximizing performance. The OpenVINO™ toolkit includes the Deep Learning Deployment Toolkit (DLDT).

OpenVINO™ toolkit:

- Enables CNN-based deep learning inference on the edge
- Supports heterogeneous execution across an Intel® CPU, Intel® Integrated Graphics, Intel® FPGA,  Intel® Neural Compute Stick 2 and Intel® Vision Accelerator Design with Intel® Movidius™ VPUs
- Speeds time-to-market via an easy-to-use library of computer vision functions and pre-optimized kernels
- Includes optimized calls for computer vision standards, including OpenCV\* and OpenCL™

## Toolkit Components 

OpenVINO™ toolkit includes the following components:

- Deep Learning Deployment Toolkit (DLDT)
    - [Deep Learning Model Optimizer](MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md) - A cross-platform command-line tool for importing models and
    preparing them for optimal execution with the Inference Engine. The Model Optimizer imports, converts, and optimizes models, which were trained in popular frameworks, such as Caffe*,
    TensorFlow*, MXNet*, Kaldi*, and ONNX*.
    - [Deep Learning Inference Engine](IE_DG/inference_engine_intro.md) - A unified API to allow high performance inference on many hardware types
    including the following:  
        - Intel® CPU
        - Intel® Integrated Graphics
        - Intel® Neural Compute Stick 2 
        - Intel® Vision Accelerator Design with Intel® Movidius™ vision processing unit (VPU)
    - [Samples](IE_DG/Samples_Overview.md) - A set of simple console applications demonstrating how to use the Inference Engine in your applications
    - [Tools](IE_DG/Tools_Overview.md) - A set of simple console tools to work with your models
- [Open Model Zoo](@ref omz_models_intel_index)     
    - [Demos](@ref omz_demos_README) - Console applications that demonstrate how you can use the Inference Engine in your applications to solve specific use cases
    - [Tools](IE_DG/Tools_Overview.md) - Additional tools to download models and check accuracy
    - [Documentation for Pretrained Models](@ref omz_models_intel_index) - Documentation for pretrained models is available in the [Open Model Zoo repository](https://github.com/opencv/open_model_zoo)
- [Post-Training Optimization tool](@ref pot_README) - A tool to calibrate a model and then execute it in the INT8 precision
- [Deep Learning Workbench](@ref workbench_docs_Workbench_DG_Introduction) -  A web-based graphical environment that allows you to easily use various sophisticated OpenVINO™ toolkit components
- Deep Learning Streamer (DL Streamer) – Streaming analytics framework, based on GStreamer, for constructing graphs of media analytics components. DL Streamer can be installed by the Intel® Distribution of OpenVINO™ toolkit installer. Its open source version is available on [GitHub](https://github.com/opencv/gst-video-analytics). For the DL Streamer documentation, see:
    - [DL Streamer Samples](IE_DG/Tools_Overview.md)
    - [API Reference](https://openvinotoolkit.github.io/dlstreamer_gst/)
    - [Elements](https://github.com/opencv/gst-video-analytics/wiki/Elements)
    - [Tutorial](https://github.com/opencv/gst-video-analytics/wiki/DL%20Streamer%20Tutorial)
- [OpenCV](https://docs.opencv.org/master/) - OpenCV* community version compiled for Intel® hardware
- Drivers and runtimes for OpenCL™ version 2.1
- [Intel® Media SDK](https://software.intel.com/en-us/media-sdk)

## Documentation Set Contents

OpenVINO™ toolkit documentation set includes the following documents:

- [Install the Intel® Distribution of OpenVINO™ Toolkit for Linux*](install_guides/installing-openvino-linux.md)
- [Install the Intel® Distribution of OpenVINO™ Toolkit for Linux with FPGA Support](install_guides/installing-openvino-linux-fpga.md)
- [Install the Intel® Distribution of OpenVINO™ Toolkit for Windows*](install_guides/installing-openvino-windows.md)
- [Install the Intel® Distribution of OpenVINO™ Toolkit for macOS*](install_guides/installing-openvino-macos.md)
- [Install the Intel® Distribution of OpenVINO™ Toolkit for Raspbian*](install_guides/installing-openvino-raspbian.md)
- [Install OpenVINO™ Deep Learning Workbench](@ref workbench_docs_Workbench_DG_Install_Workbench)
- [Introduction to Deep Learning Deployment Toolkit](IE_DG/Introduction.md)
- [Model Optimizer Developer Guide](MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
- [Inference Engine Developer Guide](IE_DG/Deep_Learning_Inference_Engine_DevGuide.md)
- [Post-Training Optimization Tool](@ref pot_README)
- [Inference Engine Samples](IE_DG/Samples_Overview.md)
- [Demo Applications](@ref omz_demos_README)
- [Tools](IE_DG/Tools_Overview.md)
- [Pretrained Models](@ref omz_models_intel_index)
- [Known Issues](IE_DG/Known_Issues_Limitations.md)
- [Legal Information](@ref omz_demos_README)

> **Typical Next Step:** [Introduction to Deep Learning Deployment Toolkit](IE_DG/Introduction.md)
