# Configurations for Intel® Processor Graphics (GPU) with OpenVINO™ {#openvino_docs_install_guides_configurations_for_intel_gpu}


@sphinxdirective

.. _gpu guide:

@endsphinxdirective



## Linux

If you have installed OpenVINO Runtime from the archive file, APT, or YUM, follow these steps to work with GPU:

1. Check your operating system and its version: 
   ```sh 
   cat /etc/os-release
   ```

2. You can install the latest available **Intel® Graphics Compute Runtime for OpenCL™** for your OS, see the [Install Guides](https://dgpu-docs.intel.com/installation-guides/index.html) or choose one that we used in internal tests

Operation System | Driver version
--- |-------------------------
Ubuntu 20.04 | [22.35.24055](https://github.com/intel/compute-runtime/releases/tag/22.28.23726)  
Ubuntu 18.04 | [21.38.21026](https://github.com/intel/compute-runtime/releases/tag/22.28.23726)  
CentOS 7 | [19.41.14441](https://github.com/intel/compute-runtime/releases/tag/22.28.23726)  
RHEL 8 | [22.28.23726](https://github.com/intel/compute-runtime/releases/tag/22.28.23726)                     

> **NOTE**: To use the **Intel® Iris® Xe MAX Graphics**, see the [Intel® Iris® Xe MAX Graphics with Linux*](https://dgpu-docs.intel.com/devices/iris-xe-max-graphics/index.html) page for driver installation instructions.

Additionally, it is recommended that you refer to the [Intel® Graphics Compute Runtime Github page](https://github.com/intel/compute-runtime/) for the latest instructions and recommendations on GPU driver installation.

You've completed all required configuration steps to perform inference on processor graphics.

@sphinxdirective

.. _gpu guide windows:

@endsphinxdirective

## Windows

This section will help you check if you require driver installation. Install indicated version or higher.

If your applications offload computation to **Intel® Integrated Graphics**, you must have the Intel Graphics Driver for Windows installed on your hardware.
[Download and install the recommended version](https://downloadcenter.intel.com/download/30079/Intel-Graphics-Windows-10-DCH-Drivers). 

To check if you have this driver installed:

1. Type **device manager** in your **Search Windows** box and press Enter. The **Device Manager** opens.

2. Click the drop-down arrow to view the **Display adapters**. You can see the adapter that is installed in your computer:
   ![](../img/DeviceManager.PNG)

3. Right-click the adapter name and select **Properties**.

4. Click the **Driver** tab to see the driver version. 
   ![](../img/DeviceDriverVersion.PNG)

You are done updating your device driver and are ready to use your GPU.
## What’s Next?

You can try out the toolkit with:

Developing in Python:
   * [Start with tensorflow models with OpenVINO™](https://docs.openvino.ai/latest/notebooks/101-tensorflow-to-openvino-with-output.html)
   * [Start with ONNX and PyTorch models with OpenVINO™](https://docs.openvino.ai/latest/notebooks/102-pytorch-onnx-to-openvino-with-output.html)
   * [Start with PaddlePaddle models with OpenVINO™](https://docs.openvino.ai/latest/notebooks/103-paddle-onnx-to-openvino-classification-with-output.html)

Developing in C++:
   * [Image Classification Async C++ Sample](@ref openvino_inference_engine_samples_classification_sample_async_README)
   * [Hello Classification C++ Sample](@ref openvino_inference_engine_samples_hello_classification_README)
   * [Hello Reshape SSD C++ Sample](@ref openvino_inference_engine_samples_hello_reshape_ssd_README)

