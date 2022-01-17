# Introduction to Inference Engine {#openvino_docs_IE_DG_inference_engine_intro}

> **NOTE**: [Intel® System Studio](https://software.intel.com/en-us/system-studio) is an all-in-one, cross-platform tool suite, purpose-built to simplify system bring-up and improve system and IoT device application performance on Intel® platforms. If you are using the Intel® Distribution of OpenVINO™ with Intel® System Studio, go to [Get Started with Intel® System Studio](https://software.intel.com/en-us/articles/get-started-with-openvino-and-intel-system-studio-2019).

This Guide provides an overview of the Inference Engine describing the typical workflow for performing
inference of a pre-trained and optimized deep learning model and a set of sample applications.

> **NOTE**: Before you perform inference with the Inference Engine, your models should be converted to the Inference Engine format using the Model Optimizer or built directly in run-time using nGraph API. To learn about how to use Model Optimizer, refer to the [Model Optimizer Developer Guide](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md). To learn about the pre-trained and optimized models delivered with the OpenVINO™ toolkit, refer to [Pre-Trained Models](@ref omz_models_group_intel).

After you have used the Model Optimizer to create an Intermediate Representation (IR), use the Inference Engine to infer the result for a given input data.

Inference Engine is a set of C++ libraries providing a common API to deliver inference solutions on the platform of your choice: CPU, GPU, or VPU. Use the Inference Engine API to read the Intermediate Representation, set the input and output formats, and execute the model on devices. While the C++ libraries is the primary implementation, C libraries and Python bindings are also available.

For Intel® Distribution of OpenVINO™ toolkit, Inference Engine binaries are delivered within release packages.

The open source version is available in the [OpenVINO™ toolkit GitHub repository](https://github.com/openvinotoolkit/openvino) and can be built for supported platforms using the <a href="https://github.com/openvinotoolkit/openvino/wiki/BuildingCode">Inference Engine Build Instructions</a>.

To learn about how to use the Inference Engine API for your application, see the [Integrating Inference Engine in Your Application](Integrate_with_customer_application_new_API.md) documentation.

For complete API Reference, see the [Inference Engine API References](./api_references.html) section.

Inference Engine uses a plugin architecture. Inference Engine plugin is a software component that contains complete implementation for inference on a certain Intel&reg; hardware device: CPU, GPU, VPU, etc. Each plugin implements the unified API and provides additional hardware-specific APIs.

Modules in the Inference Engine component
-----------------------------------------

### Core Inference Engine Libraries ###

Your application must link to the core Inference Engine libraries:
* Linux* OS:
    - `libov_runtime.so`, which depends on `libtbb.so`, `libtbbmalloc.so`
* Windows* OS:
    - `ov_runtime.dll`, which depends on `tbb.dll`, `tbbmalloc.dll`
* macOS*:
    - `libov_runtime.dylib`, which depends on `libtbb.dylib`, `libtbbmalloc.dylib`

The required C++ header files are located in the `include` directory.

This library contains the classes to:
* Create Inference Engine Core object to work with devices and read network (InferenceEngine::Core)
* Manipulate network information (InferenceEngine::CNNNetwork)
* Execute and pass inputs and outputs (InferenceEngine::ExecutableNetwork and InferenceEngine::InferRequest)

### Plugin Libraries to read a network object ###

Starting from 2022.1 release, OpenVINO Runtime introduced a concept of frontend plugins. Such plugins can be automatically dynamically loaded by OpenVINO Runtime dynamically depending on file format:
* Unix* OS:
    - `libov_ir_frontend.so` to read a network from IR
    - `libov_paddle_frontend.so` to read a network from PaddlePaddle model format
    - `libov_onnx_frontend.so` to read a network from ONNX model format
* Windows* OS:
    - `ov_ir_frontend.dll` to read a network from IR
    - `ov_paddle_frontend.dll` to read a network from PaddlePaddle model format
    - `ov_onnx_frontend.dll` to read a network from ONNX model format

### Device-specific Plugin Libraries ###

For each supported target device, Inference Engine provides a plugin — a DLL/shared library that contains complete implementation for inference on this particular device. The following plugins are available:

| Plugin  | Device Type                   |
| ------- | ----------------------------- |
|CPU      |	Intel® Xeon® with Intel® AVX2 and AVX512, Intel® Core™ Processors with Intel® AVX2, Intel® Atom® Processors with Intel® SSE |
|GPU      | Intel® Processor Graphics, including Intel® HD Graphics and Intel® Iris® Graphics |
|MYRIAD   |	Intel® Neural Compute Stick 2 powered by the Intel® Movidius™ Myriad™ X |
|GNA      |	Intel&reg; Speech Enabling Developer Kit, Amazon Alexa* Premium Far-Field Developer Kit, Intel&reg; Pentium&reg; Silver J5005 Processor, Intel&reg; Pentium&reg; Silver N5000 Processor, Intel&reg; Celeron&reg; J4005 Processor, Intel&reg; Celeron&reg; J4105 Processor, Intel&reg; Celeron&reg; Processor N4100, Intel&reg; Celeron&reg; Processor N4000, Intel&reg; Core&trade; i3-8121U Processor, Intel&reg; Core&trade; i7-1065G7 Processor, Intel&reg; Core&trade; i7-1060G7 Processor, Intel&reg; Core&trade; i5-1035G4 Processor, Intel&reg; Core&trade; i5-1035G7 Processor, Intel&reg; Core&trade; i5-1035G1 Processor, Intel&reg; Core&trade; i5-1030G7 Processor, Intel&reg; Core&trade; i5-1030G4 Processor, Intel&reg; Core&trade; i3-1005G1 Processor, Intel&reg; Core&trade; i3-1000G1 Processor, Intel&reg; Core&trade; i3-1000G4 Processor |
|HETERO   | Automatic splitting of a network inference between several devices (for example if a device doesn't support certain layers|
|MULTI    | Simultaneous inference of the same network on several devices in parallel|

The table below shows the plugin libraries and additional dependencies for Linux, Windows and macOS platforms.

| Plugin | Library name for Linux      | Dependency libraries for Linux                              | Library name for Windows | Dependency libraries for Windows                                                                       | Library name for macOS       | Dependency libraries for macOS              |
|--------|-----------------------------|-------------------------------------------------------------|--------------------------|--------------------------------------------------------------------------------------------------------|------------------------------|---------------------------------------------|
| CPU    | `libov_intel_cpu_plugin.so`        | | `ov_intel_cpu_plugin.dll`       | | `libov_intel_cpu_plugin.so`      | |
| GPU    | `libov_intel_gpu_plugin.so`         | `libOpenCL.so` | `ov_intel_gpu_plugin.dll`        | `OpenCL.dll` |  Is not supported            |  -                                          |
| MYRIAD | `libov_intel_vpu_plugin.so` | `libusb.so`                                                 | `ov_intel_vpu_plugin.dll`| `usb.dll`                                                                                              | `libov_intel_vpu_plugin.so`   | `libusb.dylib`                              |
| HDDL   | `libHDDLPlugin.so`          | `libbsl.so`, `libhddlapi.so`, `libmvnc-hddl.so`             | `HDDLPlugin.dll`         | `bsl.dll`, `hddlapi.dll`, `json-c.dll`, `libcrypto-1_1-x64.dll`, `libssl-1_1-x64.dll`, `mvnc-hddl.dll` |  Is not supported            |  -                                          |
| GNA    | `libov_intel_gna_plugin.so`           | `libgna.so`,                                                | `ov_intel_gna_plugin.dll`          | `gna.dll`                                                                                              |  Is not supported            |  -                                          |
| HETERO | `libov_hetero_plugin.so`        | Same as for selected plugins                                | `ov_hetero_plugin.dll`       | Same as for selected plugins                                                                           | `libov_hetero_plugin.so`      |  Same as for selected plugins               |
| MULTI  | `libov_auto_plugin.so`   | Same as for selected plugins                                | `ov_auto_plugin.dll`  | Same as for selected plugins                                                                           | `libov_auto_plugin.so` |  Same as for selected plugins               |
| AUTO | `libov_auto_plugin.so`   | Same as for selected plugins                                | `ov_auto_plugin.dll`  | Same as for selected plugins                                                                           | `libov_auto_plugin.so` |  Same as for selected plugins               |

> **NOTE**: All plugin libraries also depend on core Inference Engine libraries.

Make sure those libraries are in your computer's path or in the place you pointed to in the plugin loader. Make sure each plugin's related dependencies are in the:

* Linux: `LD_LIBRARY_PATH`
* Windows: `PATH`
* macOS: `DYLD_LIBRARY_PATH`

On Linux and macOS, use the script `setupvars.sh` to set the environment variables.

On Windows, run the `setupvars.bat` batch file to set the environment variables.

To learn more about supported devices and corresponding plugins, see the [Supported Devices](supported_plugins/Supported_Devices.md) chapter.

Common Workflow for Using the Inference Engine API
--------------------------------------------------
The common workflow contains the following steps:

1. **Create Inference Engine Core object** - Create an `InferenceEngine::Core` object to work with different devices, all device plugins are managed internally by the `Core` object. Register extensions with custom nGraph operations (`InferenceEngine::Core::AddExtension`).

2. **Read the Intermediate Representation** - Using the `InferenceEngine::Core` class, read an Intermediate Representation file into an object of the `InferenceEngine::CNNNetwork` class. This class represents the network in the host memory.

3. **Prepare inputs and outputs format** - After loading the network, specify input and output precision and the layout on the network. For these specification, use the `InferenceEngine::CNNNetwork::getInputsInfo()` and `InferenceEngine::CNNNetwork::getOutputsInfo()`.

4. **Pass per device loading configurations** specific to this device (`InferenceEngine::Core::SetConfig`) and register extensions to this device (`InferenceEngine::Core::AddExtension`).

5. **Compile and Load Network to device** - Use the `InferenceEngine::Core::LoadNetwork()` method with specific device (e.g. `CPU`, `GPU`, etc.) to compile and load the network on the device. Pass in the per-target load configuration for this compilation and load operation.

6. **Set input data** - With the network loaded, you have an `InferenceEngine::ExecutableNetwork` object. Use this object to create an `InferenceEngine::InferRequest` in which you signal the input buffers to use for input and output. Specify a device-allocated memory and copy it into the device memory directly, or tell the device to use your application memory to save a copy.

7. **Execute** - With the input and output memory now defined, choose your execution mode:

    * Synchronously - `InferenceEngine::InferRequest::Infer()` method. Blocks until inference is completed.
    * Asynchronously - `InferenceEngine::InferRequest::StartAsync()` method. Check status with the `InferenceEngine::InferRequest::Wait()` method (0 timeout), wait, or specify a completion callback.

8. **Get the output** - After inference is completed, get the output memory or read the memory you provided earlier. Do this with the `InferenceEngine::IInferRequest::GetBlob()` method.


Further Reading
---------------

For more details on the Inference Engine API, refer to the [Integrating Inference Engine in Your Application](Integrate_with_customer_application_new_API.md) documentation.
