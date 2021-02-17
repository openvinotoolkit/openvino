Introduction to Inference Engine {#openvino_docs_IE_DG_inference_engine_intro}
================================

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
    - `libinference_engine.so`, which depends on `libinference_engine_transformations.so`, `libtbb.so`, `libtbbmalloc.so` and `libngraph.so`
* Windows* OS:
    - `inference_engine.dll`, which depends on `inference_engine_transformations.dll`, `tbb.dll`, `tbbmalloc.dll` and `ngraph.dll`
* macOS*:
    - `libinference_engine.dylib`, which depends on `libinference_engine_transformations.dylib`, `libtbb.dylib`, `libtbbmalloc.dylib` and `libngraph.dylib`

The required C++ header files are located in the `include` directory.

This library contains the classes to:
* Create Inference Engine Core object to work with devices and read network (InferenceEngine::Core)
* Manipulate network information (InferenceEngine::CNNNetwork)
* Execute and pass inputs and outputs (InferenceEngine::ExecutableNetwork and InferenceEngine::InferRequest)

### Plugin Libraries to read a network object ###

Starting from 2020.4 release, Inference Engine introduced a concept of `CNNNetwork` reader plugins. Such plugins can be automatically dynamically loaded by Inference Engine in runtime depending on file format:
* Linux* OS:
    - `libinference_engine_ir_reader.so` to read a network from IR
    - `libinference_engine_onnx_reader.so` to read a network from ONNX model format
* Windows* OS:
    - `inference_engine_ir_reader.dll` to read a network from IR
    - `inference_engine_onnx_reader.dll` to read a network from ONNX model format

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
| CPU    | `libMKLDNNPlugin.so`        | `libinference_engine_lp_transformations.so`                 | `MKLDNNPlugin.dll`       | `inference_engine_lp_transformations.dll`                                                              | `libMKLDNNPlugin.so`      | `inference_engine_lp_transformations.dylib` |
| GPU    | `libclDNNPlugin.so`         | `libinference_engine_lp_transformations.so`, `libOpenCL.so` | `clDNNPlugin.dll`        | `OpenCL.dll`, `inference_engine_lp_transformations.dll`                                                |  Is not supported            |  -                                          |
| MYRIAD | `libmyriadPlugin.so`        | `libusb.so`,                                                | `myriadPlugin.dll`       | `usb.dll`                                                                                              | `libmyriadPlugin.so`      | `libusb.dylib`                              |
| HDDL   | `libHDDLPlugin.so`          | `libbsl.so`, `libhddlapi.so`, `libmvnc-hddl.so`             | `HDDLPlugin.dll`         | `bsl.dll`, `hddlapi.dll`, `json-c.dll`, `libcrypto-1_1-x64.dll`, `libssl-1_1-x64.dll`, `mvnc-hddl.dll` |  Is not supported            |  -                                          |
| GNA    | `libGNAPlugin.so`           | `libgna.so`,                                                | `GNAPlugin.dll`          | `gna.dll`                                                                                              |  Is not supported            |  -                                          |
| HETERO | `libHeteroPlugin.so`        | Same as for selected plugins                                | `HeteroPlugin.dll`       | Same as for selected plugins                                                                           | `libHeteroPlugin.so`      |  Same as for selected plugins               |
| MULTI  | `libMultiDevicePlugin.so`   | Same as for selected plugins                                | `MultiDevicePlugin.dll`  | Same as for selected plugins                                                                           | `libMultiDevicePlugin.so` |  Same as for selected plugins               |

> **NOTE**: All plugin libraries also depend on core Inference Engine libraries.

Make sure those libraries are in your computer's path or in the place you pointed to in the plugin loader. Make sure each plugin's related dependencies are in the:

* Linux: `LD_LIBRARY_PATH`
* Windows: `PATH`
* macOS: `DYLD_LIBRARY_PATH`

On Linux and macOS, use the script `bin/setupvars.sh` to set the environment variables.

On Windows, run the `bin\setupvars.bat` batch file to set the environment variables.

To learn more about supported devices and corresponding plugins, see the [Supported Devices](supported_plugins/Supported_Devices.md) chapter.

Common Workflow for Using the Inference Engine API
--------------------------------------------------
The common workflow contains the following steps:

1. **Create Inference Engine Core object** - Create an `InferenceEngine::Core` object to work with different devices, all device plugins are managed internally by the `Core` object. Register extensions with custom nGraph operations (`InferenceEngine::Core::AddExtension`).

2. **Read the Intermediate Representation** - Using the `InferenceEngine::Core` class, read an Intermediate Representation file into an object of the `InferenceEngine::CNNNetwork` class. This class represents the network in the host memory.

3. **Prepare inputs and outputs format** - After loading the network, specify input and output precision and the layout on the network. For these specification, use the `InferenceEngine::CNNNetwork::getInputsInfo()` and `InferenceEngine::CNNNetwork::getOutputsInfo()`.

4. Pass per device loading configurations specific to this device (`InferenceEngine::Core::SetConfig`), and register extensions to this device (`InferenceEngine::Core::AddExtension`).

4. **Compile and Load Network to device** - Use the `InferenceEngine::Core::LoadNetwork()` method with specific device (e.g. `CPU`, `GPU`, etc.) to compile and load the network on the device. Pass in the per-target load configuration for this compilation and load operation.

5. **Set input data** - With the network loaded, you have an `InferenceEngine::ExecutableNetwork` object. Use this object to create an `InferenceEngine::InferRequest` in which you signal the input buffers to use for input and output. Specify a device-allocated memory and copy it into the device memory directly, or tell the device to use your application memory to save a copy.

6. **Execute** - With the input and output memory now defined, choose your execution mode:

    * Synchronously - `InferenceEngine::InferRequest::Infer()` method. Blocks until inference is completed.
    * Asynchronously - `InferenceEngine::InferRequest::StartAsync()` method. Check status with the `InferenceEngine::InferRequest::Wait()` method (0 timeout), wait, or specify a completion callback.

7. **Get the output** - After inference is completed, get the output memory or read the memory you provided earlier. Do this with the `InferenceEngine::IInferRequest::GetBlob()` method.


Further Reading
---------------

For more details on the Inference Engine API, refer to the [Integrating Inference Engine in Your Application](Integrate_with_customer_application_new_API.md) documentation.
