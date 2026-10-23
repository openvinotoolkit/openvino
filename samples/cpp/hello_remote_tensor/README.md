# Hello Remote Tensor C++ Sample

This sample demonstrates image classification on an Intel GPU using the GPU plugin's
Level Zero (ZE) runtime and Level Zero device memory, without including OpenCL headers
or calling the OpenCL API. It follows the structure of the
[Hello Classification C++ sample](../hello_classification/README.md), but the input
and output of the infer request are remote tensors allocated in GPU device memory.
Data are moved between the host and the device with a Level Zero command list.

Models with only one input and one output are supported.

## Requirements

| Options                     | Values                                                                                           |
| ----------------------------| -------------------------------------------------------------------------------------------------|
| Model Format                | OpenVINO™ toolkit Intermediate Representation (\*.xml + \*.bin), ONNX (\*.onnx)                  |
| Supported devices           | GPU with the Level Zero runtime (`GPU`, `GPU.0`, ...)                                            |

- An Intel GPU and a compatible Level Zero GPU driver.
- OpenVINO with the experimental Level Zero GPU plugin (`GPU_RT_TYPE=ZE`).
- A **pure Level Zero** remote context (`ov::intel_gpu::ContextType::ZE`).
  With drivers that enable OpenCL interoperability through LEO, the plugin can
  expose an OpenCL context even when inference uses Level Zero. That context is
  not supported by this sample; the sample reports an error rather than treating
  an OpenCL handle as a Level Zero handle. An OpenCL-only GPU plugin is also not
  supported.
- Level Zero development headers and the `ze_loader` library, available through
  the OpenVINO source build or installed on the system.

The following C++ API is used in the application:

| Feature                   | API                                                            | Description                                                         |
| --------------------------| ---------------------------------------------------------------|---------------------------------------------------------------------|
| Remote Context            | ``ov::Core::get_default_context``,                             | Get the plugin's Level Zero context and its native handle           |
|                           | ``ov::RemoteContext::get_params``                              |                                                                     |
| Remote Tensor             | ``ov::RemoteContext::create_tensor``,                          | Allocate USM device memory and get its Level Zero pointer           |
|                           | ``ov::RemoteTensor::get_params``                               |                                                                     |
| Basic Infer Flow          | ``ov::Core::read_model``,                                      | Read a model, compile it in the remote context, create an infer     |
|                           | ``ov::Core::compile_model``,                                   | request, and bind remote input and output tensors                   |
|                           | ``ov::CompiledModel::create_infer_request``,                   |                                                                     |
|                           | ``ov::InferRequest::set_input_tensor``,                        |                                                                     |
|                           | ``ov::InferRequest::set_output_tensor``                        |                                                                     |
| Synchronous Infer         | ``ov::InferRequest::infer``                                    | Do synchronous inference                                            |
| Preprocessing             | ``ov::preprocess::PrePostProcessor``                           | Resize and layout conversion of the original image on the device    |
| Level Zero                | ``zeInitDrivers``, ``zeMemGetAllocProperties``,                | Initialize Level Zero, find the allocation's device, and copy data  |
|                           | ``zeCommandListCreateImmediate``,                              | between host staging memory and device memory                       |
|                           | ``zeCommandListAppendMemoryCopy``,                             |                                                                     |
|                           | ``zeCommandListHostSynchronize``, ``zeMemAllocHost``           |                                                                     |

## Build and run

Build the C++ samples using the usual OpenVINO sample build procedure. The target
is `hello_remote_tensor`. For a standalone samples build, select it with
`BUILD_SAMPLE_NAME=hello_remote_tensor`; CMake skips this target with a status
message if Level Zero development files are unavailable. For nonstandard system
installations, set `LEVEL_ZERO_INCLUDE_DIR` to the directory containing
`level_zero/ze_api.h` and `LEVEL_ZERO_LOADER` to the loader library.

After setting up the OpenVINO runtime environment, run:

```sh
./hello_remote_tensor <path_to_model> <path_to_image> <device_name>
./hello_remote_tensor bvlcalexnet-12.onnx dog-224x224.bmp GPU
```

The device must be a GPU selected directly, for example `GPU`, `GPU.0` or a tile such
as `GPU.0.0`. CPU, AUTO, MULTI and HETERO are not supported. The sample prints the
top 10 classification results and returns a nonzero exit code on failure.

## What the sample demonstrates

1. Initialize Level Zero with `zeInitDrivers` and create `ov::Core`.
2. Obtain the plugin's default remote context and verify that its type is `ZE`.
   The current API exposes the native `ze_context_handle_t` under the legacy
   `ov::intel_gpu::ocl_context` property even for a ZE context.
3. Read the model and the image, and configure preprocessing as in Hello Classification.
4. Compile the model in the Level Zero context.
5. Allocate input and output remote tensors using `USM_DEVICE_BUFFER` and obtain
   their Level Zero USM pointers through `ov::intel_gpu::mem_handle`.
6. Query the input allocation's device with `zeMemGetAllocProperties` so the
   application's command list uses the same device as OpenVINO.
7. Upload the image through a Level Zero immediate command list, run synchronous
   inference, and download the output through Level Zero. Transfers use host
   staging buffers allocated with `zeMemAllocHost` in the same context, because
   Level Zero copies require memory accessible by the device. Each copy waits
   with `zeCommandListHostSynchronize` before the data are used.

OpenVINO owns the context and tensor allocations. The application must not destroy
the context or free the device pointers. Both tensors outlive all accesses through
Level Zero, and the application's command list and staging buffers are released
before those tensors. The upload completes before inference starts; `infer()`
completes before the download starts. The sample does not demonstrate external
context import (which the GPU plugin does not currently support for ZE), or
asynchronous queue sharing.

## Smoke tests

The sample tests check invalid arguments without requiring a GPU. Set
`IE_APP_PATH` to the directory containing the executable. Enable the
classification tests explicitly on a machine with a pure ZE context:

```sh
cd tests/samples_tests/smoke_tests
TEST_LEVEL_ZERO_REMOTE_TENSORS=1 python -m pytest test_hello_remote_tensor.py
```
