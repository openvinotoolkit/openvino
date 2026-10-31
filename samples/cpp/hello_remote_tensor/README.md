# Hello Remote Tensor C++ Sample

This sample demonstrates how to integrate OpenVINO into an existing Level Zero application.
It runs image classification on an Intel GPU with input and output tensors allocated in
Level Zero device memory, and all GPU memory operations of the application use the
Level Zero API. OpenVINO does not provide a dedicated Level Zero remote context; the sample
relies on Level Zero - OpenCL interoperability (LEO) of the GPU driver: it takes the GPU
plugin's default OpenCL context and converts it to the underlying Level Zero context with a
single `clGetContextInfo(..., CL_L0_CONTEXT_HANDLE, ...)` query. USM device pointers of the
remote tensors are valid in both APIs, so no further conversion is needed. The sample works
with any GPU plugin runtime (`GPU_RT_TYPE=OCL`, `ZE` or `COMBINED`).
It follows the structure of the
[Hello Classification C++ sample](../hello_classification/README.md), but the input
and output of the infer request are remote tensors allocated in GPU device memory.
Data are moved between the host and the device with a Level Zero command list.

Models with only one input and one output are supported.

For more detailed information on how this sample works, check the dedicated [article](https://docs.openvino.ai/2026/get-started/learn-openvino/openvino-samples/hello-remote-tensor.html)

## Requirements

| Options                     | Values                                                                                           |
| ----------------------------| -------------------------------------------------------------------------------------------------|
| Model Format                | OpenVINO™ toolkit Intermediate Representation (\*.xml + \*.bin), ONNX (\*.onnx)                  |
| Supported devices           | GPU (`GPU`, `GPU.0`, ...)                                                                        |

- An Intel GPU with a driver that provides Level Zero - OpenCL interoperability (LEO), that
  is, a driver where an OpenCL context can be queried for its Level Zero context with
  `clGetContextInfo(..., CL_L0_CONTEXT_HANDLE, ...)`. If the query fails, the sample
  reports an error.
- OpenVINO with the GPU plugin, built with any runtime. The default remote context is an
  OpenCL context (`ov::intel_gpu::ContextType::OCL`), which the sample converts through LEO.
  Only in the edge case of the Level Zero plugin (`GPU_RT_TYPE=ZE` or `COMBINED`) running on
  a driver without LEO, the default context is already a Level Zero context
  (`ov::intel_gpu::ContextType::ZE`) and its handle is used directly.
- OpenCL headers and the OpenCL ICD loader library.
- Level Zero headers to build, and the Level Zero loader library in the system at runtime
  (for example the `libze1` package on Ubuntu, or the loader shipped with the GPU driver
  on Windows).

The following C++ API is used in the application:

| Feature                   | API                                                            | Description                                                         |
| --------------------------| ---------------------------------------------------------------|---------------------------------------------------------------------|
| Remote Context            | ``ov::Core::get_default_context``,                             | Get the plugin's context and its native handle                      |
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
| OpenCL (LEO)              | ``clGetContextInfo`` with ``CL_L0_CONTEXT_HANDLE``             | Convert the plugin's OpenCL context to its Level Zero context       |

## Build and run

Build the C++ samples using the usual OpenVINO sample build procedure. The target
is `hello_remote_tensor`. For a standalone samples build, select it with
`BUILD_SAMPLE_NAME=hello_remote_tensor`; CMake skips this target with a status
message if Level Zero or OpenCL development files are unavailable. For nonstandard system
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
2. Obtain the plugin's default remote context and convert it to a Level Zero context. The
   default context is an `OCL` context whose `cl_context` is exposed under the
   `ov::intel_gpu::ocl_context` property; the sample converts it with the LEO query
   `clGetContextInfo(..., CL_L0_CONTEXT_HANDLE, ...)`. In the edge case of a `ZE` context
   (Level Zero plugin on a driver without LEO), the same property already holds the native
   `ze_context_handle_t`.
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

## Smoke tests

The sample tests check invalid arguments without requiring a GPU. Set
`IE_APP_PATH` to the directory containing the executable. Enable the
classification tests explicitly on a machine with an Intel GPU and a LEO-capable driver:

```sh
cd tests/samples_tests/smoke_tests
TEST_LEVEL_ZERO_REMOTE_TENSORS=1 python -m pytest test_hello_remote_tensor.py
```
