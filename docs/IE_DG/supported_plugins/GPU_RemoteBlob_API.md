Remote Blob API of GPU Plugin {#openvino_docs_IE_DG_supported_plugins_GPU_RemoteBlob_API}
================================

The GPU plugin implementation of the `RemoteContext` and `RemoteBlob` interfaces supports GPU 
pipeline developers who need video memory sharing and interoperability with existing native APIs 
such as OpenCL\*, Microsoft DirectX\*, or VAAPI\*.
Using these interfaces allows to avoid any memory copy overhead when plugging the OpenVINO™ inference 
into an existing GPU pipeline. It also enables OpenCL kernels participating in the pipeline to become 
native buffer consumers or producers of the OpenVINO™ inference.
Since the GPU plugin works on top of the clDNN library, the functionality above is also implemented 
using OpenCL and its sharing extensions provided by Intel®.

There are two interoperability scenarios that are supported for the Remote Blob API:

* GPU plugin context and memory objects can be constructed from low-level device, display, or memory 
handles and used to create the OpenVINO™ `ExecutableNetwork` or `Blob` class. 
* OpenCL context or buffer handles can be obtained from existing GPU plugin objects, and used in OpenCL processing.

Class and function declarations for the API are defined in the following files:
* Windows\*: `gpu/gpu_context_api_ocl.hpp` and `gpu/gpu_context_api_dx.hpp` 
* Linux\*: `gpu/gpu_context_api_ocl.hpp` and `gpu/gpu_context_api_va.hpp`

The most common way to enable the interaction of your application with the Remote Blob API is to use user-side utility classes 
and functions that consume or produce native handles directly. 

## Execution Context User-Side Wrappers

GPU plugin classes that implement the `RemoteContext` interface are responsible for context sharing.
Obtaining a pointer to a context object is the first step of sharing pipeline objects. 
The context object of the GPU plugin directly wraps OpenCL context, setting a scope for sharing 
`ExecutableNetwork` and `RemoteBlob` objects.
To create such objects within user context, explicitly provide the context to the plugin using the 
`make_shared_context()` overloaded function. Depending on the platform, the function accepts the 
`cl_context` handle, the pointer to the `ID3D11Device` interface, or the `VADisplay` handle, and 
returns a smart pointer to the `RemoteContext` plugin object.

If you do not provide any user context, the plugin uses its default internal context.
The plugin attempts to use the same internal context object as long as plugin options are kept the same.
Therefore, all ExecutableNetwork objects created during this time share the same context. 
Once the plugin options are changed, the internal context is replaced by the new one.

To request the current default context of the plugin, call the `GetDefaultContext()` method of the core engine. 
To request the internal context of the given `ExecutableNetwork`, use the `GetContext()` method.

## Shared Blob User-Side Wrappers

The classes that implement the `RemoteBlob` interface both are wrappers for native API 
memory handles (which can be obtained from them at any moment) and act just like regular OpenVINO™ 
`Blob` objects.

Once you obtain the context, you can use it to compile a new `ExecutableNetwork` or create `RemoteBlob` 
objects.
For network compilation, use a dedicated flavor of `LoadNetwork()`, which accepts the context as an 
additional parameter.

To create a shared blob from a native memory handle, use `make_shared_blob()` overloaded functions 
that can accept the `cl::Buffer`, `cl::Image2D`, `cl_mem` handles, and either `ID3D11Buffer`,
`ID3D11Texture2D` pointers or the `VASurfaceID` handle. 
All `make_shared_blob()` flavors return a smart pointer to the `Blob` object, which can be directly 
passed to the `SetBlob() `method of an inference request object.

## Direct NV12 video surface input

To support the direct consumption of a hardware video decoder output, plugin accepts two-plane video 
surfaces as arguments for the `make_shared_blob_nv12()` function, which creates an `NV12Blob` object 
and returns a smart pointer to it, which is cast to `Blob::Ptr`.

To ensure that the plugin generates the correct execution graph for the NV12 dual-plane input, set 
the `CLDNNConfigParams::KEY_CLDNN_NV12_TWO_INPUTS` plugin configuration flag to `PluginConfigParams::YES`.

## Low-Level Methods and Their Parameter Description

The high-level wrappers above bring a direct dependency on native APIs to the user program. 
If you want to avoid the dependency, you still can directly use the `CreateContext()`, 
`CreateBlob()`, and `getParams()` methods.
On this level, native handles are re-interpreted as void pointers and all arguments are passed 
using `std::map` containers that are filled with `std::string, InferenceEngine::Parameter` pairs.
Two types of map entries are possible: descriptor and container. The first map entry is a
descriptor, which sets the expected structure  and possible parameter values of the map.

**Parameter Map Entries**

| Key Name           | Description and Possible Parameter Values                                 |
|----------------|---------------------------------------------------------------------|
| `CONTEXT_TYPE` | Describes the type of the shared context in a map. Can be `OCL` (for pure OpenCL context) or `VA_SHARED` (for context shared with a video decoding device). |
| `OCL_CONTEXT` | Contains the OpenCL context handle. |
| `VA_DEVICE` | Contains the native video decoding device handle. Can be `VADisplay` or `ID3D11Device` (a pointer). |
| `SHARED_MEM_TYPE` | Describes the type of the shared memory buffer in a map. Can be `OCL_BUFFER` (clBuffer), `OCL_IMAGE2D` (clImage2D), `VA_SURFACE()`,  or `DX_BUFFER`.  |
| `MEM_HANDLE` | Contains the OpenCL memory handle. |
| `DEV_OBJECT_HANDLE` | Contains the native video decoder surface handle. |
| `VA_PLANE` | Contains the NV12 video decoder surface plane index. Can be `0` or `1`. |

> **NOTE**: To initialize the entry key and value, use the `GPU_PARAM_KEY()` or `GPU_PARAM_VALUE()` macro.

## Examples

Refer to the sections below to see pseudo-code of usage examples.

> **NOTE**: For low-level parameter usage examples, see the source code of user-side wrappers from the include files mentioned above.

### OpenCL Kernel Execution on a Shared Buffer

This example uses the OpenCL context obtained from an executable network object.

```cpp
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#include <CL/cl2.hpp>
#include <gpu/gpu_context_api_ocl.hpp>

...

// initialize the plugin and load the network 
InferenceEngine::Core ie;
auto exec_net = ie.LoadNetwork(net, "GPU", config);

// obtain the RemoteContext pointer from the executable network object
auto cldnn_context = exec_net.GetContext();
// obtain the OpenCL context handle from the RemoteContext,
// get device info and create a queue
cl::Context ctx = std::dynamic_pointer_cast<ClContext>(cldnn_context);
_device = cl::Device(_context.getInfo<CL_CONTEXT_DEVICES>()[0].get(), true);
cl::CommandQueue _queue;
cl_command_queue_properties props = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
_queue = cl::CommandQueue(_context, _device, props);

// create the OpenCL buffer within the obtained context
cl::Buffer shared_buffer(ctx, CL_MEM_READ_WRITE, image_size * num_channels, NULL, &err);
// wrap the buffer into RemoteBlob
auto shared_blob = gpu::make_shared_blob(input_info->getTensorDesc(), cldnn_context, shared_buffer);

...
// execute user kernel
cl::Kernel kernel(program, kernelName.c_str());
kernel.setArg(0, shared_buffer);
queue.enqueueNDRangeKernel(kernel,
							 cl::NDRange(0),
							 cl::NDRange(image_size),
							 cl::NDRange(1),
							 0, // wait events *
							 &profileEvent);
queue.finish();
...

// pass results to the inference
inf_req_shared.SetBlob(input_name, shared_blob);
inf_req_shared.Infer();

```

### Running GPU Plugin Inference within User-Supplied Shared Context

```cpp
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#include <CL/cl2.hpp>
#include <gpu/gpu_context_api_ocl.hpp>

...

cl::Context ctx = get_my_OpenCL_context();

// share the context with GPU plugin and compile ExecutableNetwork
auto remote_context = gpu::make_shared_context(ie, "GPU", ocl_instance->_context.get());
auto exec_net_shared = ie.LoadNetwork(net, remote_context);
auto inf_req_shared = exec_net_shared.CreateInferRequest();

...
// do OpenCL processing stuff
...

// run the inference
inf_req_shared.Infer();

```
### Direct Consuming of the NV12 VAAPI Video Decoder Surface on Linux

```cpp
#include <gpu/gpu_context_api_va.hpp>
#include <cldnn/cldnn_config.hpp>

...

// initialize the objects
CNNNetwork network = ie.ReadNetwork(xmlFileName, binFileName);

...

auto inputInfoItem = *inputInfo.begin();
inputInfoItem.second->setPrecision(Precision::U8);
inputInfoItem.second->setLayout(Layout::NCHW);
inputInfoItem.second->getPreProcess().setColorFormat(ColorFormat::NV12);

VADisplay disp = get_VA_Device();
// create the shared context object
auto shared_va_context = gpu::make_shared_context(ie, "GPU", disp);
// compile network within a shared context
ExecutableNetwork executable_network = ie.LoadNetwork(network,
													  shared_va_context,
													  { { CLDNNConfigParams::KEY_CLDNN_NV12_TWO_INPUTS,
													      PluginConfigParams::YES } });

// decode/inference loop
for (int i = 0; i < nframes; i++) {
	...
	// execute decoding and obtain decoded surface handle
	decoder.DecodeFrame();
	VASurfaceID va_surface = decoder.get_VA_output_surface();
	...
	//wrap decoder output into RemoteBlobs and set it as inference input
	auto nv12_blob = gpu::make_shared_blob_nv12(ieInHeight,
												ieInWidth,
												shared_va_context,
												va_surface
												);
	inferRequests[currentFrame].SetBlob(input_name, nv12_blob);
	inferRequests[currentFrame].StartAsync();
	inferRequests[prevFrame].Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
}
```

## See Also

* InferenceEngine::Core
* InferenceEngine::RemoteBlob
