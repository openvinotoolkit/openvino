# NPU Plugin

## Introduction &nbsp;

This is the OpenVINO Plugin for Intel&reg; Neural Processing Unit (NPU) devices.

&nbsp;
## Supported Platforms

OpenVINO™ toolkit is officially supported and validated on the following platforms:

| Host                         | NPU device  | OS (64-bit)                          |
| :---                         | :---        | :---                                 |
| Meteor Lake (integrated NPU)   | NPU 3720    | Ubuntu* 22, MS Windows* 11           |
| Lunar Lake (integrated NPU)    | NPU 4000    | Ubuntu* 22, MS Windows* 11           |


&nbsp;
## High Level Design

![High Level Design](./docs/img/high_level_design.png)


&nbsp;
## Description

NPU Plugin is a software library that:
* Implements the unified OpenVINO Plugin API used to compile and execute neural networks on NPU devices.
* Uses the graph extension API exposed by the driver to convert the OpenVINO specific representation of the model into a proprietary format. The compiler performs platform specific optimizations in order to efficiently schedule the execution of layers and memory transactions on various NPU hardware submodules.
* Uses the Level Zero API implemented by the NPU user mode driver (UMD) to execute the model on the device.

The plugin library is included inside the OpenVINO package while the compiler is packaged inside UMD and released separately.

Note: Aligning with the platform and OpenVINO documentation, neural networks will be referred to with the more generic term of models in the rest of this document.

&nbsp;
## Model Compilation

NPU plugin implements the OpenVINO Core "compile_model" API that converts the model representation into a proprietary format that can be executed on the NPU device:

```
    ov::CompiledModel compiled_model = core.compile_model(model, "NPU" [, config]);
```

### Model caching

There are two important compilation related metrics when executing models on NPU devices:
* First Ever Inference Latency (FEIL): Measures all steps required to compile and execute a model on the device for the first time. It includes model compilation time, the time required to load and initialize the model on the device and the first inference execution.
* First Inference Latency (FIL): Measures the time required to load and initialize the pre-compiled model on the device and the first inference execution.


#### UMD dynamic model caching

UMD model caching is enabled by default in the current NPU driver to improve time to first inference (FIL). The model is stored in the cache after the compilation (included in FEIL) based on a hash key. The UMD generates the key from the input IR model and build arguments and then requests the DirectX Shader cache session to store the model with the computed key. Any subsequent request to compile the same IR model with the same arguments would cause the pre-compiled model to be read from the cache instead of being recompiled.

#### OpenVINO model caching

It is enabled when `ov::cache_dir` property is set and it is a common mechanism for all OpenVINO plugins. UMD model caching will be automatically bypassed by the NPU plugin when `ov::cache_dir` is set so the model will only be stored in the OpenVINO cache after the compilation. When a cache hit occurs for subsequent compilation requests, plugin will import the model instead of recompiling it.

More details about OpenVINO model caching can be found here: [Model Caching Overview](https://docs.openvino.ai/2023.0/openvino_docs_OV_UG_Model_caching_overview.html).

### Compiler adapters

Two additional layers are required to support the compiler from driver:
* Compiler Adapter - It serializes the OpenVINO internal representation of the model (ov::model) into an in-memory IR that will be provided to the NPU driver  
* VCL - It deserializes the in-memory IR given by the NPU driver and prepares it for the compiler

The interface between plugin and driver is based on an in-memory IR to facilitate backward and forward compatibility between two software packages (OpenVINO and NPU driver) that inherently have a different release cadence.

&nbsp;
## Model Execution

NPU plugin will use the Level Zero (L0) API to execute the precompiled model on the NPU Device. The inference is executed as a standard L0 workload by describing the required tasks inside a command list and by submitting the list to the command queue for execution. The plugin will not use the CPU to execute any part of the inference workload. No pre/post processing workloads are executed on the CPU either, the entire inference will be offloaded on the NPU device.

### Device management

There is currently no support for multiple devices, which means only one level-zero device will be enumerated during level-zero backend initialization. Support for multiple devices will be added in future releases.

### Inference pipeline

The result of the model compilation is represented through an IGraph object, which contains a valid level zero graph handle that can later be used to execute multiple inferences in parallel for the same model. By default, weights are loaded into the NPU memory right after the model is compiled, but this step can be postponed until the creation of the first inference request through the use of an internal NPU property: "NPU_DEFER_WEIGHTS_LOAD".

Users can create one or more inference requests for a compiled model using OpenVINO API:

```
    ov::InferRequest request = compiled_model.create_infer_request();
```

One unique level zero command queue is currently used to execute all inference requests created for the same model.


### Memory allocation

Each inference request is linked to an already compiled model and maintains a reference to the common model description. The level zero (user mode) driver parses the model description and returns the number and size of input/output buffers that need to be allocated per inference request. NPU plugin allocates input/output buffers through dedicated level zero API. Pointers to input/output buffers need to be provided when the command list is created so a different command list is used for each inference request.  

The set of input and output buffers allocated by the NPU plugin for each inference request can be retrieved using the following API:
```
    ov::Tensor requestTensor = inferRequest.get_tensor(tensor_name);
```

Users can access the "data" member of the specified tensor and can directly populate it or read it. This is the recommended usage of input and output buffers that guarantees no extra copies will be performed by the NPU plugin.  

Alternatively, users can configure different input or output buffers to be used during inference:
```
    inferRequest.set_tensor(tensor_name, tensor);
```

Since these tensors are not natively accessible by the NPU device, plugin will perform the required memory copies to and from the original buffers that were allocated when the inference request was created. This has an impact on the inference latency and should be avoided whenever possible.

Once the inference request is created and input/output tensors are prepared for the inference, the execution can be triggered either synchronously or asynchronously:
* Synchronous execution:
```
    inferRequest.infer();
```
* Asynchronous execution using:
```
    inferRequest.start_async();
    inferRequest.wait(); // optional, in case user callback is not provided
```

Multiple inferences can be executed in parallel, either from the same application thread through the use of asynchronous methods or from multiple threads with any of the available methods (synchronous or asynchronous). There is an optimal number of inference requests to be executed in parallel that would yield the best throughput without impacting the latency observed for each inference. This optimal number of requests is different for each model and depends on the ratio between the duration of the model execution on the DPU HW and the rest of the latency required to pass the request through the entire software stack. The NPU plugin returns the optimal number of inference requests through a dedicated property (`ov::optimal_number_of_infer_requests`).

Note: the current implementation of this property does not estimate or check the duration of the model execution. A fixed number of recommended inference requests is currently returned based on the existing performance data gathered from a reduced set of models with different topologies.

&nbsp;
## Supported Properties

Properties can be used to query and adjust the behavior of the NPU plugin itself or various parameters that control model compilation and execution.  

The following methods are made available to return the value of a given property (at core level or model specific):
```
    plugin_properties = ov.get_property("NPU", <property_name>);
    [...]
    model_properties = compiled_model.get_property(<property_name>);
```

The following methods are made available to set the value of a given property (at core level or model specific):
```
    ov.set_property("NPU", {{Key, Value}});
    [...]
    compiled_model.set_property({{Key, Value}});
```

The following properties are supported:

| Parameter Name |            | Description | Supported Values | Default Value |
| :---           | :---       | :---        |:---              |:--            |
| `ov::supported_properties`/</br>`SUPPORTED_METRICS`/</br>`SUPPORTED_CONFIG_KEYS` | RO | Returns a list of all supported properties.</br> Can be queried on runtime. | `N/A` | `N/A` |
| `ov::caching_properties`/</br>`CACHING_PROPERTIES` | RW | Returns a list of all properties that are used by OpenVINO cache to build the hash key. | `N/A` | `N/A` |
| `ov::compilation_num_threads`/</br>`COMPILATION_NUM_THREADS` | RW | Maximum number of threads that can be used for compilation tasks. | `N/A` | `N/A` |
| `ov::num_streams`/</br>`NUM_STREAMS` | RO | Not used by the NPU plugin.</br> Always set to 1. | `AUTO/`</br>`INT` | `1` |
| `ov::optimal_number_of_infer_requests`/</br>`OPTIMAL_NUMBER_OF_INFER_REQUESTS` | RO | Returns the optimal number of inference requests to be used by the application. Depends on the platform version and on ov::hint::performance_mode. Please see the table below. | `N/A` | `N/A` |
| `ov::range_for_async_infer_requests`/</br>`RANGE_FOR_ASYNC_INFER_REQUESTS` | RO | Returns a tuple (bottom, top, step). </br> Not used by the NPU plugin. | `N/A` | `N/A` |
| `ov::range_for_streams`/</br>`RANGE_FOR_STREAMS` | RO | Returns a tuple (bottom, top).</br> Not used by the NPU plugin. | `N/A`| `N/A` |
| `ov::enable_profiling`/</br>`PERF_COUNT` | RW | Enables or disables performance counters. | `YES`/ `NO` | `NO` |
| `ov::hint::performance_mode`/</br>`PERFORMANCE_HINT` | RW | Sets the performance profile used to determine default values of DPUs/DMAs/NIREQs.</br>Default values for each profile are documented below. | `THROUGHPUT`/</br>`LATENCY`/</br>`UNDEFINED` | `UNDEFINED` |
| `ov::hint::num_requests`/</br>`PERFORMANCE_HINT_NUM_REQUESTS` | RW | Sets the number of outstanding inference requests. | `[0-]` | `1` |
| `ov::hint::model_priority`/</br>`MODEL_PRIORITY` | RW | Assigns a priority for the model execution. | `LOW`/</br>`MEDIUM`/</br>`HIGH` | `MEDIUM` |
| `ov::hint::enable_cpu_pinning`/</br>`ENABLE_CPU_PINNING` | RW | Allows CPU threads pinning during inference. | `YES`/ `NO` /</br>`NO` 
| `ov::log::level`/</br>`LOG_LEVEL` | RW |  Sets the log level for NPU Plugin. An environment variable is also made available to expose logs from early initialization phase: OV_NPU_LOG_LEVEL. | `LOG_NONE`/</br>`LOG_ERROR`/</br>`LOG_WARNING`/</br>`LOG_INFO`/</br>`LOG_DEBUG`/</br>`LOG_TRACE` |  `LOG_NONE` |
| `ov::cache_dir`/</br>`CACHE_DIR` | RW | Folder path to be used by the OpenVINO cache. | `N/A` | empty |
| `ov::available_devices`/</br>`AVAILABLE_DEVICES` | RO | Returns the list of enumerated NPU devices. </br> NPU plugin does not currently support multiple devices. | `N/A`| `N/A` |
| `ov::device::id`/</br>`DEVICE_ID` | RW | Device identifier. Empty means auto detection. | empty/</br> `3720`/</br> `4000` | empty |
| `ov::device::uuid`/</br> | RO | Returns the Universal Unique ID of the NPU device. | `N/A`| `N/A` |
| `ov::device::architecture`/</br>`DEVICE_ARCHITECTURE` | RO | Returns the platform information. | `N/A`| `N/A` |
| `ov::device::full_name`/</br>`FULL_DEVICE_NAME` | RO | Returns the full name of the NPU device. | `N/A`| `N/A` |
| `ov::internal::exclusive_async_requests`/</br>`EXCLUSIVE_ASYNC_REQUESTS` | RW | Allows to use exclusive task executor for asynchronous infer requests. | `YES`/ `NO`| `NO` |
| `ov::device::type`/</br>`DEVICE_TYPE` | RO | Returns the type of device, discrete or integrated. | `DISCRETE` /</br>`INTEGRATED` | `N/A` |
| `ov::device::gops`/</br>`DEVICE_GOPS` | RO | Returns the Giga OPS per second count (GFLOPS or GIOPS) for a set of precisions supported by specified device. | `N/A`| `N/A` |
| `ov::device::pci_info`/</br>`DEVICE_PCI_INFO` | RO | Returns the PCI bus information of device. See PCIInfo struct definition for details | `N/A`| `N/A` |
| `ov::intel_npu::device_alloc_mem_size`/</br>`NPU_DEVICE_ALLOC_MEM_SIZE` | RO | Size of already allocated NPU DDR memory | `N/A` | `N/A` |
| `ov::intel_npu::device_total_mem_size`/</br>`NPU_DEVICE_TOTAL_MEM_SIZE` | RO | Size of available NPU DDR memory | `N/A` | `N/A` |
| `ov::intel_npu::driver_version`/</br>`NPU_DRIVER_VERSION` | RO | NPU driver version. | `N/A` | `N/A` |
| `ov::intel_npu::compilation_mode_params`/</br>`NPU_COMPILATION_MODE_PARAMS` | RW | Set various parameters supported by the NPU compiler. (See bellow) | `<std::string>`| `N/A` |
| `ov::intel_npu::turbo`/</br>`NPU_TURBO` | RW | Set Turbo mode on/off | `YES`/ `NO`| `NO` |
| `ov::intel_npu::tiles`/</br>`NPU_TILES` | RW | Sets the number of npu tiles to compile the model for | `[0-]` | `-1` |
| `ov::intel_npu::max_tiles`/</br>`NPU_MAX_TILES` | RW | Maximum number of tiles supported by the device we compile for. Can be set for offline compilation. If not set, it will be populated by driver.| `[0-]` | `[1-6] depends on npu platform` |
| `ov::intel_npu::bypass_umd_caching`/</br>`NPU_BYPASS_UMD_CACHING` | RW | Bypass the caching of compiled models in UMD. | `YES`/ `NO`| `NO` |

&nbsp;
### Performance Hint: Default Number of DPU Groups / DMA Engines

The following table shows the default values for the number of DPU Groups (Tiles) and DMA Engines selected by the plugin based on the performance mode (THROUGHPUT/LATENCY) and based on the platform:

| Performance hint | NPU Platform        | Number of DPU Groups | Number of DMA Engines           |
| :---             | :---                | :---                 | :---                            |
| THROUGHPUT       | 3720                | 2 (all of them)      | 2 (all of them)                 |
| THROUGHPUT       | 4000                | 2 (out of 5/6)       | 2 (all of them)                 |
| LATENCY          | 3720                | 2 (all of them)      | 2 (all of them)                 |
| LATENCY          | 4000                | 4 (out of 5/6)       | 2 (all of them)                 |

&nbsp;
### Performance Hint: Optimal Number of Inference Requests

The following table shows the optimal number of inference requests returned by the plugin based on the performance mode (THROUGHPUT/LATENCY) and based on the platform:

| NPU Platform        | Nr. of Inference Requests </br> THROUGHPUT  | Nr. of Inference Requests </br> LATENCY |
| :---                | :---                                        | :---                                    |
| 3720                | 4                                           | 1                                       |
| 4000                | 8                                           | 1                                       |

&nbsp;
### Compilation mode parameters
``ov::intel_npu::compilation_mode_params`` is an NPU-specific property that allows to control model compilation for NPU.
Note: The functionality is in experimental stage currently, can be a subject for deprecation and may be replaced with generic OV API in future OV releases.

Following configuration options are supported:

#### optimization-level
Defines a preset of optimization passes to be applied during compilation. Supported values:

| Value  | Description                                                    |
| :---   | :---                                                           | 
| 0      | Reduced subset of optimization passes. Smaller compile time.   |
| 1      | Default. Balanced performance/compile time.                    |
| 2      | Prioritize performance over compile time that may be an issue. |

#### performance-hint-override
An extension for LATENCY mode being specified using ``ov::hint::performance_mode``
Has no effect for other ``ov::hint::PerformanceMode`` hints.

Supported values:

| Value      | Description                                          |
| :---       | :---                                                 | 
| efficiency | Default. Balanced performance and power consumption. |
| latency    | Prioritize performance over power efficiency.        |

#### Usage example:
```
    map<str, str> config = {ov::intel_npu::compilation_mode_params.name(), ov::Any("optimization-level=1 performance-hint-override=latency")};
    compile_model(model, config);
```

### ov::intel_npu::max_tiles and ov::intel_npu::tiles

The max_tiles property is read-write to enable compiling models off-device.  
When on NPU, max_tiles will return the number of tiles the device has.  
Setting the number of tiles to compile for (via intel_npu::tiles), when on device,
must be preceded by reading intel_npu::max_tiles first, to make sure that  
``ov::intel_npu::tiles`` <= ``ov::intel_npu::max_tiles``  
to avoid exceptions from the compiler.

   Note that ``ov::intel_npu::tiles`` overrides the default number of tiles selected by the compiler based on performance hints
   (``ov::hint::performance_mode``).
   Any tile number other than 1 may be a problem for cross platform compatibility,
   if not tested explicitly versus the max_tiles value.
&nbsp;
## Stateful models

Key ingredients to support stateful models which distinguish them from other models are:
* Implementing ReadValue and Assign operators
* Implementing Query State API (to give user an API to reset/get/set states)
* Implementing initialization for a state

More details on how OpenVINO supports stateful models can be found here: [Stateful models](https://docs.openvino.ai/2022.3/openvino_docs_OV_UG_network_state_intro.html).

The current implementation of state variables inside the NPU plugin is illustrated by the below diagram:

![High Level Design](./docs/img/stateful_models.png)

Notes on the implementation:
* One network with N inputs + K state variables + M outputs will be converted by the compiler into a model with (N+K) inputs and (M+K) outputs. State variables are represented by a set of input/output nodes. This is currently needed because the underlying software stack (driver and runtime) does not support state variables. 
* The input and output nodes corresponding to state variables have different buffers allocated through the Level Zero API.
* The content of the output buffer is copied back into the input buffer by the plugin through the use of an intermediate state buffer:
    * NPU Plugin allocates and maintains one additional state buffer which is exposed through the GetState/SetState API
    * The actual level zero input buffer for the state is updated when the inference is triggered with the content of the state buffer
    * The state buffer is updated once the inference is completed with the content of the output level zero buffer

The implementation of state variables in the NPU plugin will be improved for upcoming releases.

&nbsp;
## Dynamic shapes
Dynamic shapes are not supported by the NPU plugin yet.
