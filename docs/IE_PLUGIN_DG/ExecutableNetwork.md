# Executable Network {#executable_network}

`ExecutableNetwork` class functionality:
- Compile an InferenceEngine::ICNNNetwork instance to a hardware-specific graph representation
- Create an arbitrary number of `InferRequest` objects
- Hold some common resources shared between different instances of `InferRequest`. For example:
	- InferenceEngine::ExecutableNetworkInternal::_taskExecutor task executor to implement asynchronous execution
	- InferenceEngine::ExecutableNetworkInternal::_callbackExecutor task executor to run an asynchronous inference request callback in a separate thread

`ExecutableNetwork` Class
------------------------

Inference Engine Plugin API provides the helper InferenceEngine::ExecutableNetworkThreadSafeDefault class recommended to use as a base class for an executable network. Based on that, a declaration of an executable network class can look as follows: 

@snippet src/template_executable_network.hpp executable_network:header

#### Class Fields

The example class has several fields:

- `_requestId` - Tracks a number of created inference requests, which is used to distinguish different inference requests during profiling via the Intel® Instrumentation and Tracing Technology (ITT) library.
- `_name` - Provides a network name.
- `_cfg` - Defines a configuration an executable network was compiled with.
- `_plugin` - Refers to a plugin instance.

### `ExecutableNetwork` Constructor with `ICNNNetwork`

This constructor accepts a generic representation of a neural network as an InferenceEngine::ICNNNetwork reference and is compiled into a hardware-specific device graph:

@snippet src/template_executable_network.cpp executable_network:ctor_cnnnetwork

The implementation `CompileGraph` is fully device-specific.

### `CompileGraph()`

The function accepts a const shared pointer to `const ngraph::Function` object and performs the following steps:

1. Deep copies a const object to a local object, which can later be modified.
2. Applies common and plugin-specific transformations on a copied graph to make the graph more friendly to hardware operations. For details how to write custom plugin-specific transformation, please, refer to [Writing ngraph transformations](@ref new_ngraph_transformation) guide.
3. Maps the transformed graph to a plugin-specific graph representation (for example, to MKLDNN graph for CPU). See details topics about network representation:
    * [Intermediate Representation and Operation Sets](../_docs_MO_DG_IR_and_opsets.html)
    * [Quantized networks](@ref quantized_networks).
4. Allocates and fills memory for graph weights.

@snippet src/template_executable_network.cpp executable_network:compile_graph

> **NOTE**: After all these steps, the hardware-specific graph is ready to create inference requests and perform inference.

### `ExecutableNetwork` Constructor Importing from Stream

This constructor creates a hardware-specific graph by importing from a stream object:

> **NOTE**: The export of hardware-specific graph is done in the `ExportImpl` method, and data formats must be the same for both import and export.

@snippet src/template_executable_network.cpp executable_network:ctor_import_stream

### `ExportImpl()`

**Implementation details:**   
Base InferenceEngine::ExecutableNetworkThreadSafeDefault class implements the public InferenceEngine::ExecutableNetworkThreadSafeDefault::Export method as following:
- Writes `_plugin->GetName()` to the `model` stream.
- Calls the `ExportImpl` method defined in a derived class to dump a hardware-specific graph.

The implementation of the method should write all data to the `model` stream, which is required to import a hardware-specific graph later in the `Plugin::Import` method:

@snippet src/template_executable_network.cpp executable_network:export_impl

### `CreateInferRequest()`

The method creates an asynchronous inference request and returns it. While the public Inference Engine API has a single interface for inference request, which can be executed in synchronous and asynchronous modes, a plugin library implementation has two separate classes:

- [Synchronous inference request](@ref infer_request), which defines pipeline stages and runs them synchronously in the `Infer` method.
- [Asynchronous inference request](@ref async_infer_request), which is a wrapper for a synchronous inference request and can run a pipeline asynchronously. Depending on a device pipeline structure, it can has one or several stages:
   - For single-stage pipelines, there is no need to define this method and create a class derived from InferenceEngine::AsyncInferRequestThreadSafeDefault. For single stage pipelines, a default implementation of this method creates InferenceEngine::AsyncInferRequestThreadSafeDefault wrapping a synchronous inference request and runs it asynchronously in the `_taskExecutor` executor.
   - For pipelines with multiple stages, such as performing some preprocessing on host, uploading input data to a device, running inference on a device, or downloading and postprocessing output data, schedule stages on several task executors to achieve better device use and performance. You can do it by creating a sufficient number of inference requests running in parallel. In this case, device stages of different inference requests are overlapped with preprocessing and postprocessing stage giving better performance.

   > **IMPORTANT**: It is up to you to decide how many task executors you need to optimally execute a device pipeline.

@snippet src/template_executable_network.cpp executable_network:create_infer_request

### `CreateInferRequestImpl()`

This is a helper method used by `CreateInferRequest` to create a [synchronous inference request](@ref infer_request), which is later wrapped with the asynchronous inference request class:

@snippet src/template_executable_network.cpp executable_network:create_infer_request_impl

### `GetMetric()`

Returns a metric value for a metric with the name `name`.  A metric is a static type of information about an executable network. Examples of metrics:

- EXEC_NETWORK_METRIC_KEY(NETWORK_NAME) - name of an executable network
- EXEC_NETWORK_METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS) - heuristic to denote an optimal (or at least sub-optimal) number of inference requests needed to run asynchronously to use the current device fully
- Any other executable network metric specific for a particular device. Such metrics and possible values must be declared in a plugin configuration public header, for example, `template/template_config.hpp`

@snippet src/template_executable_network.cpp executable_network:get_metric

The IE_SET_METRIC helper macro sets metric value and checks that the actual metric type matches a type of the specified value.

### `GetConfig()`

Returns a current value for a configuration key with the name `name`. The method extracts configuration values an executable network is compiled with.

@snippet src/template_executable_network.cpp executable_network:get_config

This function is the only way to get configuration values when a network is imported and compiled by other developers and tools (for example, the [Compile tool](../_inference_engine_tools_compile_tool_README.html)).

The next step in plugin library implementation is the [Synchronous Inference Request](@ref infer_request) class.
