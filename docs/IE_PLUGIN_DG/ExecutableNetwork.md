# Executable Network {#openvino_docs_ie_plugin_dg_executable_network}

`ExecutableNetwork` class functionality:
- Compile an InferenceEngine::ICNNNetwork instance to a backend specific graph representation
- Create an arbitrary number of `InferRequest` objects
- Hold some common resources shared between different instances of `InferRequest`. For example:
	- InferenceEngine::IExecutableNetworkInternal::_taskExecutor task executor to implement asynchronous execution
	- InferenceEngine::IExecutableNetworkInternal::_callbackExecutor task executor to run an asynchronous inference request callback in a separate thread

`ExecutableNetwork` Class
------------------------

Inference Engine Plugin API provides the helper InferenceEngine::ExecutableNetworkThreadSafeDefault class recommended to use as a base class for an executable network. Based on that, a declaration of an executable network class can look as follows: 

@snippet src/template_executable_network.hpp executable_network:header

#### Class Fields

The example class has several fields:

- `_requestId` - Tracks a number of created inference requests, which is used to distinguish different inference requests during profiling via the Intel® Instrumentation and Tracing Technology (ITT) library.
- `_cfg` - Defines a configuration an executable network was compiled with.
- `_plugin` - Refers to a plugin instance.
- `_function` - Keeps a reference to transformed `ngraph::Function` which is used in ngraph reference backend computations. Note, in case of other backends with backend specific graph representation `_function` has different type and represents backend specific graph or just a set of computational kernels to perform an inference.
- `_inputIndex` - maps a name of input with its index among all network inputs.
- `_outputIndex` - maps a name of output with its index among all network outputs.

### `ExecutableNetwork` Constructor with `ICNNNetwork`

This constructor accepts a generic representation of a neural network as an InferenceEngine::ICNNNetwork reference and is compiled into a backend specific device graph:

@snippet src/template_executable_network.cpp executable_network:ctor_cnnnetwork

The implementation `CompileNetwork` is fully device-specific.

### `CompileNetwork()`

The function accepts a const shared pointer to `ngraph::Function` object and performs the following steps:

1. Applies ngraph passes using `TransformNetwork` function, which defines plugin-specific conversion pipeline. To support low precision inference, the pipeline can include Low Precision Transformations. These transformations are usually hardware specific. You can find how to use and configure Low Precisions Transformations in [Low Precision Transformations](@ref openvino_docs_OV_UG_lpt) guide.
2. Maps the transformed graph to a backend specific graph representation (for example, to MKLDNN graph for Intel CPU).
3. Allocates and fills memory for graph weights, backend specific memory handles and so on.

@snippet src/template_executable_network.cpp executable_network:map_graph

> **NOTE**: After all these steps, the backend specific graph is ready to create inference requests and perform inference.

### `ExecutableNetwork` Constructor Importing from Stream

This constructor creates a backend specific graph by importing from a stream object:

> **NOTE**: The export of backend specific graph is done in the `Export` method, and data formats must be the same for both import and export.

@snippet src/template_executable_network.cpp executable_network:ctor_import_stream

### `Export()`

The implementation of the method should write all data to the `model` stream, which is required to import a backend specific graph later in the `Plugin::Import` method:

@snippet src/template_executable_network.cpp executable_network:export

### `CreateInferRequest()`

The method creates an asynchronous inference request and returns it. While the public Inference Engine API has a single interface for inference request, which can be executed in synchronous and asynchronous modes, a plugin library implementation has two separate classes:

- [Synchronous inference request](@ref openvino_docs_ie_plugin_dg_infer_request), which defines pipeline stages and runs them synchronously in the `Infer` method.
- [Asynchronous inference request](@ref openvino_docs_ie_plugin_dg_async_infer_request), which is a wrapper for a synchronous inference request and can run a pipeline asynchronously. Depending on a device pipeline structure, it can has one or several stages:
   - For single-stage pipelines, there is no need to define this method and create a class derived from InferenceEngine::AsyncInferRequestThreadSafeDefault. For single stage pipelines, a default implementation of this method creates InferenceEngine::AsyncInferRequestThreadSafeDefault wrapping a synchronous inference request and runs it asynchronously in the `_taskExecutor` executor.
   - For pipelines with multiple stages, such as performing some preprocessing on host, uploading input data to a device, running inference on a device, or downloading and postprocessing output data, schedule stages on several task executors to achieve better device use and performance. You can do it by creating a sufficient number of inference requests running in parallel. In this case, device stages of different inference requests are overlapped with preprocessing and postprocessing stage giving better performance.
   > **IMPORTANT**: It is up to you to decide how many task executors you need to optimally execute a device pipeline.

@snippet src/template_executable_network.cpp executable_network:create_infer_request

### `CreateInferRequestImpl()`

This is a helper method used by `CreateInferRequest` to create a [synchronous inference request](@ref openvino_docs_ie_plugin_dg_infer_request), which is later wrapped with the asynchronous inference request class:

@snippet src/template_executable_network.cpp executable_network:create_infer_request_impl

### `GetMetric()`

Returns a metric value for a metric with the name `name`.  A metric is a static type of information about an executable network. Examples of metrics:

- EXEC_NETWORK_METRIC_KEY(NETWORK_NAME) - name of an executable network
- EXEC_NETWORK_METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS) - heuristic to denote an optimal (or at least sub-optimal) number of inference requests needed to run asynchronously to use the current device fully
- Any other executable network metric specific for a particular device. Such metrics and possible values must be declared in a plugin configuration public header, for example, `template/template_config.hpp`

@snippet src/template_executable_network.cpp executable_network:get_metric

The IE_SET_METRIC_RETURN helper macro sets metric value and checks that the actual metric type matches a type of the specified value.

### `GetConfig()`

Returns a current value for a configuration key with the name `name`. The method extracts configuration values an executable network is compiled with.

@snippet src/template_executable_network.cpp executable_network:get_config

This function is the only way to get configuration values when a network is imported and compiled by other developers and tools (for example, the [Compile tool](../_inference_engine_tools_compile_tool_README.html)).

The next step in plugin library implementation is the [Synchronous Inference Request](@ref openvino_docs_ie_plugin_dg_infer_request) class.
