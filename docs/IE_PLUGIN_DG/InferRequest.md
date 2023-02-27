# Synchronous Inference Request {#openvino_docs_ie_plugin_dg_infer_request}

`InferRequest` class functionality:
- Allocate input and output blobs needed for a backend-dependent network inference.
- Define functions for inference process stages (for example, `preprocess`, `upload`, `infer`, `download`, `postprocess`). These functions can later be used to define an execution pipeline during [Asynchronous Inference Request](@ref openvino_docs_ie_plugin_dg_async_infer_request) implementation.
- Call inference stages one by one synchronously.

`InferRequest` Class
------------------------

Inference Engine Plugin API provides the helper InferenceEngine::IInferRequestInternal class recommended 
to use as a base class for a synchronous inference request implementation. Based of that, a declaration 
of a synchronous request class can look as follows: 

@snippet src/sync_infer_request.hpp infer_request:header

#### Class Fields

The example class has several fields:

- `_executableNetwork` - reference to an executable network instance. From this reference, an inference request instance can take a task executor, use counter for a number of created inference requests, and so on.
- `_profilingTask` - array of the `std::array<InferenceEngine::ProfilingTask, numOfStages>` type. Defines names for pipeline stages. Used to profile an inference pipeline execution with the Intel® instrumentation and tracing technology (ITT).
- `_durations` - array of durations of each pipeline stage.
- `_networkInputBlobs` - input blob map.
- `_networkOutputBlobs` - output blob map.
- `_parameters` - `ngraph::Function` parameter operations.
- `_results` - `ngraph::Function` result operations.
- backend specific fields:
	- `_inputTensors` - inputs tensors which wrap `_networkInputBlobs` blobs. They are used as inputs to backend `_executable` computational graph.
	- `_outputTensors` - output tensors which wrap `_networkOutputBlobs` blobs. They are used as outputs from backend `_executable` computational graph.
	- `_executable` - an executable object / backend computational graph.

### `InferRequest` Constructor

The constructor initializes helper fields and calls methods which allocate blobs:

@snippet src/sync_infer_request.cpp infer_request:ctor

> **NOTE**: Call InferenceEngine::CNNNetwork::getInputsInfo and InferenceEngine::CNNNetwork::getOutputsInfo to specify both layout and precision of blobs, which you can set with InferenceEngine::InferRequest::SetBlob and get with InferenceEngine::InferRequest::GetBlob. A plugin uses these hints to determine its internal layouts and precisions for input and output blobs if needed. 

### `~InferRequest` Destructor

Decrements a number of created inference requests: 

@snippet src/sync_infer_request.cpp infer_request:dtor

### `InferImpl()`

**Implementation details:** Base IInferRequestInternal class implements the public InferenceEngine::IInferRequestInternal::Infer method as following:
- Checks blobs set by users
- Calls the `InferImpl` method defined in a derived class to call actual pipeline stages synchronously

@snippet src/sync_infer_request.cpp infer_request:infer_impl

#### 1. `inferPreprocess`

Below is the code of the `inferPreprocess` method to demonstrate Inference Engine common preprocessing step handling:

@snippet src/sync_infer_request.cpp infer_request:infer_preprocess

**Details:**
* `InferImpl` must call the InferenceEngine::IInferRequestInternal::execDataPreprocessing function, which executes common Inference Engine preprocessing step (for example, applies resize or color conversion operations) if it is set by the user. The output dimensions, layout and precision matches the input information set via InferenceEngine::CNNNetwork::getInputsInfo.
* If `inputBlob` passed by user differs in terms of precisions from precision expected by plugin, `blobCopy` is performed which does actual precision conversion.

#### 2. `startPipeline`

Executes a pipeline synchronously using `_executable` object:

@snippet src/sync_infer_request.cpp infer_request:start_pipeline

#### 3. `inferPostprocess`

Converts output blobs if precisions of backend output blobs and blobs passed by user are different:

@snippet src/sync_infer_request.cpp infer_request:infer_postprocess

### `GetPerformanceCounts()`

The method sets performance counters which were measured during pipeline stages execution:

@snippet src/sync_infer_request.cpp infer_request:get_performance_counts

The next step in the plugin library implementation is the [Asynchronous Inference Request](@ref openvino_docs_ie_plugin_dg_async_infer_request) class.
