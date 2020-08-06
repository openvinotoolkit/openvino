# Synchronous Inference Request {#infer_request}

`InferRequest` class functionality:
- Allocate input and output blobs needed for a hardware-dependent network inference.
- Define functions for inference process stages (for example, `preprocess`, `upload`, `infer`, `download`, `postprocess`). These functions can later be used to define an execution pipeline during [Asynchronous Inference Request](@ref async_infer_request) implementation.
- Call inference stages one by one synchronously.

`InferRequest` Class
------------------------

Inference Engine Plugin API provides the helper InferenceEngine::InferRequestInternal class recommended 
to use as a base class for a synchronous inference request implementation. Based of that, a declaration 
of a synchronous request class can look as follows: 

@snippet src/template_infer_request.hpp infer_request:header

#### Class Fields

The example class has several fields:

- `_executableNetwork` - reference to an executable network instance. From this reference, an inference request instance can take a task executor, use counter for a number of created inference requests, and so on.
- `_profilingTask` - array of the `std::array<InferenceEngine::ProfilingTask, numOfStages>` type. Defines names for pipeline stages. Used to profile an inference pipeline execution with the Intel® instrumentation and tracing technology (ITT).
- `_inputsNCHW` - input blob map
- `_outputsNCHW` - output blob map
- Several double values to hold an execution time for pipeline stages.

### `InferRequest` Constructor

The constructor initializes helper fields and calls methods which allocate blobs:

@snippet src/template_infer_request.cpp infer_request:ctor

The implementation of function allocating device buffers is fully device-specific and not provided in the guide. 
The implementation of function allocating host buffers assumes that the `Template` device works 
natively only with the InferenceEngine::NCHW input and output layout, while the user can specify the InferenceEngine::NHWC as a layout 
of InferenceEngine::CNNNetwork inputs and outputs and set InferenceEngine::NHWC blobs via the InferenceEngine::InferRequest::SetBlob method.

> **NOTE**: Call InferenceEngine::CNNNetwork::getInputsInfo and InferenceEngine::CNNNetwork::getOutputsInfo to specify both layout and precision of blobs, which you can set with InferenceEngine::InferRequest::SetBlob and get with InferenceEngine::InferRequest::GetBlob. A plugin uses these hints to determine its internal layouts and precisions for input and output blobs if needed. 

### `~InferRequest` Destructor

Decrements a number of created inference requests: 

@snippet src/template_infer_request.cpp infer_request:dtor

### `InferImpl()`

**Implementation details:** Base InferRequestInternal class implements the public InferenceEngine::InferRequestInternal::Infer method as following:
- Checks blobs set by users
- Calls the `InferImpl` method defined in a derived class to call actual pipeline stages synchronously

@snippet src/template_infer_request.cpp infer_request:infer_impl

Below is the code of the the `inferPreprocess` method to demonstrate Inference Engine common preprocessing step handling:

@snippet src/template_infer_request.cpp infer_request:infer_preprocess

**Details:**
* `InferImpl` must call the InferenceEngine::InferRequestInternal::execDataPreprocessing function, which executes common Inference Engine preprocessing step (for example, applies resize or color conversion operations) if it is set by the user. The output dimensions, layout and precision matches the input information set via InferenceEngine::CNNNetwork::getInputsInfo.
* To handle both InferenceEngine::NCHW and InferenceEngine::NHWC input layouts, the `TemplateInferRequest` class has the `_inputsNCHW` field, which holds blobs in the InferenceEngine::NCHW layout. During Inference Request execution, `InferImpl` copies from the input InferenceEngine::NHWC layout to `_inputsNCHW` if needed.
* The next logic of `InferImpl` works with `_inputsNCHW`.

### `GetPerformanceCounts()`

The method sets performance counters which were measured during pipeline stages execution:

@snippet src/template_infer_request.cpp infer_request:get_performance_counts

The next step in the plugin library implementation is the [Asynchronous Inference Request](@ref async_infer_request) class.
