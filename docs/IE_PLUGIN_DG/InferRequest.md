# Synchronous Inference Request {#openvino_docs_ie_plugin_dg_infer_request}

@sphinxdirective

``InferRequest`` class functionality:

* Allocate input and output blobs needed for a backend-dependent network inference.
* Define functions for inference process stages (for example, ``preprocess``, ``upload``, ``infer``, ``download``, ``postprocess``). These functions can later be used to define an execution pipeline during :doc:`Asynchronous Inference Request <openvino_docs_ie_plugin_dg_async_infer_request>` implementation.
* Call inference stages one by one synchronously.

`InferRequest` Class
####################

Inference Engine Plugin API provides the helper :ref:`InferenceEngine::IInferRequestInternal <doxid-class_inference_engine_1_1_i_infer_request_internal>` class recommended 
to use as a base class for a synchronous inference request implementation. Based of that, a declaration 
of a synchronous request class can look as follows: 

.. doxygensnippet:: src/sync_infer_request.hpp
   :language: cpp
   :fragment: [infer_request:header]

Class Fields
++++++++++++

The example class has several fields:

* ``_executableNetwork`` - reference to an executable network instance. From this reference, an inference request instance can take a task executor, use counter for a number of created inference requests, and so on.
* ``_profilingTask` - array of the ``std::array<InferenceEngine::ProfilingTask, numOfStages>`` type. Defines names for pipeline stages. Used to profile an inference pipeline execution with the Intel® instrumentation and tracing technology (ITT).
* ``_durations`` - array of durations of each pipeline stage.
* ``_networkInputBlobs`` - input blob map.
* ``_networkOutputBlobs`` - output blob map.
* ``_parameters`` - ``:ref:`ngraph::Function <doxid-classngraph_1a14d7fe7c605267b52c145579e12d2a5f>``` parameter operations.
* ``_results`` - ``:ref:`ngraph::Function <doxid-classngraph_1a14d7fe7c605267b52c145579e12d2a5f>``` result operations.
* backend specific fields:

	* ``_inputTensors`` - inputs tensors which wrap ``_networkInputBlobs`` blobs. They are used as inputs to backend `_executable` computational graph.
	* ``_outputTensors`` - output tensors which wrap ``_networkOutputBlobs`` blobs. They are used as outputs from backend `_executable`` computational graph.
	* ``_executable`` - an executable object / backend computational graph.

``InferRequest`` Constructor
############################

The constructor initializes helper fields and calls methods which allocate blobs:

.. doxygensnippet:: src/sync_infer_request.cpp
   :language: cpp
   :fragment: [infer_request:ctor]

.. note:: 

	Call :ref:`InferenceEngine::CNNNetwork::getInputsInfo <doxid-class_inference_engine_1_1_c_n_n_network_1a76de2a6101fe8276f56b0dc0f99c7ff7>` and :ref:`InferenceEngine::CNNNetwork::getOutputsInfo <doxid-class_inference_engine_1_1_c_n_n_network_1af8a6200f549b15a895e2cfefd304a9c2>` to specify both layout and precision of blobs, which you can set with :ref:`InferenceEngine::InferRequest::SetBlob <doxid-class_inference_engine_1_1_infer_request_1a27fb179e3bae652d76076965fd2a5653>` and get with :ref:`InferenceEngine::InferRequest::GetBlob <doxid-class_inference_engine_1_1_infer_request_1a9601a4cda3f309181af34feedf1b914c>`. A plugin uses these hints to determine its internal layouts and precisions for input and output blobs if needed. 

``~InferRequest`` Destructor
############################

Decrements a number of created inference requests: 

.. doxygensnippet:: src/sync_infer_request.cpp
   :language: cpp
   :fragment: [infer_request:dtor]

``InferImpl()``
###############

**Implementation details:** Base IInferRequestInternal class implements the public :ref:`InferenceEngine::IInferRequestInternal::Infer <doxid-class_inference_engine_1_1_i_infer_request_internal_1afb61e1de4ffb9927431085a91a40f352>` method as following:

* Checks blobs set by users
* Calls the ``InferImpl`` method defined in a derived class to call actual pipeline stages synchronously

.. doxygensnippet:: src/sync_infer_request.cpp
   :language: cpp
   :fragment: [infer_request:infer_impl]

1. `inferPreprocess`
++++++++++++++++++++

Below is the code of the ``inferPreprocess`` method to demonstrate Inference Engine common preprocessing step handling:

.. doxygensnippet:: src/sync_infer_request.cpp
   :language: cpp
   :fragment: [infer_request:infer_preprocess]

**Details:**

* ``InferImpl`` must call the :ref:`InferenceEngine::IInferRequestInternal::execDataPreprocessing <doxid-class_inference_engine_1_1_i_infer_request_internal_1a1ca532a389eb95c12ff9c8d463e93268>` function, which executes common Inference Engine preprocessing step (for example, applies resize or color conversion operations) if it is set by the user. The output dimensions, layout and precision matches the input information set via :ref:`InferenceEngine::CNNNetwork::getInputsInfo <doxid-class_inference_engine_1_1_c_n_n_network_1a76de2a6101fe8276f56b0dc0f99c7ff7>`.
* If ``inputBlob`` passed by user differs in terms of precisions from precision expected by plugin, ``blobCopy`` is performed which does actual precision conversion.

2. `startPipeline`
++++++++++++++++++

Executes a pipeline synchronously using ``_executable`` object:

.. doxygensnippet:: src/sync_infer_request.cpp
   :language: cpp
   :fragment: [infer_request:start_pipeline]

3. `inferPostprocess`
+++++++++++++++++++++

Converts output blobs if precisions of backend output blobs and blobs passed by user are different:

.. doxygensnippet:: src/sync_infer_request.cpp
   :language: cpp
   :fragment: [infer_request:infer_postprocess]

`GetPerformanceCounts()`
########################

The method sets performance counters which were measured during pipeline stages execution:

.. doxygensnippet:: src/sync_infer_request.cpp
   :language: cpp
   :fragment: [infer_request:get_performance_counts]

The next step in the plugin library implementation is the :doc:`Asynchronous Inference Request <openvino_docs_ie_plugin_dg_async_infer_request>` class.

@endsphinxdirective
