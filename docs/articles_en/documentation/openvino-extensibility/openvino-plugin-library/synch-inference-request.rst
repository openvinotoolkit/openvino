Synchronous Inference Request
=============================


.. meta::
   :description: Use the ov::ISyncInferRequest interface as the base class to implement a synchronous inference request in OpenVINO.


``InferRequest`` class functionality:

* Allocate input and output tensors needed for a backend-dependent network inference.
* Define functions for inference process stages (for example, ``preprocess``, ``upload``, ``infer``, ``download``, ``postprocess``). These functions can later be used to define an execution pipeline during :doc:`Asynchronous Inference Request <asynch-inference-request>` implementation.
* Call inference stages one by one synchronously.

InferRequest Class
##################

OpenVINO Plugin API provides the interface ov::ISyncInferRequest which should be
used as a base class for a synchronous inference request implementation. Based of that, a declaration
of a synchronous request class can look as follows:

.. doxygensnippet:: src/plugins/template/src/sync_infer_request.hpp
   :language: cpp
   :fragment: [infer_request:header]

Class Fields
++++++++++++

The example class has several fields:

* ``m_profiling_task`` - array of the ``std::array<openvino::itt::handle_t, numOfStages>`` type. Defines names for pipeline stages. Used to profile an inference pipeline execution with the Intel® instrumentation and tracing technology (ITT).

* ``m_durations`` - array of durations of each pipeline stage.

* backend-specific fields:

  * ``m_backend_input_tensors`` - input backend tensors.
  * ``m_backend_output_tensors`` - output backend tensors.
  * ``m_executable`` - an executable object / backend computational graph.
  * ``m_eval_context`` - an evaluation context to save backend states after the inference.
  * ``m_variable_states`` - a vector of variable states.

InferRequest Constructor
++++++++++++++++++++++++

The constructor initializes helper fields and calls methods which allocate tensors:

.. doxygensnippet:: src/plugins/template/src/sync_infer_request.cpp
   :language: cpp
   :fragment: [infer_request:ctor]

.. note::

   Use inputs/outputs information from the compiled model to understand shape and element type of tensors, which you can set with ov::InferRequest::set_tensor and get with ov::InferRequest::get_tensor. A plugin uses these hints to determine its internal layouts and element types for input and output tensors if needed.

~InferRequest Destructor
++++++++++++++++++++++++

Destructor can contain plugin specific logic to finish and destroy infer request.

.. doxygensnippet:: src/plugins/template/src/sync_infer_request.cpp
   :language: cpp
   :fragment: [infer_request:dtor]

set_tensors_impl()
+++++++++++++++++++

The method allows to set batched tensors in case if the plugin supports it.

.. doxygensnippet:: src/plugins/template/src/sync_infer_request.cpp
   :language: cpp
   :fragment: [infer_request:set_tensors_impl]

query_state()
+++++++++++++

The method returns variable states from the model.

.. doxygensnippet:: src/plugins/template/src/sync_infer_request.cpp
   :language: cpp
   :fragment: [infer_request:query_state]

infer()
+++++++

The method calls actual pipeline stages synchronously. Inside the method plugin should check input/output tensors, move external tensors to backend and run the inference.

.. doxygensnippet:: src/plugins/template/src/sync_infer_request.cpp
   :language: cpp
   :fragment: [infer_request:infer]

1. infer_preprocess()
----------------------

Below is the code of the ``infer_preprocess()`` method. The method checks user input/output tensors and demonstrates conversion from user tensor to backend specific representation:

.. doxygensnippet:: src/plugins/template/src/sync_infer_request.cpp
   :language: cpp
   :fragment: [infer_request:infer_preprocess]

2. start_pipeline()
--------------------

Executes a pipeline synchronously using ``m_executable`` object:

.. doxygensnippet:: src/plugins/template/src/sync_infer_request.cpp
   :language: cpp
   :fragment: [infer_request:start_pipeline]

3. wait_pipeline()
--------------------

Waits a pipeline in case of plugin asynchronous execution:

.. doxygensnippet:: src/plugins/template/src/sync_infer_request.cpp
   :language: cpp
   :fragment: [infer_request:wait_pipeline]

4. infer_postprocess()
----------------------

Converts backend specific tensors to tensors passed by user:

.. doxygensnippet:: src/plugins/template/src/sync_infer_request.cpp
   :language: cpp
   :fragment: [infer_request:infer_postprocess]

get_profiling_info()
+++++++++++++++++++++

The method returns the profiling info which was measured during pipeline stages execution:

.. doxygensnippet:: src/plugins/template/src/sync_infer_request.cpp
   :language: cpp
   :fragment: [infer_request:get_profiling_info]

cancel()
+++++++++

The plugin specific method allows to interrupt the synchronous execution from the AsyncInferRequest:

.. doxygensnippet:: src/plugins/template/src/sync_infer_request.cpp
   :language: cpp
   :fragment: [infer_request:cancel]


The next step in the plugin library implementation is the :doc:`Asynchronous Inference Request <asynch-inference-request>` class.

