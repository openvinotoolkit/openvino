# OpenVINO™ Runtime Python API Inference {#openvino_docs_OV_UG_Python_API_inference}


.. warning:: All mentioned methods are very dependant on a specific hardware and software set-up. Please consider conducting your own experiments with various models and different input/output sizes. Presented methods are not universal, they may or may not apply to the specific pipeline. Please consider all tradeoffs and avoid premature optimizations. 


Direct Inference with ``CompiledModel``
#######################################

``CompiledModel`` class provides ``__call__`` method which runs a single synchronous inference using given model. In addition to a compact code, all future calls to ``CompiledModel.__call__`` result in less overhead as the object reuse already created ``InferRequest``.


.. doxygensnippet:: docs/snippets/ov_python_inference.py
   :language: python
   :fragment: [direct_inference]


Shared Memory on Inputs
#######################

While using ``CompiledModel``, ``InferRequest`` and ``AsyncInferQueue`` inference methods, OpenVINO™ Runtime Python API provides additional "Shared Memory" mode. Specify ``shared_memory`` flag to enable or disable this feature. "Shared Memory" mode could be beneficial when inputs are large and copying of data is considered an expensive operation. This feature creates shared ``Tensor`` instances with "zero-copy" approach, reducing overhead of setting inputs to minimum. Example usage:


.. doxygensnippet:: docs/snippets/ov_python_inference.py
   :language: python
   :fragment: [shared_memory_inference]


.. note:: "Shared Memory" is enabled by default in ``CompiledModel.__call__``. For other methods like ``InferRequest.infer`` or ``InferRequest.start_async``, it is required to manually set the flag to ``True``.


.. warning:: When data is being shared, all modifications may affect inputs of the inference! Use this feature with caution, especially in multi-threaded/parallel code where data can be modified outside of the function's control flow.


Hiding Latency with Asynchronous Calls
######################################

Asynchronous calls allow to hide latency to optimize overall runtime of a codebase. For example, ``InferRequest.start_async`` releases the GIL and provides non-blocking call. It is beneficial to process other calls while waiting to finish compute-intensive inference. Example usage:


.. doxygensnippet:: docs/snippets/ov_python_inference.py
   :language: python
   :fragment: [hiding_latency]


.. note:: It is up to users/developers to optimize flow in a codebase to benefit from potential parallelization.


"Postponed Return" with Asynchronous Calls
##########################################

"Postponed Return" is a practice to omit overhead of ``OVDict``, which is always returned from synchronous calls. "Postponed Return" could be applied when:
* only part of output data is required. For example, only one specific output is significant in given pipeline step and all outputs are large thus expensive to copy.
* data is not required "now". For example, it can be later extracted inside the pipeline as a part of latency hiding.
* data return is not required at all. For example, models are being chained with the pure ``Tensor`` interface.


.. doxygensnippet:: docs/snippets/ov_python_inference.py
   :language: python
   :fragment: [no_return_inference]
