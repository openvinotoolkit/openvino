Python API Advanced Inference
===============================================================================================

.. meta::
   :description: OpenVINO™ Python API enables you to share memory for input and
                 output data, hide the latency with asynchronous calls and implement
                 “postponed return”.

.. important::

   All mentioned methods are not universal, as they are hardware and software specific,
   and may not be applicable in your pipeline. You can experiment with various models and
   different input/output sizes. Consider all tradeoffs and avoid premature optimization.

Inference with ``CompiledModel``
###############################################################################################

The ``CompiledModel`` class provides the ``__call__`` method that runs a single synchronous
inference on a given model. In addition to a compact code, all future calls to
``CompiledModel.__call__`` will result in less overhead, as the object reuses the already
created ``InferRequest``. See the example below:

.. doxygensnippet:: docs/articles_en/assets/snippets/ov_python_inference.py
   :language: python
   :fragment: [direct_inference]


Shared Memory for Input and Output Data
###############################################################################################

OpenVINO™ Python API provides “Shared Memory”  - an additional mode
for ``CompiledModel``, ``InferRequest`` and ``AsyncInferQueue``.
Specify the ``share_inputs`` and ``share_outputs`` flag to enable or disable this feature.

This feature creates shared ``Tensor`` instances with the “zero-copy” approach,
reducing overhead of setting input to minimum. It also creates numpy views on output data.

The “Shared Memory” mode is recommended for dealing with large input/output data.
To enable this feature, set the ``share_inputs`` and ``share_outputs`` flags to ``True``.
See the examples below:

.. doxygensnippet:: docs/articles_en/assets/snippets/ov_python_inference.py
   :language: python
   :fragment: [shared_memory_inference]


“Shared Memory” for input data is enabled by default in ``CompiledModel.__call__``.
For other methods, like ``InferRequest.infer`` or ``InferRequest.start_async``,
you need to set the flag to ``True`` manually.

“Shared Memory” for output data is disabled by default in all sequential inference
methods (``CompiledModel.__call__`` and ``InferRequest.infer``). Set it to ``True``
to enable it.

.. warning::

   * When data is being shared, all modifications, including subsequent inference calls,
     may affect input and output of the inference!
   * Use this feature with caution, especially in multi-threaded/parallel code,
     where data can be modified outside of the function's control flow.

Hiding Latency with Asynchronous Calls
###############################################################################################

You can hide latency to optimize overall runtime of a codebase by using the
``InferRequest.start_async`` call, which releases the GIL and provides a non-blocking
call. Your application may benefit from processing other calls while waiting to finish
compute-intensive inference. See the example below:

.. doxygensnippet:: docs/articles_en/assets/snippets/ov_python_inference.py
   :language: python
   :fragment: [hiding_latency]


.. note::

   Consider optimizing the flow in a codebase to make the most of potential parallelization.

Postponed Return with Asynchronous Calls
###############################################################################################

“Postponed Return” is used to ignore overhead of :ref:`OVDict <inference_results_ovdict>`,
always returned from synchronous calls, and can be applied when:

* partial output data is required. For example, when the output is large, thus, expensive to
  copy and only its specific part is needed.
* data is not required “now”. For example, in attempt to hide the latency, it can be
  extracted later.
* data return is not required at all. For example, models are being chained with the
  pure ``Tensor`` interface.

.. doxygensnippet:: docs/articles_en/assets/snippets/ov_python_inference.py
   :language: python
   :fragment: [no_return_inference]
