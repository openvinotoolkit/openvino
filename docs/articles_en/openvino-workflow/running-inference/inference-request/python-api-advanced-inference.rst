OpenVINO™ Runtime Python API Advanced Inference
=================================================


.. meta::
   :description: OpenVINO™ Runtime Python API enables you to share memory on inputs, hide
                 the latency with asynchronous calls and implement "postponed return".


.. warning::

   All mentioned methods are very dependent on a specific hardware and software set-up.
   Consider conducting your own experiments with various models and different input/output
   sizes. The methods presented here are not universal, they may or may not apply to the
   specific pipeline. Please consider all tradeoffs and avoid premature optimizations.


Direct Inference with ``CompiledModel``
#######################################

The ``CompiledModel`` class provides the ``__call__`` method that runs a single synchronous inference using the given model. In addition to a compact code, all future calls to ``CompiledModel.__call__`` will result in less overhead, as the object reuses the already created ``InferRequest``.


.. doxygensnippet:: docs/articles_en/assets/snippets/ov_python_inference.py
   :language: python
   :fragment: [direct_inference]


Shared Memory on Inputs and Outputs
###################################

While using ``CompiledModel``, ``InferRequest`` and ``AsyncInferQueue``,
OpenVINO™ Runtime Python API provides an additional mode - "Shared Memory".
Specify the ``share_inputs`` and ``share_outputs`` flag to enable or disable this feature.
The "Shared Memory" mode may be beneficial when inputs or outputs are large and copying data is considered an expensive operation.

This feature creates shared ``Tensor``
instances with the "zero-copy" approach, reducing overhead of setting inputs
to minimum. For outputs this feature creates numpy views on data. Example usage:


.. doxygensnippet:: docs/articles_en/assets/snippets/ov_python_inference.py
   :language: python
   :fragment: [shared_memory_inference]


.. note::

   "Shared Memory" on inputs is enabled by default in ``CompiledModel.__call__``.
   For other methods, like ``InferRequest.infer`` or ``InferRequest.start_async``,
   it is required to set the flag to ``True`` manually.
   "Shared Memory" on outputs is disabled by default in all sequential inference methods (``CompiledModel.__call__`` and ``InferRequest.infer``). It is required to set the flag to ``True`` manually.

.. warning::

   When data is being shared, all modifications (including subsequent inference calls) may affect inputs and outputs of the inference!
   Use this feature with caution, especially in multi-threaded/parallel code,
   where data can be modified outside of the function's control flow.


Hiding Latency with Asynchronous Calls
######################################

Asynchronous calls allow to hide latency to optimize overall runtime of a codebase.
For example, ``InferRequest.start_async`` releases the GIL and provides non-blocking call.
It is beneficial to process other calls while waiting to finish compute-intensive inference.
Example usage:

.. doxygensnippet:: docs/articles_en/assets/snippets/ov_python_inference.py
   :language: python
   :fragment: [hiding_latency]


.. note::

   It is up to the user/developer to optimize the flow in a codebase to benefit from potential parallelization.


"Postponed Return" with Asynchronous Calls
##########################################

"Postponed Return" is a practice to omit overhead of ``OVDict``, which is always returned from
synchronous calls. "Postponed Return" could be applied when:

* only a part of output data is required. For example, only one specific output is significant in a given pipeline step and all outputs are large, thus, expensive to copy.
* data is not required "now". For example, it can be later extracted inside the pipeline as a part of latency hiding.
* data return is not required at all. For example, models are being chained with the pure ``Tensor`` interface.


.. doxygensnippet:: docs/articles_en/assets/snippets/ov_python_inference.py
   :language: python
   :fragment: [no_return_inference]


