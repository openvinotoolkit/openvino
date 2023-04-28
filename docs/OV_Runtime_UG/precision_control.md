# Precision Control {#openvino_docs_OV_UG_Precision_Control}

@sphinxdirective

Regardless of IR precision, devices will run in high performance mode by default. For GPU this means ``fp16`` inference and for CPU - ``bf16`` inference (if available). Previously, you had to convert the IR to ``fp16`` for the GPU to run in ``fp16``, and for the CPU, it was high precision by default. Now the devices have been aligned and this selection has been disconnected from IR precision. If high performance is causing the accuracy issue (only seen a few times in history), you can use the ``inference_precision`` hint and set it to accuracy.

Separately, you can control the IR precision. Essentially the IR precision becomes a way to compress your model by reducing the precision of the weights, and it doesn't affect how devices execute the model.

This change clears up a lot of confusion where, for example, you couldn't execute a high-performance model on the GPU by default, and the behavior between devicess was different.

.. note::

   All devices (except GNA) only support floating-point data types (``f32``, ``f16``, ``bf16``) as a value for ``inference_precision`` attribute, because quantization cannot be done in Runtime. The GNA plugin has the ability to perform model quantization on ``core.compile_model()`` call, so it supports integer data types in addition to ``f32``.

Execution Mode
##############

``ov::hint::execution_mode`` is a high-level hint to control whether the user wants to keep the best accuracy (**ACCURACY mode**) or if the plugin can do some optimizations that may lower the accuracy for performance reasons (**PERFORMANCE mode**)

Inference Precision
###################

``ov::hint::inference`` precision is a lower-level property that allows you to specify the exact precision the user wants, but is less portable. For example, CPU supports ``f32`` inference precision and ``bf16`` on some platforms, GPU supports ``fp32`` and ``fp16`` while GNA supports ``i8`` and ``i16``, so if a user wants to an application that uses multiple devices, they have to handle all these combinations manually or let OV do it automatically by using higher level ``execution_mode`` property. Another thing is that ``inference_precision`` is also a hint, so the value provided is not guaranteed to be used by Runtime (mainly in cases where the current device does not have the required hardware capabilities).

Additional Resources
####################

* :doc:`Supported Devices <openvino_docs_OV_UG_Working_with_devices>`

@endsphinxdirective

