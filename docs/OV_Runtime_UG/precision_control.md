# Precision Control {#openvino_docs_OV_UG_Precision_Control}

@sphinxdirective

The choice of data types is essential to the inference runtime, which can have a huge impact on the performance and other metrics. Usually 2 types of precision are identified:

1. Model storage precision (IR precision),
2. Model inference precision.

Previously, these 2 precisions were interrelated, and model storage precision could affect the inference precision in some devices (e.g. GPU did ``f16`` inference only for ``f16`` IRs).

With the ``2023.0`` release this behavior has been changed and the inference precision no longer depends on the precision of IR. Now users have several knobs to find the balance between model performance and accuracy.

Essentially, the IR precision becomes a way of compressing the model by reducing the precision of the weights, and it does not affect how the devices execute the model. And using lower precision is important for performance because compute bandwidth tends to be higher for smaller data types, and hardware often has special blocks for efficient multiply-accumulate operations with smaller data types only (e.g. XMX on GPU and AMX on CPU do not support ``fp32``). Also, I/O operations requires less memory due to the smaller tensor byte size. This guide will focus on how to control inference precision.

This change clears up a lot of confusion where, for example, you couldn't execute a high-performance model on the GPU by default, and the behavior between devicess was different.

Execution Mode
##############

``ov::hint::execution_mode`` is a high-level hint to control whether the user wants to keep the best accuracy (**ACCURACY mode**) or if the device can do some optimizations that may lower the accuracy for performance reasons (**PERFORMANCE mode**)

* In **ACCURACY mode**, the device cannot convert floating point tensors to a smaller floating point type, so devices try to keep the accuracy metrics as close as possible to the original values ​​obtained after training relative to the device's real capabilities. This means that most devices will infer with ``fp32`` precision if your device supports it.
* In **PERFORMANCE mode**, the device can convert to smaller data types and apply other optimizations that may have some impact on accuracy rates, although we still try to minimize accuracy loss and may use mixed precision execution in some cases.

Inference Precision
###################

``ov::hint::inference`` precision is a lower-level property that allows you to specify the exact precision the user wants, but is less portable. For example, CPU supports ``f32`` inference precision and ``bf16`` on some platforms, GPU supports ``fp32`` and ``fp16`` while GNA supports ``i8`` and ``i16``, so if a user wants to an application that uses multiple devices, they have to handle all these combinations manually or let OV do it automatically by using higher level ``execution_mode`` property. Another thing is that ``inference_precision`` is also a hint, so the value provided is not guaranteed to be used by Runtime (mainly in cases where the current device does not have the required hardware capabilities).

.. note::

   All devices (except GNA) only support floating-point data types (``f32``, ``f16``, ``bf16``) as a value for ``inference_precision`` attribute, because quantization cannot be done in Runtime. The GNA plugin has the ability to perform model quantization on ``core.compile_model()`` call, so it supports integer data types in addition to ``f32``.


Additional Resources
####################

* :doc:`Supported Devices <openvino_docs_OV_UG_Working_with_devices>`

@endsphinxdirective

