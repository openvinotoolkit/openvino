Precision Control
=================


The choice of data types is essential to the inference runtime, which can have a huge impact on
the performance and other metrics. Usually 2 types of precision are identified:

1. Model storage precision (IR precision),
2. Model inference precision.

Inference precision no longer depends on the precision of IR, which means that users have
several options to find the balance between model performance and accuracy.

Essentially, the IR precision becomes a way of compressing the model by reducing the precision
of the weights, and it does not affect how the devices execute the model. This change clears up
a lot of confusion where, for example, you couldn't execute a high-performance model on the GPU
by default, and the behavior between devices was different.

This guide will focus on how to control inference precision. And using lower precision is
important for performance because compute bandwidth tends to be higher for smaller data
types, and hardware often has special blocks for efficient multiply-accumulate operations
with smaller data types only (e.g. Intel Xᵉ Matrix Extensions (XMX) on GPU and Intel
Advanced Matrix Extensions (AMX) on CPU do not support ``f32``). Also, I/O operations
requires less memory due to the smaller tensor byte size. This guide will focus on how
to control inference precision.

.. _execution-mode:

Execution Mode
##############

``ov::hint::execution_mode`` is a high-level hint to control whether the user wants to keep
the best accuracy (**ACCURACY mode**) or if the device can do some optimizations that
may lower the accuracy for performance reasons (**PERFORMANCE mode**)

* In **ACCURACY mode**, the device cannot convert floating point tensors to a smaller
  floating point type, so devices try to keep the accuracy metrics as close as possible to
  the original values obtained after training relative to the device's real capabilities.
  This means that most devices will infer with ``f32`` precision if your device supports it.
  In this mode, the :ref:`Dynamic Quantization <enabling-runtime-optimizations>` is disabled.
* In **PERFORMANCE mode**, the device can convert to smaller data types and apply other
  optimizations that may have some impact on accuracy rates, although we still try to
  minimize accuracy loss and may use mixed precision execution in some cases.

If the model has been quantized using
:doc:`OpenVINO optimization tools <../../model-optimization-guide/quantizing-models-post-training>`
or any other method, the quantized operators will be executed with the target integer
precision if the device has hardware acceleration for that type. For example, quantized
``int8`` primitives are executed with ``int8`` precision for both **ACCURACY** and
**PERFORMANCE modes** if the device provides higher compute bandwidth for 8-bit data types
compared to any available floating-point type. On the other hand, devices without hardware
acceleration for the ``int8`` data type can keep such operators in floating point precision,
and the exact floating point type will be affected by ``execution_mode`` and
``inference_precision`` properties.

Code examples:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_execution_mode.py
         :language: python
         :fragment: [ov:execution_mode:part0]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_execution_mode.cpp
         :language: cpp
         :fragment: [ov:execution_mode:part0]


Inference Precision
###################

``ov::hint::inference_precision`` precision is a lower-level property that allows you
to specify the exact precision the user wants, but is less portable. For example, CPU
supports ``f32`` inference precision and ``bf16`` on some platforms, GPU supports ``f32``
and ``f16``, so if a user wants to an application that uses multiple devices, they have
to handle all these combinations manually or let OV do it automatically by using higher
level ``execution_mode`` property.

.. note::

   When using ``execution_mode``, you need to be aware that using **ACCURACY mode**
   will result in enabling ``f32`` inference precision, but it will also disable
   :ref:`dynamic quantization <enabling-runtime-optimizations>`. This may highly affect
   inference performance (esp. on the Intel® Xeon® platforms and Intel® GPU devices)

Another thing is that ``inference_precision`` is also a hint, so the value provided is not guaranteed
to be used by Runtime (mainly in cases where the current device does not have the required hardware
capabilities).

.. note::

   All devices only support floating-point data types (``f32``, ``f16``, ``bf16``) as a value
   for ``inference_precision`` attribute.


.. _limited_inference_precision:

Limitation of the ``bf16`` inference precision
++++++++++++++++++++++++++++++++++++++++++++++

It is important to mention that inferring FP16 and FP32 LLM models with the ``bf16`` runtime
precision may result in higher accuracy loss than the pre-determined threshold of 0.5%.
Higher accuracy drop may occur when inferring **dolly-v2-12b**, **dolly-v2-3b**, and
**gpt-neox-20b** original Pytorch models with ``bf16``, and is caused by a limited
precision representation.

To solve the issue, you might use an INT8 model and force the FP32 inference precision.
The accuracy of an INT8 model with FP32 is nearly the same as of an FP16 model with ``f32``.
Additionally, selective FP32 execution of ops on CPU plugin together with the NNCF ``bf16``
calibration could potentially mitigate the accuracy loss.

However, the solutions mentioned above would, unfortunately, also result in significant
performance drop during a large batch size inference task on machines with Intel AMX-BF16 SPR.
In such cases, the fused multiply-add operation (FMA) is used instead of AMX. Also,
in a compute-bound case, such as the LLM batch inference/serving, these workarounds
would drastically reduce the throughput by more than 60%.



Additional Resources
####################

* :doc:`Inference Devices and Modes <../inference-devices-and-modes>`


