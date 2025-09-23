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

* In **ACCURACY mode**, the device does not convert floating-point tensors to a smaller
  floating-point type. This ensures that the accuracy metrics remain close to the original
  values obtained during training, based on the device’s actual capabilities.
  For example, if the device supports both ``f16`` and ``f32``, and the model is created for
  ``f16``, the network will execute in ``f16``. Similarly, if the model is created for ``f32``,
  the network will execute in ``f32``.
  Additionally, :ref:`Dynamic Quantization <enabling-runtime-optimizations>` is disabled in this mode.
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
to handle such cases manually. So if possible, it is generally recommended to use high
level ``execution_mode`` property.

Another thing is that ``inference_precision`` is also a hint, so the value provided is not guaranteed
to be used by Runtime (mainly in cases where the current device does not have the required hardware
capabilities).

.. note::

   All devices only support floating-point data types (``f32``, ``f16``, ``bf16``) as a value
   for ``inference_precision`` attribute.


Activations Scaling
###################

Since ``f16`` has a smaller dynamic range compared to ``f32`` or ``bf16``, overflow might occur when using ``f16`` for ``inference_precision``.
To address this issue, ``activation scaling`` divides the input of linear operations like ``MatMul`` or ``Convolution`` by the ``activations scale factor``, ensuring the layer's output does not exceed ``f16``'s dynamic range.
The layer's output must then be multiplied by the ``activations scale factor`` to restore it to its original value, but overflow can occur again during this process.
``Activation scaling`` utilizes :doc:`LPT <../../../openvino-extensibility/openvino-plugin-library/advanced-guides/low-precision-transformations>` to delay the multiplication by the scale factor as much as possible, preventing this from happening.
The ``activations scale factor`` can be specified to the ``rt_info`` in the IR or specified via ``ov::hint::activations_scale_factor``.
Currently, this property is supported by GPU.

.. scrollbox::   

   .. code-block:: cpp

      <?xml version="1.0" ?>
      <net name="model_file_name" version="10">
         ...
         <rt_info>
            ...
            <runtime_options>
                  <ACTIVATIONS_SCALE_FACTOR value="8.0" />
            </runtime_options>
            ...
         </rt_info>
      </net>



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


