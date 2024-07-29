Quantization Scheme
==============================


.. meta::
   :description: Learn about quantization scheme.

.. toctree::
   :maxdepth: 1
   :caption: Low Precision Transformations

Key steps in the quantization scheme:

* Low Precision Transformations: ``FakeQuantize`` decomposition to Quantize with a low precision output and Dequantize. For more details, refer to the :doc:`Quantize decomposition <../low-precision-transformations>` section.
* Low Precision Transformations: move Dequantize through operations. For more details, refer to the :doc:`Main transformations <./step3-main>` section.
* Plugin: fuse operations with Quantize and inference in low precision.

Quantization scheme features:

* Quantization operation is expressed through the ``FakeQuantize`` operation, which involves more than scale and shift. For more details, see: :doc:`FakeQuantize-1 <../../../../openvino-ir-format/operation-sets/operation-specs/quantization/fake-quantize-1>`. If the ``FakeQuantize`` input and output intervals are the same, ``FakeQuantize`` degenerates to ``Multiply``, ``Subtract`` and ``Convert`` (scale & shift).
* Dequantization operation is expressed through element-wise ``Convert``, ``Subtract`` and ``Multiply`` operations. ``Convert`` and ``Subtract`` are optional. These operations can be handled as typical element-wise operations, for example, fused or transformed to another.
* OpenVINO plugins fuse ``Dequantize`` and ``Quantize`` operations after a low precision operation and do not fuse ``Quantize`` before it.

Here is a quantization scheme example for int8 quantization applied to a part of a model with two ``Convolution`` operations in CPU plugin.

.. image:: ../../../../../assets/images/quantization_scheme.svg
   :alt: Quantization scheme
