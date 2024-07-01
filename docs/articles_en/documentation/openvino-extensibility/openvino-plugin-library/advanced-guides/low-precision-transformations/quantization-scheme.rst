.. {#openvino_docs_OV_UG_lpt_step2_markup}

Quantization Scheme
==============================


.. meta::
   :description: Learn about quantization scheme.

.. toctree::
   :maxdepth: 1
   :caption: Low Precision Transformations

Key steps in quantization scheme:
* Low Precision Transformations: Quantize decomposition
* Low Precision Transformations: move Dequantize through operations
* Plugin: fuse operations and Quantize

Quantization scheme features:
* Quantization operation is expressed through the ``FakeQuantize`` operation. ``FakeQuantize`` is not just scale and shift. You can find more details here: :doc:`FakeQuantize-1 <../../../openvino-ir-format/operation-sets/operation-specs/quantization/fake-quantize-1>`. Note, please, if ``FakeQuantize`` input interval is equal output interval then ``FakeQuantize`` degenerates to ``Multiply``, ``Subtract`` and ``Convert`` (scale & shift).
* Dequantization operation is expressed through the element-wise ``Convert``, ``Subtract`` and ``Multiply`` operations. ``Convert`` and ``Subtract`` can be skipped. The operations can be handled as  typical element-wise operations, for example: fused or transformed to another.
* OpenVINO plugins fuse ``Dequantize`` and ``Quantize`` operation after low precision operation and don't fuse ``Quantize`` before.

Quantization scheme example for int8 quantization for two ``Convolution`` operations in CPU plugin.
.. image:: ../../../../../assets/images/quantization_scheme.svg
   :alt: Quantization scheme
