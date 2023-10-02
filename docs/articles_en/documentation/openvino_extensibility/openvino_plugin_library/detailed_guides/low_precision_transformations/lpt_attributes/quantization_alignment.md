# QuantizationAlignment Attribute {#openvino_docs_OV_UG_lpt_QuantizationAlignment}

@sphinxdirective

.. meta::
   :description: Learn about QuantizationAlignment attribute, which describes a subgraph with the same quantization alignment.


:ref:`ngraph::QuantizationAlignmentAttribute <doxid-classngraph_1_1_quantization_alignment_attribute>` class represents the ``QuantizationAlignment`` attribute.

The attribute defines a subgraph with the same quantization alignment. ``FakeQuantize`` operations are not included. The attribute is used by quantization operations.

.. list-table::
    :header-rows: 1

    * - Property name
      - Values
    * - Required
      - Yes
    * - Defined
      - Operation
    * - Properties
      - value (boolean)

@endsphinxdirective
