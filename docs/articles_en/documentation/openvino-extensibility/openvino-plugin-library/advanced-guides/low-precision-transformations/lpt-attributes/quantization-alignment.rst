QuantizationAlignment Attribute
===============================


.. meta::
   :description: Learn about QuantizationAlignment attribute, which describes a subgraph with the same quantization alignment.


``ov::QuantizationAlignmentAttribute`` class represents the ``QuantizationAlignment`` attribute.

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

