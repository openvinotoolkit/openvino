# IntervalsAlignment Attribute {#openvino_docs_OV_UG_lpt_IntervalsAlignment}

@sphinxdirective

.. meta::
   :description: Learn about IntervalsAlignment attribute, which describes a subgraph with the same quantization intervals alignment.


:ref:`ngraph::IntervalsAlignmentAttribute <doxid-classngraph_1_1_intervals_alignment_attribute>` class represents the ``IntervalsAlignment`` attribute.

The attribute defines a subgraph with the same quantization intervals alignment. ``FakeQuantize`` operations are included. The attribute is used by quantization operations.

.. list-table::
    :header-rows: 1

    * - Property name
      - Values
    * - Required
      - Yes
    * - Defined
      - Operation
    * - Properties
      - combined interval, minimal interval, minimal levels, preferable precisions

@endsphinxdirective
