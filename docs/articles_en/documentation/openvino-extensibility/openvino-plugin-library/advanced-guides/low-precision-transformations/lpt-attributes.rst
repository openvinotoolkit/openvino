.. {#../low-precision-transformations_attributes}

Attributes
==========


.. meta::
   :description: Check the lists of attributes created or used by model transformations.


.. toctree::
   :maxdepth: 1
   :caption: Attributes
   :hidden:

   AvgPoolPrecisionPreserved <../low-precision-transformations_AvgPoolPrecisionPreserved>
   IntervalsAlignment <../low-precision-transformations_IntervalsAlignment>   
   PrecisionPreserved <../low-precision-transformations_PrecisionPreserved>
   Precisions <../low-precision-transformations_Precisions>
   QuantizationAlignment <../low-precision-transformations_QuantizationAlignment>
   QuantizationGranularity <../low-precision-transformations_QuantizationGranularity>

Introduction
############

.. list-table::
    :header-rows: 1

    * - Name
      - Target
      - Required
      - Mutable
    * - :doc:`AvgPoolPrecisionPreserved <../low-precision-transformations_AvgPoolPrecisionPreserved>`
      - Precision
      - No
      - Yes
    * - :doc:`IntervalsAlignment <../low-precision-transformations_IntervalsAlignment>`
      - Quantization interval
      - Yes
      - Yes
    * - :doc:`PrecisionPreserved <../low-precision-transformations_PrecisionPreserved>`
      - Precision
      - Yes
      - Yes
    * - :doc:`Precisions <../low-precision-transformations_Precisions>`
      - Precision
      - Yes
      - Yes
    * - :doc:`QuantizationAlignment <../low-precision-transformations_QuantizationAlignment>`
      - Quantization granularity
      - Yes
      - Yes
    * - :doc:`QuantizationGranularity <../low-precision-transformations_QuantizationGranularity>`
      - Quantization granularity
      - Yes
      - No 
      

``Target`` attribute group defines attribute usage during model transformation for the best performance:

* ``Precision`` - the attribute defines the most optimal output port precision.
* ``Quantization interval`` - the attribute defines quantization interval.
* ``Quantization alignment`` - the attribute defines quantization granularity in runtime: per-channel or per-tensor quantization.
* ``Quantization granularity`` - the attribute is set by plugin to define quantization granularity: per-channel or per-tensor quantization.

``Required`` attribute group defines if attribute usage is required to get an optimal model during transformation:

* ``Yes`` - the attribute is used by all OpenVINO plugins for low-precision optimization.
* ``No`` - the attribute is used in a specific OpenVINO plugin.

``Mutable`` attribute group defines if transformation can update an existing attribute:

* ``Yes`` - the attribute can be updated by the next transformations in the pipeline. But attribute update order is still important.
* ``No`` - existing attribute can not be updated by the next transformation. Previous handled transformation has optimized a model according to the current value.

``FakeQuantize`` decomposition is a mandatory part of low precision transformations. Attributes used during decomposition are mandatory. Optional attributes are required only for certain operations.

Attributes usage by transformations:

.. list-table::
    :header-rows: 1

    * - Attribute name
      - Created by transformations
      - Used by transformations
    * - PrecisionPreserved
      - MarkupPrecisions, MarkupAvgPoolPrecisionPreserved
      - AlignQuantizationIntervals, AlignQuantizationParameters, FakeQuantizeDecompositionTransformation, MarkupAvgPoolPrecisionPreserved
    * - AvgPoolPrecisionPreserved
      - MarkupAvgPoolPrecisionPreserved
      - 
    * - Precisions
      - MarkupCanBeQuantized, MarkupPrecisions
      - FakeQuantizeDecompositionTransformation
    * - PerTensorQuantization
      - MarkupPerTensorQuantization
      - 
    * - IntervalsAlignment
      - AlignQuantizationIntervals
      - FakeQuantizeDecompositionTransformation
    * - QuantizationAlignment
      - AlignQuantizationParameters
      - FakeQuantizeDecompositionTransformation

.. note::                                                                     
   The same type of attribute instances can be created in different transformations. This approach is the result of the transformation single-responsibility principle. For example, ``Precision`` attribute instances are created in ``MarkupCanBeQuantized`` and ``MarkupPrecisions`` transformations, but the reasons for their creation are different.

