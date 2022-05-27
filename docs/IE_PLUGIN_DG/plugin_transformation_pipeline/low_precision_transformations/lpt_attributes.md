# Attributes {#openvino_docs_OV_UG_lpt_attributes}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :caption: Attributes
   :hidden:

   AvgPoolPrecisionPreserved <openvino_docs_OV_UG_lpt_AvgPoolPrecisionPreserved>
   IntervalsAlignment <openvino_docs_OV_UG_lpt_IntervalsAlignment>   
   PrecisionPreserved <openvino_docs_OV_UG_lpt_PrecisionPreserved>
   Precisions <openvino_docs_OV_UG_lpt_Precisions>
   QuantizationAlignment <openvino_docs_OV_UG_lpt_QuantizationAlignment>
   QuantizationGranularity <openvino_docs_OV_UG_lpt_QuantizationGranularity>

@endsphinxdirective

## Introduction

| Name                                                                                | Target                   | Required | Mutable |
|-------------------------------------------------------------------------------------|--------------------------|----------|---------|
| [AvgPoolPrecisionPreserved](@ref openvino_docs_OV_UG_lpt_AvgPoolPrecisionPreserved) | Precision                | No       | Yes     |
| [IntervalsAlignment](@ref openvino_docs_OV_UG_lpt_IntervalsAlignment)               | Quantization interval    | Yes      | Yes     |
| [PrecisionPreserved](@ref openvino_docs_OV_UG_lpt_PrecisionPreserved)               | Precision                | Yes      | Yes     |
| [Precisions](@ref openvino_docs_OV_UG_lpt_Precisions)                               | Precision                | Yes      | Yes     |
| [QuantizationAlignment](@ref openvino_docs_OV_UG_lpt_QuantizationAlignment)         | Quantization granularity | Yes      | Yes     |
| [QuantizationGranularity](@ref openvino_docs_OV_UG_lpt_QuantizationGranularity)     | Quantization granularity | Yes      | No      |

> `Target` attribute group defines attribute usage during model transformation for the best performance:
>  - `Precision` - the attribute defines the most optimal output port precision.
>  - `Quantization interval` - the attribute defines quantization interval.
>  - `Quantization alignment` - the attribute defines quantization granularity in runtime: per-channel or per-tensor quantization.
>  - `Quantization granularity` - the attribute is set by plugin to define quantization granularity: per-channel or per-tensor quantization.
>
> `Required` attribute group checks if attribute usage is required to get an optimal model during transformation:
>  - `Yes` - the attribute is used by all OpenVINO plugins for low-precision optimization.
>  - `No` - the attribute is used in a specific OpenVINO plugin.
>
> `Mutable` attribute group checks if transformation can update an existing attribute:
>  - `Yes` - the attribute can be updated by the next transformations in the pipeline. But attribute update order is still important.
>  - `No` - existing attribute can not be updated by the next transformation. Previous handled transformation has optimized a model according to the current value.

`FakeQuantize` decomposition is a mandatory part of low precision transformations. Attributes used during decomposition are mandatory. Optional attributes are required only for certain operations.

Attributes usage by transformations:

| Attribute name            | Created by transformations                        | Used by transformations                                                                                                           |
|---------------------------|---------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| PrecisionPreserved        | MarkupPrecisions, MarkupAvgPoolPrecisionPreserved | AlignQuantizationIntervals, AlignQuantizationParameters, FakeQuantizeDecompositionTransformation, MarkupAvgPoolPrecisionPreserved |
| AvgPoolPrecisionPreserved | MarkupAvgPoolPrecisionPreserved                   |                                                                                                                                   |
| Precisions                | MarkupCanBeQuantized, MarkupPrecisions            | FakeQuantizeDecompositionTransformation                                                                                           |
| PerTensorQuantization     | MarkupPerTensorQuantization                       |                                                                                                                                   |
| IntervalsAlignment        | AlignQuantizationIntervals                        | FakeQuantizeDecompositionTransformation                                                                                           |
| QuantizationAlignment     | AlignQuantizationParameters                       | FakeQuantizeDecompositionTransformation                                                                                           |

> **Note:** the same type of attribute instances can be created in different transformations. This approach is the result of the single-responsibility principle of the transformation. For example, `Precision` attribute instances are created in `MarkupCanBeQuantized` and `MarkupPrecisions` transformations, but the reasons for their creation are different.