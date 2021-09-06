# OpenVINO™ Low Precision Transformations: attributes {#openvino_docs_IE_DG_lpt_attributes}

## Introduction

| Name                                                                                | Target                 | Required | Mutable |
|-------------------------------------------------------------------------------------|------------------------|----------|---------|
| [AvgPoolPrecisionPreserved](@ref openvino_docs_IE_DG_lpt_AvgPoolPrecisionPreserved) | Precision              | No       | Yes     |
| [IntervalsAlignment](@ref openvino_docs_IE_DG_lpt_IntervalsAlignment)               | Quantization interval  | Yes      | Yes     |
| [PerTensorQuantization](@ref openvino_docs_IE_DG_lpt_PerTensorQuantization)         | Precision              | Yes      | No      |
| [PrecisionPreserved](@ref openvino_docs_IE_DG_lpt_PrecisionPreserved)               | Precision              | Yes      | Yes     |
| [Precisions](@ref openvino_docs_IE_DG_lpt_Precisions)                               | Precision              | Yes      | Yes     |
| [QuantizationAlignment](@ref openvino_docs_IE_DG_lpt_QuantizationAlignment)         | Quantization alignment | Yes      | Yes     |

> `Target` attribute group defines attribute usage during model transformation for the best performance:
>  - `Precision` - the attribute is used to define the most optimal output port precision.
>  - `Quantization interval` - the attribute is used to define quantization interval.
>  - `Quantization alignment` - the attribute is used to define quantization alignment: per-channel or per-tensor quantization.
>
> `Required` attribute group defines if attribute usage is required to get optimal model during transformation or not:
>  - `Yes` - the attribute is used in low precision optimization which is used by all OpenVINO plugins.
>  - `No` - the attribute is used in specific OpenVINO plugin.
>
> `Mutable` attribute group defines if transformation can update existing attribute or not:
>  - `Yes` - the attribute can be updated by the next transformations in pipeline. But attribute update order is still important.
>  - `No` - existing attribute can not be updated by the next transformation. Previous handled transformation has optimized model in accordance with current value.

`FakeQuantize` decomposition is mandatory part of low precision transformations. Attributes which are used during decomposition are mandatory. Optional attributes are required for some operations only.

Attributes usage by transformations:

| Attribute name            | Created by transformations                        | Used by transformations                                                                                                           |
|---------------------------|---------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| PrecisionPreserved        | MarkupPrecisions, MarkupAvgPoolPrecisionPreserved | AlignQuantizationIntervals, AlignQuantizationParameters, FakeQuantizeDecompositionTransformation, MarkupAvgPoolPrecisionPreserved |
| AvgPoolPrecisionPreserved | MarkupAvgPoolPrecisionPreserved                   |                                                                                                                                   |
| Precisions                | MarkupCanBeQuantized, MarkupPrecisions            | FakeQuantizeDecompositionTransformation                                                                                           |
| PerTensorQuantization     | MarkupPerTensorQuantization                       |                                                                                                                                   |
| IntervalsAlignment        | AlignQuantizationIntervals                        | FakeQuantizeDecompositionTransformation                                                                                           |
| QuantizationAlignment     | AlignQuantizationParameters                       | FakeQuantizeDecompositionTransformation                                                                                           |

> Note, please, the same type attribute instances can be created in different transformations. This approach is result of transformation single-responsibility principle. For example `Precision` attribute instances are created in `MarkupCanBeQuantized` and `MarkupPrecisions` transformations but the creation reason is different.