---
sidebar_position: 2
sidebar_label: "Attributes"
description: "Check the lists of attributes created or used by model transformations."
---

# Attributes


## Introduction

| Name | Target | Required | Mutable |
| --- | --- | --- | --- |
| [AvgPoolPrecisionPreserved](avg-pool-precision-preserved.md) | Precision | No | Yes |
| [IntervalsAlignment](intervals-alignment.md) | Quantization interval | Yes | Yes |
| [PrecisionPreserved](precision-preserved.md) | Precision | Yes | Yes |
| [Precisions](precisions.md) | Precision | Yes | Yes |
| [QuantizationAlignment](quantization-alignment.md) | Quantization granularity | Yes | Yes |
| [QuantizationGranularity](quantization-granularity.md) | Quantization granularity | Yes | No |

`Target` attribute group defines attribute usage during model transformation for the best performance:

- `Precision` - the attribute defines the most optimal output port precision.
- `Quantization interval` - the attribute defines quantization interval.
- `Quantization alignment` - the attribute defines quantization granularity in runtime: per-channel or per-tensor quantization.
- `Quantization granularity` - the attribute is set by plugin to define quantization granularity: per-channel or per-tensor quantization.

`Required` attribute group defines if attribute usage is required to get an optimal model during transformation:

- `Yes` - the attribute is used by all OpenVINO plugins for low-precision optimization.
- `No` - the attribute is used in a specific OpenVINO plugin.

`Mutable` attribute group defines if transformation can update an existing attribute:

- `Yes` - the attribute can be updated by the next transformations in the pipeline. But attribute update order is still important.
- `No` - existing attribute can not be updated by the next transformation. Previous handled transformation has optimized a model according to the current value.

`FakeQuantize` decomposition is a mandatory part of low precision transformations. Attributes used during decomposition are mandatory. Optional attributes are required only for certain operations.

Attributes usage by transformations:

| Attribute name | Created by transformations | Used by transformations |
| --- | --- | --- |
| PrecisionPreserved | MarkupPrecisions, MarkupAvgPoolPrecisionPreserved | AlignQuantizationIntervals, AlignQuantizationParameters, FakeQuantizeDecompositionTransformation, MarkupAvgPoolPrecisionPreserved |
| AvgPoolPrecisionPreserved | MarkupAvgPoolPrecisionPreserved |  |
| Precisions | MarkupCanBeQuantized, MarkupPrecisions | FakeQuantizeDecompositionTransformation |
| PerTensorQuantization | MarkupPerTensorQuantization |  |
| IntervalsAlignment | AlignQuantizationIntervals | FakeQuantizeDecompositionTransformation |
| QuantizationAlignment | AlignQuantizationParameters | FakeQuantizeDecompositionTransformation |

:::note
The same type of attribute instances can be created in different transformations. This approach is the result of the transformation single-responsibility principle. For example, `Precision` attribute instances are created in `MarkupCanBeQuantized` and `MarkupPrecisions` transformations, but the reasons for their creation are different.
:::
