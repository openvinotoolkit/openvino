# OpenVINO™ LPT: step #2. Markup transformations {#openvino_docs_IE_DG_lpt_step2_markup}

This step defines the most optimal `FakeQuantize` decomposition precisions for the best inference performance via operations markup with runtime attribute instances. Attributes are created for input & output ports and operations. Operation output port precisions are not changed in these transformations. Transformations order is important. A model markup low precision logic is decomposed and implemented into following common markup transformations (in usage order, order is important):
1. [MarkupCanBeQuantized](@ref openvino_docs_IE_DG_lpt_MarkupCanBeQuantized)
2. [MarkupPrecisions](@ref openvino_docs_IE_DG_lpt_MarkupPrecisions)
3. [MarkupPerTensorQuantization](@ref openvino_docs_IE_DG_lpt_MarkupPerTensorQuantization)
4. [MarkupAvgPoolPrecisionPreserved](@ref openvino_docs_IE_DG_lpt_MarkupAvgPoolPrecisionPreserved)
5. [PropagatePrecisions](@ref openvino_docs_IE_DG_lpt_PropagatePrecisions)
6. [AlignQuantizationIntervals](@ref openvino_docs_IE_DG_lpt_AlignQuantizationIntervals)
7. [AlignQuantizationParameters](@ref openvino_docs_IE_DG_lpt_AlignQuantizationParameters)

<details>
<summary>Click to explore transformations and used attributes in one table</summary>

| Transformation name             | Create attributes             | Use attributes                            |
|---------------------------------|-------------------------------|-------------------------------------------|
| MarkupCanBeQuantized            | Precisions                    |                                           |
| MarkupPrecisions                | Precisions,PrecisionPreserved |                                           |
| MarkupPerTensorQuantization     | PerTensorQuantization         |                                           |
| MarkupAvgPoolPrecisionPreserved | AvgPoolPrecisionPreserved     | Precisions, PrecisionPreserved            |
| PropagatePrecisions             | Precisions                    | Precisions, PrecisionPreserved            |
| AlignQuantizationIntervals      | IntervalsAlignment            | PrecisionPreserved                        |
| AlignQuantizationParameters     | QuantizationAlignment         | PrecisionPreserved, PerTensorQuantization |

</details>

Note, please, the same type attribute instances can be created in different transformations. This approach is result of transformation single-responsibility principle. For example `Precision` attribute instances are created in `MarkupCanBeQuantized` and `MarkupPrecisions` transformations but the creation reason is different.

Common markup transformations can be decomposed to simpler utility markup transformations. Markup utility transformations (order is not important):
* [CreateAttribute](@ref openvino_docs_IE_DG_lpt_CreateAttribute)
* [CreatePrecisionsDependentAttribute](@ref openvino_docs_IE_DG_lpt_CreatePrecisionsDependentAttribute)
* [PropagateThroughPrecisionPreserved](@ref openvino_docs_IE_DG_lpt_PropagateThroughPrecisionPreserved)
* [PropagateToInput](@ref openvino_docs_IE_DG_lpt_PropagateToInput)
* [UpdateSharedPrecisionPreserved](@ref openvino_docs_IE_DG_lpt_UpdateSharedPrecisionPreserved)

Let's explore all transformations and their relations in details on the same model:

![](img/step2_markup_original.png) 

The original model key features:
* The first `concat1` concatenation operation has not quantized `convolution1` consumer.
* The second `concat2` concatenation operation has quantized `convolution2` consumer with requirements: 1) support `unsigned int8` on activations 2) per-tensor quantization.
* Between `concat2` concatenation operation and `Convolution` there is `AvgPool` operation which mathematically have to return `f32` tensor. But `MarkupAvgPoolPrecisionPreserved` transformation is active, which allows low precision after `AvgPool` transformation to propagate low precision tensor to the next consumer. 

Transformations are ran with parameters:

@snippet snippets/lpt_mkldnn_plugin.cpp lpt_markup_pipeline

## 1. MarkupCanBeQuantized
The transformation marks operations which can not be quantized. The transformation doesn't require any attributes before.

Changes in example model after `MarkupCanBeQuantized` transformation:
* Not quantized `convolution1` operation is marked by `Precisions` attribute with empty values. This attribute allows to ignore not quantized operation by the next transformations.

Result model:

![MarkupCanBeQuantized](img/step2_markup1.png)

> Model display features (here and below):
> 1. Added by current transformation attributes are marked in bold.
> 2. If attributes are not fit into one line, then one line consists only one attribute.

## 2. MarkupPrecisions
The transformation is required and include two tasks:
1. Mark operation input ports (create `Precision` attribute instance) by provided restrictions: input port index and required precisions. Restrictions are provided as input argument in `ngraph::pass::low_precision::LowPrecision` constructor. 
2. Mark precision preserved operations. 

The transformation doesn't require any attributes before. Changes in example model after `MarkupPrecisions` transformation:
* Both concatenation operations are marked as precision preserved operation. It allows to propagate precision via these operations.
* Quantized `convolution2` operation is marked by `Precisions` attribute with `u8` precision on activations and `i8` precisions on weights in accordance with provided restrictions. This attribute instance allows to specify which precisions are required for quantized `Convolution` operation.

Result model:

![MarkupPrecisions result](img/step2_markup2.png)

## 3. MarkupPerTensorQuantization
The transformation is required and marks operations (create `PerTensorQuantization` attribute instance) by provided restrictions: operation which requires per-tensor quantization. The transformation doesn't require any attributes before. 
Changes in example model after `MarkupPerTensorQuantization` transformation:
* both `Convolution` operations are marked by `PerTensorQuantization`

Result model:

![MarkupPerTensorQuantization result](img/step2_markup3.png)

## 4. MarkupAvgPoolPrecisionPreserved
The transformation is optional. `MarkupAvgPoolPrecisionPreserved` marks `AvgPool` operations as precision preserved or not precision preserved. `AvgPool` operation is precision preserved if next not precision preserved operation can be inferred in low precision. In other words: `AvgPool` operations became precision preserved operations to speed up model inference. The transformation uses `PrecisionPreserved` attributes created before. The transformation is combined and uses:
* CreatePrecisionsDependentAttribute
* PropagateThroughPrecisionPreserved
* UpdateSharedPrecisionPreserved

Changes in example model after `MarkupAvgPoolPrecisionPreserved` transformation:
* `AvgPool` operations are marked by `PrecisionPreserved` and `AvgPoolPrecisionPreserved` (not used below).

Result model:

![MarkupAvgPoolPrecisionPreserved](img/step2_markup4.png)

## 5. PropagatePrecisions
The transformation is required. `PropagatePrecision` is a key transformation in markup pipeline which marks `FakeQuantize` output port precisions. The transformation uses `PrecisionPreserved` attribute instances which are created before. The transformation is combined and uses:
* CreateAttribute
* PropagateThroughPrecisionPreserved
* PropagateToInput

Changes in example model after `PropagatePrecisions` transformation:
* All precision preserved operations are marked by `Precisions` attribute instance which defines required precision for the operation.
* `FakeQuantize` operation output ports are marked by `Precisions` attribute instances which defines target precision for decomposition. In sample model `FakeQuantize` operations have signed intervals but `Precisions` attributes initialized by `u8` (`unsigned int8`) values as result applied during transformations restrictions for `Convolution` operations.

Result model:

![PropagatePrecisions](img/step2_markup5.png)

> `AlignQuantizationIntervals` and `AlignQuantizationParameters` transformations are required if model has quantized concatenation operations.

## 6. AlignQuantizationIntervals
The transformation is required for models with quantized     operation. The transformation marks `FakeQuantize` operation and precision preserved consumers to combine quantization information from different `FakeQuantize` operations for future quantization intervals alignment. The transformation is combined and uses:
* CreateAttribute
* PropagateThroughPrecisionPreserved

Changes in example model after `AlignQuantizationIntervals` transformation:
* All `FakeQuantize` operation and theirs precision preserved consumers are marked by `IntervalsAlignment` attribute instance.

Result model:

![AlignQuantizationIntervals](img/step2_markup6.png)

## 7. AlignQuantizationParameters
The transformation is required for models with quantized concatenation operation. The transformation marks `FakeQuantize` precision preserved consumers to align quantization intervals. The transformation is combined and uses:
* CreateAttribute
* PropagateThroughPrecisionPreserved
* UpdateSharedPrecisionPreserved


Changes in example model after `AlignQuantizationParameters` transformation:
* All `FakeQuantize` precision preserved consumers are marked by `QuantizationAlignment` attribute instance. `convolution1` input ports are marked by `Precisions` attribute instances with empty precisions collection. As result `convolution1` operation was detected as not quantized and `QuantizationAlignment` attribute default value `false` was not changed. `convolution2` input ports are marked by `Precisions` attribute instances with not empty precisions collection. As result `convolution2` operation was detected as quantized with `PerTensorQuantization` attribute and `QuantizationAlignment` attribute default value was changed to `true`.

Final model:

![AlignQuantizationParameters](img/step2_markup7.png)
