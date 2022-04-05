# Step 2. Markup Transformations {#openvino_docs_OV_UG_lpt_step2_markup}

This step defines the optimal `FakeQuantize` decomposition precisions for the best inference performance via operations markup with runtime attribute instances. Attributes are created for input and output ports and operations. Transformations do not change the operation output port precisions. A model markup low precision logic is decomposed and implemented into the following common markup transformations. The order of transformations is important:

1. [MarkupCanBeQuantized](@ref openvino_docs_OV_UG_lpt_MarkupCanBeQuantized)
2. [MarkupPrecisions](@ref openvino_docs_OV_UG_lpt_MarkupPrecisions)
3. [MarkupPerTensorQuantization](@ref openvino_docs_OV_UG_lpt_MarkupPerTensorQuantization)
4. [MarkupAvgPoolPrecisionPreserved](@ref openvino_docs_OV_UG_lpt_MarkupAvgPoolPrecisionPreserved)
5. [PropagatePrecisions](@ref openvino_docs_OV_UG_lpt_PropagatePrecisions)
6. [AlignQuantizationIntervals](@ref openvino_docs_OV_UG_lpt_AlignQuantizationIntervals)
7. [AlignQuantizationParameters](@ref openvino_docs_OV_UG_lpt_AlignQuantizationParameters)

The table of transformations and used attributes:

| Transformation name             | Create attributes             | Use attributes                            |
|---------------------------------|-------------------------------|-------------------------------------------|
| MarkupCanBeQuantized            | Precisions                    |                                           |
| MarkupPrecisions                | Precisions,PrecisionPreserved |                                           |
| MarkupPerTensorQuantization     | PerTensorQuantization         |                                           |
| MarkupAvgPoolPrecisionPreserved | AvgPoolPrecisionPreserved     | Precisions, PrecisionPreserved            |
| PropagatePrecisions             | Precisions                    | Precisions, PrecisionPreserved            |
| AlignQuantizationIntervals      | IntervalsAlignment            | PrecisionPreserved                        |
| AlignQuantizationParameters     | QuantizationAlignment         | PrecisionPreserved, PerTensorQuantization |

> **Note:** the same type of attribute instances can be created in different transformations. This approach is the result of the transformation single-responsibility principle. For example, `Precision` attribute instances are created in `MarkupCanBeQuantized` and `MarkupPrecisions` transformations, but the reasons for their creation are different

Common markup transformations can be decomposed into simpler utility markup transformations. The order of Markup utility transformations is not important:
* [CreateAttribute](@ref openvino_docs_OV_UG_lpt_CreateAttribute)
* [CreatePrecisionsDependentAttribute](@ref openvino_docs_OV_UG_lpt_CreatePrecisionsDependentAttribute)
* [PropagateThroughPrecisionPreserved](@ref openvino_docs_OV_UG_lpt_PropagateThroughPrecisionPreserved)
* [PropagateToInput](@ref openvino_docs_OV_UG_lpt_PropagateToInput)
* [UpdateSharedPrecisionPreserved](@ref openvino_docs_OV_UG_lpt_UpdateSharedPrecisionPreserved)

Let's explore all transformations and their relations in detail, using one and the same model:

![](img/step2_markup_original.png) 

The original model key features:
* The first `concat1` concatenation operation has not quantized `convolution1` consumer.
* The second `concat2` concatenation operation has quantized `convolution2` consumer with requirements: 
   - support `unsigned int8` on activations,
   - per-tensor quantization.
* Between the `concat2` concatenation operation and `Convolution` there is an `AvgPool` operation, which mathematically should return an `f32` tensor. But the `MarkupAvgPoolPrecisionPreserved` transformation is active. This allows the low precision transformation, that goes after the `AvgPool`, to propagate low precision tensor to the next consumer. 

Transformations are run with the following parameters:

@snippet snippets/lpt_intel_cpu_plugin.cpp lpt_markup_pipeline

## 1. MarkupCanBeQuantized
The transformation marks operations that cannot be quantized. No attributes are required before the transformation.

Changes in the example model after `MarkupCanBeQuantized` transformation:
* Not quantized `convolution1` operation is marked by the `Precisions` attribute with empty values. This attribute allows the next transformation to ignore not quantized operation.

Result model:

![MarkupCanBeQuantized](img/step2_markup1.png)

Model display features (here and below):
* The attributes added by the current transformation are marked in bold.
* If attributes do not fit into one line, then one line consists of only one attribute.

## 2. MarkupPrecisions
The transformation is required and includes two tasks:
1. Mark operation input ports (create `Precision` attribute instance) by provided restrictions: input port index and required precisions. Restrictions are provided as input argument in `ngraph::pass::low_precision::LowPrecision` constructor. 
2. Mark precision preserved operations. 

No attributes are required before the transformation. Changes in the example model after `MarkupPrecisions` transformation:
* Both concatenation operations are marked as precision preserved operations. It allows to propagate precision via these operations.
* Quantized `convolution2` operation is marked by the `Precisions` attribute with `u8` precision on activations and `i8` precisions on weights according to the provided restrictions. This attribute instance allows to specify which precisions are required for quantized `Convolution` operation.

Result model:

![MarkupPrecisions result](img/step2_markup2.png)

## 3. MarkupPerTensorQuantization
The transformation is required and marks operations (create `PerTensorQuantization` attribute instance) by provided restrictions: an operation that requires per-tensor quantization. No attributes are required before the transformation.

Changes in the example model after `MarkupPerTensorQuantization` transformation:
* both `Convolution` operations are marked by `PerTensorQuantization`

Result model:

![MarkupPerTensorQuantization result](img/step2_markup3.png)

## 4. MarkupAvgPoolPrecisionPreserved
The transformation is optional. `MarkupAvgPoolPrecisionPreserved` marks `AvgPool` operations as precision preserved or not precision preserved. `AvgPool` operation is precision preserved if next not precision preserved operation can be inferred in low precision. In other words, `AvgPool` operations become precision preserved operations to speed up model inference. The transformation uses `PrecisionPreserved` attributes created before. The transformation is combined and uses:
* CreatePrecisionsDependentAttribute
* PropagateThroughPrecisionPreserved
* UpdateSharedPrecisionPreserved

Changes in the example model after `MarkupAvgPoolPrecisionPreserved` transformation:
* `AvgPool` operations are marked by `PrecisionPreserved` and `AvgPoolPrecisionPreserved` (not used below).

Result model:

![MarkupAvgPoolPrecisionPreserved](img/step2_markup4.png)

## 5. PropagatePrecisions
The transformation is required. `PropagatePrecision` is a key transformation in the markup pipeline, which marks `FakeQuantize` output port precisions. The transformation uses `PrecisionPreserved` attribute instances created before. The transformation is combined and uses:

* CreateAttribute
* PropagateThroughPrecisionPreserved
* PropagateToInput

Changes in the example model after `PropagatePrecisions` transformation:
* All precision preserved operations are marked by the `Precisions` attribute instance, which defines the required precision for the operation.
* `FakeQuantize` operation output ports are marked by `Precisions` attribute instances, which define target precision for decomposition. In the sample model, `FakeQuantize` operations have signed intervals, but the `Precisions` attributes are initialized by `u8` (`unsigned int8`) values as the result applied during transformations restrictions for `Convolution` operations.

Result model:

![PropagatePrecisions](img/step2_markup5.png)

> **NOTE**: `AlignQuantizationIntervals` and `AlignQuantizationParameters` transformations are required if the model has quantized concatenation operations.

## 6. AlignQuantizationIntervals
The transformation is required for models with the quantized operation. The transformation marks `FakeQuantize` operation and precision preserved consumers to combine quantization information from different `FakeQuantize` operations for future quantization intervals alignment. The transformation is combined and uses:
* CreateAttribute
* PropagateThroughPrecisionPreserved

Changes in the example model after `AlignQuantizationIntervals` transformation:
* All `FakeQuantize` operations and their precision preserved consumers are marked by the `IntervalsAlignment` attribute instance.

Result model:

![AlignQuantizationIntervals](img/step2_markup6.png)

## 7. AlignQuantizationParameters
The transformation is required for models with quantized concatenation operation. The transformation marks `FakeQuantize` precision preserved consumers to align quantization intervals. The transformation is combined and uses:
* CreateAttribute
* PropagateThroughPrecisionPreserved
* UpdateSharedPrecisionPreserved


Changes in the example model after `AlignQuantizationParameters` transformation:
* All `FakeQuantize` precision preserved consumers are marked by `QuantizationAlignment` attribute instance. `convolution1` input ports are marked by `Precisions` attribute instances with empty precisions collection. As a result, the `convolution1` operation was detected as not quantized, and the `QuantizationAlignment` attribute default value `false` does not change. `convolution2` input ports are marked by `Precisions` attribute instances with not empty precisions collection.  `convolution2` operation was detected as quantized with the `PerTensorQuantization` attribute, and the `QuantizationAlignment` attribute default value changed to `true`.

Final model:

![AlignQuantizationParameters](img/step2_markup7.png)
