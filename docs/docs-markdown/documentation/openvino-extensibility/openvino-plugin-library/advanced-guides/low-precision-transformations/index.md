---
sidebar_position: 2
sidebar_label: "OpenVINO™ Low Precision Transformations"
description: "Learn about low precision transformations used to infer a quantized model in low precision with the maximum performance on Intel CPU, GPU, and ARM platforms."
---

# OpenVINO™ Low Precision Transformations


## Introduction

Low precision transformations (known as LPT) are a set of OpenVINO transformations, which are combined in one library. The library is mandatory part of OpenVINO to infer quantized model in low precision with the maximum performance on Intel CPU, GPU and ARM platforms. The library includes more than 45 transformations and supports more then 30 operations. Some transformations are mandatory, some of them are optional and developed for specific device.

The goal of Low Precision Transformations (LPT) is to transform a quantized model from its original precision (FP16 or FP32) to a low precision (INT8: `signed int8` or `unsigned int8`), so that it is prepared for low precision inference in OpenVINO™ plugin. It is achieved by two main principles:

1. `FakeQuantize` operation decomposition to two parts:
   
   - part 1: quantize operation - new `FakeQuantize` operation with output quantization intervals in low precision range (signed int8: [-128, 127] or [-127, 127], unsigned int8: [0, 255] or [0, 256]) and with low precision output (`signed int8` or `unsigned int8`).
   - part 2: dequantization operations with low precision input and original precision output.
2. Propagation of the dequantization operation through original model's operations. It is done to avoid dequantization operations before original model operations, thus the quantize operations with low precision output remain before the original model operations.

As result, operation input tensor precisions will be changed from original to low precision and operations can be inferred by OpenVINO™ plugin in low precision.

For a more detailed description on how to quantize a model, see the [Low precision tools](#low-precision-tools) section below. For more information about model quantization, refer to **Brief History of Lower Precision in Deep Learning** section in [this whitepaper](https://www.intel.com/content/dam/develop/external/us/en/documents/lower-numerical-precision-deep-learning-jan2018-754765.pdf).

## Input model requirements

LPT transformations propagate dequantization operations through the following operations:

- [Add-1](../../../../openvino-ir-format/operation-sets/operation-specs/arithmetic/add-1.md)
- [AvgPool-1](../../../../openvino-ir-format/operation-sets/operation-specs/pooling/avg-pool-1.md)
- [Clamp-1](../../../../openvino-ir-format/operation-sets/operation-specs/activation/clamp-1.md)
- [Concat-1](../../../../openvino-ir-format/operation-sets/operation-specs/movement/concat-1.md)
- [Convolution-1](../../../../openvino-ir-format/operation-sets/operation-specs/convolution/convolution-1.md)
- [ConvolutionBackpropData-1](../../../../openvino-ir-format/operation-sets/operation-specs/convolution/convolution-backprop-data-1.md)
- [DepthToSpace-1](../../../../openvino-ir-format/operation-sets/operation-specs/movement/depth-to-space-1.md)
- [FakeQuantize-1](../../../../openvino-ir-format/operation-sets/operation-specs/quantization/fake-quantize-1.md)
- [GroupConvolution-1](../../../../openvino-ir-format/operation-sets/operation-specs/convolution/group-convolution-1.md)
- [Interpolate-1](../../../../openvino-ir-format/operation-sets/operation-specs/image/interpolate-1.md)
- [Interpolate-4](../../../../openvino-ir-format/operation-sets/operation-specs/image/interpolate-4.md)
- [MatMul-1](../../../../openvino-ir-format/operation-sets/operation-specs/matrix/matmul-1.md)
- [MaxPool-1](../../../../openvino-ir-format/operation-sets/operation-specs/pooling/max-pool-1.md)
- [Multiply-1](../../../../openvino-ir-format/operation-sets/operation-specs/arithmetic/multiply-1.md)
- [MVN-1](../../../../openvino-ir-format/operation-sets/operation-specs/normalization/mvn-1.md)
- [NormalizeL2-1](../../../../openvino-ir-format/operation-sets/operation-specs/normalization/normalize-l2-1.md)
- [PRelu-1](../../../../openvino-ir-format/operation-sets/operation-specs/activation/prelu-1.md)
- [ReduceMax-1](../../../../openvino-ir-format/operation-sets/operation-specs/reduction/reduce-max-1.md)
- [ReduceMean-1](../../../../openvino-ir-format/operation-sets/operation-specs/reduction/reduce-mean-1.md)
- [ReduceMin-1](../../../../openvino-ir-format/operation-sets/operation-specs/reduction/reduce-min-1.md)
- [ReduceSum-1](../../../../openvino-ir-format/operation-sets/operation-specs/reduction/reduce-sum-1.md)
- [Relu-1](../../../../openvino-ir-format/operation-sets/operation-specs/activation/relu-1.md)
- [Reshape-1](../../../../openvino-ir-format/operation-sets/operation-specs/shape/reshape-1.md)
- [Split-1](../../../../openvino-ir-format/operation-sets/operation-specs/movement/split-1.md)
- [Squeeze-1](../../../../openvino-ir-format/operation-sets/operation-specs/shape/reshape-1.md)
- [StridedSlice-1](../../../../openvino-ir-format/operation-sets/operation-specs/movement/strided-slice-1.md)
- [Transpose-1](../../../../openvino-ir-format/operation-sets/operation-specs/movement/transpose-1.md)
- [Gather-7](../../../../openvino-ir-format/operation-sets/operation-specs/movement/gather-7.md)
- [Gather-8](../../../../openvino-ir-format/operation-sets/operation-specs/movement/gather-8.md)
- [Unsqueeze-1](../../../../openvino-ir-format/operation-sets/operation-specs/shape/unsqueeze-1.md)
- [VariadicSplit-1](../../../../openvino-ir-format/operation-sets/operation-specs/movement/variadic-split-1.md)

If operation is not supported by LPT then dequantization operation will not be propagated, input tensor precisions will not be changed to low precision and operation will be executed in original precision.

For example, if you would like to infer a model with `Convolution` operation in low precision then the model can look as on picture below:

![Quantized Convolution](model_fq_and_convolution.common.svg)

There are several supported quantization approaches on activations and on weights. All supported approaches are described in [Quantization approaches](#quantization-approaches) section below. In demonstrated model [FakeQuantize operation quantization](#fakequantize-operation) approach is used.

### Low precision tools

For more details on how to get a quantized model, refer to [Model Optimization](../../../../../openvino-workflow/model-optimization/index.md) document.

## Quantization approaches

LPT transformations support two quantization approaches:

1. `FakeQuantize` operation,
2. Quantize and dequantization operations

Let's explore both approaches in details on `Convolution` operation.

### FakeQuantize operation

In this case `FakeQuantize` operation is used on activations and quantized constant on weights. Original input model:

![Original model with FakeQuantize](model_fq_and_convolution.common.svg)

### Quantize and dequantization operations

In this case `FakeQuantize` operation and `Convert` are used as quantize operation and return quantized low precision tensor. After quantize operation on activations there are `Convert` and dequantization operations to compensate decomposition. Original input model:

![Original model with Q/DQ](model_qdq_and_convolution.common.svg)

In both cases result is the same. In LPT result model you can see that:

1. if necessary, `FakeQuantize` operations on activations were decomposed to two part:
   
   - new `FakeQuantize` operation with updated output intervals in low precision range and low precision output,
   - dequantization operations on activations;
2. if necessary, an existing `FakeQuantize` decomposition can be reworked to get better precision;
3. dequantization operations were propagated through `Convolution`.

LPT result model:

![Result model](model_fq_and_convolution.transformed.svg)

### Low precision transformations pipeline

LPT transformation pipeline has several steps. For each transformation inside one step pattern matcher is unique per transformation, but each operation can be assigned to several transformations.

![Low precision transformations pipeline](low_precision_transformation_pipeline.svg)

Inside each step LPT transformations handle input model operation by operation, applying transformation matching pattern for each transformation from the step to an operation, and execute transformation if pattern is matched. Decomposition transformation decomposes `FakeQuantize` to quantize and dequantization operations. Dequantization operations from previous transformation result is used for the current one and so on, until the end of the model is achieved.

As result, usually all operations are inferred by plugin in low precision. If plugin doesn't support an operation inference in low precision, then corresponding LPT transformation can be disabled, and input tensor precisions for the operation will not be changed. In this case the operation is inferred in the original precision.

Low precision transformations pipeline includes four steps:

- [Step 1: Prerequisites](step1-prerequisites/index.md)
- [Step 2: Markup transformations](step2-markup/index.md)
- [Step 3: Main transformations](step3-main/index.md)
- [Step 4: Cleanup transformations](step4-cleanup/index.md)

#### Step 1. Prerequisites

This step fuses and propagates some operations in the model to prepare for the next step. It is required for OpenVINO plugins. Transformations:

- [PullReshapeThroughDequantization](step1-prerequisites/pull-reshape-through-dequantization.md)
- [PullTransposeThroughDequantization](step1-prerequisites/pull-transpose-through-dequantization.md)
- [LinOpSequenceFusion](step1-prerequisites/lin-op-sequence-fusion.md)

The model on this step is changed. There are more details in developer guide [Prerequisites transformations](step1-prerequisites/index.md).

#### Step 2. Markup

This step creates runtime attributes for operations. These attributes will be used in next step. Transformations:

- [MarkupBias](step2-markup/markup-bias.md)
- [MarkupCanBeQuantized](step2-markup/markup-can-be-quantized.md)
- [MarkupPrecisions](step2-markup/markup-precisions.md)
- [MarkupPerTensorQuantization](step2-markup/markup-per-tensor-quantization.md)
- [MarkupAvgPoolPrecisionPreserved](step2-markup/markup-avg-pool-precision-preserved.md)
- [PropagatePrecisions](step2-markup/propagate-precisions.md)
- [AlignQuantizationIntervals](step2-markup/align-quantization-intervals.md)
- [AlignQuantizationParameters](step2-markup/align-quantization-parameters.md)

The model on this step is changed: only new attributes are added to some operations. There are more details in developer guide [Markup transformations](step2-markup/index.md).

#### Step 3. Main transformations, FakeQuantize decomposition and dequantization operations handling

This step has the most transformations. These transformations can be separated in two groups: decomposition transformation and dequantization operations handling. There are more details in developer guide [Main transformations](step3-main/index.md).

Transformations:

- [AddTransformation](step3-main/arithmetic/add.md)
- [AvgPoolTransformation](step3-main/pooling/avg-pool.md)
- [ClampTransformation](step3-main/pooling/avg-pool.md)
- [BatchToSpaceTransformation](step3-main/shape/batch-to-space.md)
- [ConcatTransformation](step3-main/movement/concat.md)
- [ConvolutionTransformation](step3-main/convolution/convolution.md)
- [ConvolutionBackpropDataTransformation](step3-main/convolution/convolution-backprop-data.md)
- [DepthToSpaceTransformation](step3-main/movement/depth-to-space.md)
- [FakeQuantizeDecompositionTransformation](step4-cleanup/fake-quantize-decomposition.md)
- [FakeQuantizeTransformation](step3-main/quantization/fake-quantize.md)
- [InterpolateTransformation](step3-main/image/interpolate.md)
- [GroupConvolutionTransformation](step3-main/convolution/group-convolution.md)
- [GatherTransformation](step3-main/movement/gather.md)
- [MatMulTransformation](step3-main/matrix/mat-mul.md)
- [MaxPoolTransformation](step3-main/pooling/max-pool.md)
- [MultiplyPartialTransformation](step3-main/arithmetic/multiply-partial.md)
- [MVNTransformation](step3-main/normalization/mvn.md)
- [NormalizeL2Transformation](step3-main/normalization/normalize-l2.md)
- [PReluTransformation](step3-main/activation/prelu.md)
- [ReduceMaxTransformation](step3-main/reduction/reduce-max.md)
- [ReduceMeanTransformation](step3-main/reduction/reduce-mean.md)
- [ReduceMinTransformation](step3-main/reduction/reduce-min.md)
- [ReduceSumTransformation](step3-main/reduction/reduce-sum.md)
- [ReluTransformation](step3-main/activation/relu.md)
- [ReshapeTransformation](step3-main/shape/reshape.md)
- [SqueezeTransformation](step3-main/shape/squeeze.md)
- [ShuffleChannelsTransformation](step3-main/movement/shuffle-channels.md)
- [SpaceToBatchTransformation](step3-main/shape/space-to-batch.md)
- [SplitTransformation](step3-main/movement/split.md)
- [StridedSliceTransformation](step3-main/movement/strided-slice.md)
- [TransposeTransformation](step3-main/movement/transpose.md)
- [UnsqueezeTransformation](step3-main/shape/unsqueeze.md)
- [VariadicSplitTransformation](step3-main/movement/variadic-split.md)

##### Decomposition transformations

Decomposition transformations decompose the `FakeQuantize` operation to: quantize (`FakeQuantize` with low precision output) and dequantization operations (opposite to quantize, with low precision input and the original precision output). For dequantization operations LPT uses three operations: `Convert`, `Subtract` and `Multiply`. Element-wise operations `Subtract` and `Multiply` have constants on the second branches. If dequantization operations are not handled at the end of LPT pipeline, then they will be fused back to the `FakeQuantize`.

Original `FakeQuantize`:

![FakeQuantize operation before LPT](fq.common.svg)

`FakeQuantize` after decomposition to quantization and dequantization operations:

![FakeQuantize operation after LPT](fq.transformed.svg)

##### Dequantization operations handling transformations

In this step, LPT transformations fuse dequantization operations or move them through existing model operations as much as possible.

Original `Convolution` operation in FP32 with dequantization operations before:

![Convolution operation before LPT](model_fq_and_convolution.common.svg)

`Convolution` operation in INT8 after decomposition and dequantization operations handling:

![Convolution operation after LPT](model_fq_and_convolution.transformed.svg)

#### Step 4: Cleanup of the result model

LPT cleanup transformations is final stage in LPT pipeline. In this step LPT transformations clean up the result model to avoid not handled dequantization operations: fuse dequantization operations if possible (fuse at least `Convert` operations if not\` to other model operations to cleanup result model).

Transformations:

- [EliminateFakeQuantizeTransformation](step4-cleanup/eliminate-fake-quantize.md)
- [FoldConvertTransformation](step4-cleanup/fold-convert.md)
- [FoldFakeQuantizeTransformation](step3-main/quantization/fold-fake-quantize.md)
- [FuseConvertTransformation](step4-cleanup/fuse-convert.md)
- [FuseMultiplyToFakeQuantizeTransformation](step4-cleanup/fuse-multiply-to-fake-quantize.md)
- [FuseSubtractToFakeQuantizeTransformation](step4-cleanup/fuse-subtract-to-fake-quantize.md)
- [MultiplyToGroupConvolutionTransformation](step4-cleanup/multiply-to-group-convolution.md)

There are more details in developer guide [Cleanup transformations](step4-cleanup/index.md).

`FakeQuantize` operation with not handled dequantization operations:

![TODO: FakeQuantize operation with dequantization operations before LPT](fq.transformed.svg)

`FakeQuantize` operation with fused dequantization operations:

![TODO: FakeQuantize operation with fused operations after LPT](fq.common.svg)

## Low precision transformations in plugin transformation pipeline

Typical transformation pipeline described below.

### Step 1. Common optimizations

This step is optional for LPT but typically is presented in OpenVINO™ plugins. The step doesn't use any LPT transformation. Firstly, the step disables dequantization operations constant folding on constant subgraph on weights to prevent the lost of dequantization info on the next plugin transformations. After that, it optimizes the transformation function and converts operations to operation set 1. Typically, usage of this step is the simplest way to meet LPT requirements for the input quantized model. If plugin can guarantee that LPT input requirements are met, then this step can be skipped.

```cpp
// check if the function is quantized to ignore LPT transformations for not quantized function to speed up model loading
const bool useLpt = ov::pass::low_precision::LowPrecision::isFunctionQuantized(model);
auto defaultPrecisions =
    useLpt ? ov::pass::low_precision::precision_set::get_int8_support() : std::vector<ov::element::Type>{};
if (useLpt) {
    // disable constant folding on dequantization subgraphs so they can be processed by LPT
    manager.register_pass<ov::pass::MarkDequantization>(defaultPrecisions);
}

// OpenVINO common transformations happen here

if (useLpt) {
    // convert subtract constant to INT8 to prevent unnecessary FP16 to FP32 conversion
    manager.register_pass<ov::pass::low_precision::ConvertSubtractConstant>(defaultPrecisions);
}

// OpenVINO common transformations happen here

if (useLpt) {
    // convert not supported cases FakeQuantize -> Convert -> Convert -> Subtract -> Multiply to a single FakeQuantize
    pass_config->set_callback<ov::pass::ConvertQuantizeDequantize>([&defaultPrecisions](const std::shared_ptr<const ov::Node> &node) -> bool {
        return ov::pass::low_precision::NetworkHelper::areQuantizeAndDequantizeSupportedForMultiply(node, defaultPrecisions);
    });

    // convert not supported cases FakeQuantize -> Convert -> Convert -> Subtract -> Multiply to a single FakeQuantize
    pass_config->set_callback<ov::pass::ConvertSubtract>([&defaultPrecisions](const std::shared_ptr<const ov::Node> &node) -> bool {
        return ov::pass::low_precision::NetworkHelper::areQuantizeAndDequantizeSupportedForSubtract(node, defaultPrecisions);
    });
}

manager.run_passes(model);
```

### Step 2. Low precision transformations execution

This step is mandatory. It configures and runs LPT transformations.

```cpp
using namespace ov::pass::low_precision;
if (useLpt) {
    // Low precision transformations plugin specific configuration: restrictions definition
    auto supportedPrecisions = std::vector<PrecisionsRestriction>({
        PrecisionsRestriction::create<ov::opset1::Convolution>({
            {{0}, {ov::element::u8}},
            {{1}, {ov::element::i8}},
        }),
        PrecisionsRestriction::create<ov::opset1::ConvolutionBackpropData>(
            {{{0}, {ov::element::u8, ov::element::i8}}, {{1}, {ov::element::i8}}}),
        PrecisionsRestriction::create<ov::opset1::GroupConvolution>(
            {{{0}, {ov::element::u8}}, {{1}, {ov::element::i8}}}),
        PrecisionsRestriction::create<ov::opset1::Multiply>({
            {{0}, {ov::element::u8}},
            {{1}, {ov::element::i8}},
        }),
    });

    // Low precision transformations plugin specific configuration: per-tensor quantization operations definition
    auto perTensorQuantization = std::vector<QuantizationGranularityRestriction>({
        QuantizationGranularityRestriction::create<ov::opset1::Convolution>({0}),
        QuantizationGranularityRestriction::create<ov::opset1::ConvolutionBackpropData>({0})
    });

    // Low precision transformations instantiation and registration in pass manager
    ov::pass::Manager lptManager;
    lptManager.register_pass<ov::pass::low_precision::LowPrecision>(supportedPrecisions, perTensorQuantization);

    // Low precision transformations plugin specific configuration: transformation callbacks definition
    lptManager.get_pass_config()->set_callback<MarkupPrecisions>([](const std::shared_ptr<const ov::Node>& node) -> bool {
        if (const auto multiply = ov::as_type_ptr<const ov::opset1::Multiply>(node)) {
            return !MultiplyToGroupConvolutionTransformation::canBeTransformedToGroupConvolution(multiply);
        }
        return false;
    });
    lptManager.get_pass_config()->set_callback<ConvolutionBackpropDataTransformation>([&defaultPrecisions](const std::shared_ptr<const ov::Node>& node) -> bool {
        return LayerTransformation::isAsymmetricQuantization(node, defaultPrecisions) || WeightableLayerTransformation::isAsymmetricOnWeights(node);
    });
    lptManager.get_pass_config()->set_callback<MultiplyToGroupConvolutionTransformation>([](const std::shared_ptr<const ov::Node>& node) -> bool {
        return MultiplyToGroupConvolutionTransformation::isDynamicOrScalar(node);
    });

    // Low precision transformations execution
    lptManager.run_passes(model);
}
```

### Step 3. Plugin-specific transformations

This step is optional. It modifies the transformation function to a device-specific operation set.

```cpp
ov::pass::Manager deviceSpecificManager;
deviceSpecificManager.register_pass<ov::pass::device::ConvertOpSet1ToDeviceSpecific>();
deviceSpecificManager.run_passes(model);
```

## Result model overview

Let's explore the resnet-50-tf model, quantized to `fp16`, which is a TensorFlow
implementation of [ResNet-50](https://github.com/tensorflow/models/tree/v2.2.0/official/r1/resnet)
- an image classification model pre-trained on the ImageNet dataset. Originally
redistributed in the "Saved model" format, converted to a frozen graph using the
"tf.graph\_util" module.

### Inference

The simplest way to infer the model and collect performance counters is [Benchmark Application](../../../../../get-started/learn-openvino/openvino-samples/benchmark-tool.md).

```bash
./benchmark_app -m resnet-50-tf.xml -d CPU -niter 1 -api sync -report_type average_counters  -report_folder pc_report_dir
```

If you infer the model with the OpenVINO™ CPU plugin and collect performance counters, all operations (except last not quantized SoftMax) are executed in INT8 precision.

### Results analysis

Result model depends on different factors:

- The original model quantization possibility and quantization quality. For some models, some operations are not possible to be quantized by NNCF tool. In this case `FakeQuantize` operations are absent before these operations and they will be inferred in original precision.
- LPT customization and plugin supported operations. If plugin doesn't support INT8 inference for some operation then corresponding LPT transformation should be disabled and the operation will be inferred in original precision.

Information about layer precision is stored in the performance counters that are
available from the OpenVINO Runtime API. For example, the part of performance counters table for the resnet-50-tf model inferred on CPU Plugin looks as follows:

| layerName | execStatus | layerType | execType | realTime (ms) | cpuTime (ms) |
| --- | --- | --- | --- | --- | --- |
| resnet\_model/batch\_normalization\_15/FusedBatchNorm/Add | EXECUTED | Convolution | jit\_avx512\_1x1\_I8 | 0.377 | 0.377 |
| resnet\_model/conv2d\_16/Conv2D/fq\_input\_0 | NOT\_RUN | FakeQuantize | undef | 0 | 0 |
| resnet\_model/batch\_normalization\_16/FusedBatchNorm/Add | EXECUTED | Convolution | jit\_avx512\_I8 | 0.499 | 0.499 |
| resnet\_model/conv2d\_17/Conv2D/fq\_input\_0 | NOT\_RUN | FakeQuantize | undef | 0 | 0 |
| resnet\_model/batch\_normalization\_17/FusedBatchNorm/Add | EXECUTED | Convolution | jit\_avx512\_1x1\_I8 | 0.399 | 0.399 |
| resnet\_model/add\_4/fq\_input\_0 | NOT\_RUN | FakeQuantize | undef | 0 | 0 |
| resnet\_model/add\_4 | NOT\_RUN | Eltwise | undef | 0 | 0 |
| resnet\_model/add\_5/fq\_input\_1 | NOT\_RUN | FakeQuantize | undef | 0 | 0 |

The `execStatus` column of the table includes possible values:

- `EXECUTED` - layer was executed by standalone primitive,
- `NOT_RUN` - layer was not executed by standalone primitive or was fused with another operation and executed in another layer primitive.

The `execType` column of the table includes inference primitives with specific suffixes. The layers have the following marks:

- Suffix `I8` for layers that had 8-bit data type input and were computed in 8-bit precision
- Suffix `FP32` for layers computed in 32-bit precision

As result all operations (except not quantized `SoftMax` at the end of the model) in OpenVINO™ CPU plugin are inferred in low precision. Note, please, in the result model there are `FakeQuantize` operations in FP32 but the plugin responsibility is fuse these operations with previous operations. OpenVINO™ CPU plugin achieves maximum optimized inference for all operations by fusing INT8 `Convolution` with FP32 output with `FakeQuantize` operation with FP32 input and INT8 output. In this case OpenVINO™ CPU plugin uses INT8 and FP32 vectorized instructions but reports about one INT8 kernel usage for inference, which is the most optimized for this case.

## Mixed precision

If LPT input model operation output has `fp16` precision then dequantization computations still occurs in `fp32` precision. This approach is used to avoid accuracy loss in `fp16` arithmetic computations. The ultimate output of the dequantization operation  will have the `fp16` precision, as expected.

## Customization

Low Precision Transformations can be customizable. Built-in customization options:

- operation precision restrictions,
- operation per tensor quantization restrictions,
- update precisions,
- dequantization precision.

### Operation precision restrictions

This option defines precisions which allowed for the operation input ports. The option value is passed as input argument for `LowPrecision` constructor. For example:

```cpp
auto supportedPrecisions = std::vector<PrecisionsRestriction>({
    PrecisionsRestriction::create<ov::opset1::Convolution>({
        {{0}, {ov::element::u8}},
        {{1}, {ov::element::i8}},
    }),
});

ov::pass::Manager lptManager;
lptManager.register_pass<ov::pass::low_precision::LowPrecision>(supportedPrecisions);
lptManager.run_passes(model);
```

In provided example in result model `Convolution` operation inputs must have specific precisions: `u8` (unsigned int8) precision on input 0 (on activations) and `i8` (signed int8) precision on input 1 (on weights).

### Operation per tensor quantization restrictions

This option defines if operation supports per-tensor quantization only. The option value is passed as input argument for `LowPrecision` constructor. For example:

```cpp
using namespace ov::pass::low_precision;

const std::vector<PrecisionsRestriction> emptyRestrictions;

auto perTensorQuantization = std::vector<QuantizationGranularityRestriction>({
    QuantizationGranularityRestriction::create<ov::opset1::Convolution>({0})
});

ov::pass::Manager lptManager;
lptManager.register_pass<ov::pass::low_precision::LowPrecision>(emptyRestrictions, perTensorQuantization);
lptManager.run_passes(model);
```

In provided example in result model `Convolution` operations must have per-tensor quantization on input 0 (on activations).

### Update precisions

This option defines if each LPT transformation updates precision or not. The option value is boolean and is passed as `updatePrecisions` member of `LayerTransformation::Params` which is input argument for `LowPrecision` constructor. All transformations are affected. If `true` then low precision transformations update precisions to low precision and doesn't if `false`. Typically this option is used for plugin debugging.

### Typical customization use cases

Plugin specific customization can be implemented via transformation callbacks. For example: asymmetric quantization support can be easily customizable via `LayerTransformation::isAsymmetricQuantization` and `WeightableLayerTransformation::isAsymmetricOnWeights` methods usage in callbacks. For example:

```cpp
using namespace ov::pass::low_precision;
ov::pass::Manager lptManager;

lptManager.register_pass<ov::pass::low_precision::LowPrecision>();
lptManager.get_pass_config()->set_callback<ConvolutionBackpropDataTransformation>([&defaultPrecisions](const std::shared_ptr<const ov::Node>& node) -> bool {
    return LayerTransformation::isAsymmetricQuantization(node, defaultPrecisions) || WeightableLayerTransformation::isAsymmetricOnWeights(node);
});
lptManager.run_passes(model);
```
