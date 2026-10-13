---
sidebar_position: 5
sidebar_label: "Step 3. Main Transformations"
description: "Learn about main transformations, which are mostly low precision transformations that handle decomposition and dequantization operations."
---

# Step 3. Main Transformations


Main transformations are the majority of low precision transformations. Transformations operate with dequantization operations. Main transformations include:

- [AddTransformation](arithmetic/add.md)
- [AvgPoolTransformation](pooling/avg-pool.md)
- [BatchToSpaceTransformation](shape/batch-to-space.md)
- [ClampTransformation](activation/clamp.md)
- [ConcatTransformation](movement/concat.md)
- [ConvolutionTransformation](convolution/convolution.md)
- [ConvolutionBackpropDataTransformation](convolution/convolution-backprop-data.md)
- [DepthToSpaceTransformation](movement/depth-to-space.md)
- [FakeQuantizeDecompositionTransformation](../step4-cleanup/fake-quantize-decomposition.md)
- [FakeQuantizeTransformation](quantization/fake-quantize.md)
- [InterpolateTransformation](image/interpolate.md)
- [GroupConvolutionTransformation](convolution/group-convolution.md)
- [GatherTransformation](movement/gather.md)
- [MatMulTransformation](matrix/mat-mul.md)
- [MaxPoolTransformation](pooling/max-pool.md)
- [MultiplyPartialTransformation](arithmetic/multiply-partial.md)
- [MultiplyTransformation](arithmetic/multiply.md)
- [MVNTransformation](normalization/mvn.md)
- [NormalizeL2Transformation](normalization/normalize-l2.md)
- [PadTransformation](movement/pad.md)
- [PReluTransformation](activation/prelu.md)
- [ReduceMaxTransformation](reduction/reduce-max.md)
- [ReduceMeanTransformation](reduction/reduce-mean.md)
- [ReduceMinTransformation](reduction/reduce-min.md)
- [ReduceSumTransformation](reduction/reduce-sum.md)
- [ReluTransformation](activation/relu.md)
- [ReshapeTransformation](shape/reshape.md)
- [SpaceToBatchTransformation](shape/space-to-batch.md)
- [SqueezeTransformation](shape/squeeze.md)
- [ShuffleChannelsTransformation](movement/shuffle-channels.md)
- [SplitTransformation](movement/split.md)
- [StridedSliceTransformation](movement/strided-slice.md)
- [TransposeTransformation](movement/transpose.md)
- [UnsqueezeTransformation](shape/unsqueeze.md)
- [VariadicSplitTransformation](movement/variadic-split.md)

Let's explore some main transformations on the example model. Original model:

![Original model](step3_original.svg)

Result model after main transformations:

![Transformed model](step3_transformed.svg)

Changes in the example model after main transformation:

- All `FakeQuantize` operations (`fakeQuantize1`, `fakeQuantize2` and `fakeQuantize3`) were decomposed:
  
  - original `FakeQuantize` operations were replaced with new operations with other output intervals and output port precision,
  - dequantization operations.
- Dequantization operations were moved via precision preserved (`concat1` and `concat2`) and quantized (`convolution2`) operations.

:::note
The left branch (branch #1) does not require per-tensor quantization. As a result, the `fakeQuantize1` output interval is [0, 255]. But quantized `convolution2` requires per-tensor quantization on the right branch (branch #2). Then all connected `FakeQuantize` interval operations (`fakeQuantize1` and `fakeQuantize2`) are aligned to have per-tensor quantization after the concatenation (`concat2`) operation.
:::
