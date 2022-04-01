# Step 3. Main Transformations {#openvino_docs_OV_UG_lpt_step3_main}

Main transformations are the majority of low precision transformations. Transformations operate with dequantization operations. Main transformations include:
* [AddTransformation](@ref openvino_docs_OV_UG_lpt_AddTransformation)
* [AvgPoolTransformation](@ref openvino_docs_OV_UG_lpt_AvgPoolTransformation)
* [ClampTransformation](@ref openvino_docs_OV_UG_lpt_AvgPoolTransformation)
* [ConcatTransformation](@ref openvino_docs_OV_UG_lpt_ConcatTransformation)
* [ConvolutionTransformation](@ref openvino_docs_OV_UG_lpt_ConvolutionTransformation)
* [ConvolutionBackpropDataTransformation](@ref openvino_docs_OV_UG_lpt_ConvolutionBackpropDataTransformation)
* [DepthToSpaceTransformation](@ref openvino_docs_OV_UG_lpt_DepthToSpaceTransformation)
* [FakeQuantizeDecompositionTransformation](@ref openvino_docs_OV_UG_lpt_FakeQuantizeDecompositionTransformation)
* [FakeQuantizeTransformation](@ref openvino_docs_OV_UG_lpt_FakeQuantizeTransformation)
* [InterpolateTransformation](@ref openvino_docs_OV_UG_lpt_InterpolateTransformation)
* [GroupConvolutionTransformation](@ref openvino_docs_OV_UG_lpt_GroupConvolutionTransformation)
* [MatMulTransformation](@ref openvino_docs_OV_UG_lpt_MatMulTransformation)
* [MaxPoolTransformation](@ref openvino_docs_OV_UG_lpt_MaxPoolTransformation)
* [MultiplyTransformation](@ref openvino_docs_OV_UG_lpt_MultiplyTransformation)
* [MVNTransformation](@ref openvino_docs_OV_UG_lpt_MVNTransformation)
* [NormalizeL2Transformation](@ref openvino_docs_OV_UG_lpt_NormalizeL2Transformation)
* [PReluTransformation](@ref openvino_docs_OV_UG_lpt_PReluTransformation)
* [ReduceMaxTransformation](@ref openvino_docs_OV_UG_lpt_ReduceMaxTransformation)
* [ReduceMeanTransformation](@ref openvino_docs_OV_UG_lpt_ReduceMeanTransformation)
* [ReduceMinTransformation](@ref openvino_docs_OV_UG_lpt_ReduceMinTransformation)
* [ReduceSumTransformation](@ref openvino_docs_OV_UG_lpt_ReduceSumTransformation)
* [ReluTransformation](@ref openvino_docs_OV_UG_lpt_ReluTransformation)
* [ReshapeTransformation](@ref openvino_docs_OV_UG_lpt_ReshapeTransformation)
* [SqueezeTransformation](@ref openvino_docs_OV_UG_lpt_SqueezeTransformation)
* [ShuffleChannelsTransformation](@ref openvino_docs_OV_UG_lpt_ShuffleChannelsTransformation)
* [SplitTransformation](@ref openvino_docs_OV_UG_lpt_SplitTransformation)
* [StridedSliceTransformation](@ref openvino_docs_OV_UG_lpt_StridedSliceTransformation)
* [TransposeTransformation](@ref openvino_docs_OV_UG_lpt_TransposeTransformation)
* [UnsqueezeTransformation](@ref openvino_docs_OV_UG_lpt_UnsqueezeTransformation)
* [VariadicSplitTransformation](@ref openvino_docs_OV_UG_lpt_VariadicSplitTransformation)

Let's explore some main transformations on the example model. Original model:

![Original model](img/step3_original.png)

Result model after main transformations:

![Original model](img/step3_transformed.png)

Changes in the example model after main transformation:
* All `FakeQuantize` operations (`fakeQuantize1`, `fakeQuantize2` and `fakeQuantize3`) were decomposed:
   - original `FakeQuantize` operations were replaced with new operations with other output intervals and output port precision,
   - dequantization operations.
* Dequantization operations were moved via precision preserved (`concat1` and `concat2`) and quantized (`convolution2`) operations. 

> **Note:** the left branch (branch #1) does not require per-tensor quantization. As a result, the `fakeQuantize1`output interval is [0, 255]. But quantized `convolution2` requires per-tensor quantization on the right branch (branch #2). Then all connected `FakeQuantize` interval operations (`fakeQuantize1` and `fakeQuantize2`) are aligned to have per-tensor quantization after the concatenation (`concat2`) operation.
