// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/common_optimizations.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/common_optimizations/add_fake_quantize_fusion.hpp"
#include "transformations/common_optimizations/batch_to_space_fusion.hpp"
#include "transformations/common_optimizations/binarize_weights.hpp"
#include "transformations/common_optimizations/broadcast_elementwise_fusion.hpp"
#include "transformations/common_optimizations/clamp_fusion.hpp"
#include "transformations/common_optimizations/concat_fusion.hpp"
#include "transformations/common_optimizations/concat_reduce_fusion.hpp"
#include "transformations/common_optimizations/conv_mul_fusion.hpp"
#include "transformations/common_optimizations/conv_to_binary_conv.hpp"
#include "transformations/common_optimizations/convert_nms_gather_path_to_unsigned.hpp"
#include "transformations/common_optimizations/convert_quantize_dequantize.hpp"
#include "transformations/common_optimizations/dilated_convolution_converter.hpp"
#include "transformations/common_optimizations/disable_random_uniform_constant_folding.hpp"
#include "transformations/common_optimizations/dropout_with_random_uniform_replacer.hpp"
#include "transformations/common_optimizations/eliminate_unsqueeze_gather.hpp"
#include "transformations/common_optimizations/fq_mul_fusion.hpp"
#include "transformations/common_optimizations/fq_reshape_fusion.hpp"
#include "transformations/common_optimizations/gelu_fusion.hpp"
#include "transformations/common_optimizations/hsigmoid_fusion.hpp"
#include "transformations/common_optimizations/hswish_fusion.hpp"
#include "transformations/common_optimizations/interpolate_sequence_fusion.hpp"
#include "transformations/common_optimizations/leaky_relu_fusion.hpp"
#include "transformations/common_optimizations/lin_op_sequence_fusion.hpp"
#include "transformations/common_optimizations/mark_precision_sensitive_shapeof_subgraphs.hpp"
#include "transformations/common_optimizations/matmul_multiply_fusion.hpp"
#include "transformations/common_optimizations/moc_transformations.hpp"
#include "transformations/common_optimizations/mul_conv_fusion.hpp"
#include "transformations/common_optimizations/mul_fake_quantize_fusion.hpp"
#include "transformations/common_optimizations/mvn_fusion.hpp"
#include "transformations/common_optimizations/nop_elimination.hpp"
#include "transformations/common_optimizations/normalize_l2_fusion.hpp"
#include "transformations/common_optimizations/optimize_strided_slice.hpp"
#include "transformations/common_optimizations/pad_fusion.hpp"
#include "transformations/common_optimizations/pull_transpose_through_fq.hpp"
#include "transformations/common_optimizations/random_uniform_fusion.hpp"
#include "transformations/common_optimizations/reduce_merge.hpp"
#include "transformations/common_optimizations/relu_fake_quantize_fusion.hpp"
#include "transformations/common_optimizations/remove_filtering_boxes_by_size.hpp"
#include "transformations/common_optimizations/simplify_shape_of_sub_graph.hpp"
#include "transformations/common_optimizations/skip_gather_before_transpose_and_reshape.hpp"
#include "transformations/common_optimizations/softmax_fusion.hpp"
#include "transformations/common_optimizations/softplus_fusion.hpp"
#include "transformations/common_optimizations/softplus_to_mish_fusion.hpp"
#include "transformations/common_optimizations/space_to_batch_fusion.hpp"
#include "transformations/common_optimizations/split_squeeze_concat_fusion.hpp"
#include "transformations/common_optimizations/strides_optimization.hpp"
#include "transformations/common_optimizations/swish_fusion.hpp"
#include "transformations/common_optimizations/transpose_reshape_elimination_for_matmul.hpp"
#include "transformations/common_optimizations/transpose_sinking.hpp"
#include "transformations/common_optimizations/transpose_to_reshape.hpp"
#include "transformations/common_optimizations/weights_dequantize_to_fake_quantize.hpp"
#include "transformations/control_flow/unroll_if.hpp"
#include "transformations/fp16_compression/convert_compression_only_to_legacy.hpp"
#include "transformations/fp16_compression/mark_decompression_convert_constant_folding.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/op_conversions/batch_norm_decomposition.hpp"
#include "transformations/op_conversions/bidirectional_sequences_decomposition.hpp"
#include "transformations/op_conversions/convert_avgpool_downgrade.hpp"
#include "transformations/op_conversions/convert_bitwise_to_logical_bool.hpp"
#include "transformations/op_conversions/convert_broadcast_to_tiles.hpp"
#include "transformations/op_conversions/convert_convertlike.hpp"
#include "transformations/op_conversions/convert_deformable_conv_v8_to_v1.hpp"
#include "transformations/op_conversions/convert_depth_to_space.hpp"
#include "transformations/op_conversions/convert_divide.hpp"
#include "transformations/op_conversions/convert_embedding_bag_offsets15_downgrade.hpp"
#include "transformations/op_conversions/convert_embedding_bag_packed15_downgrade.hpp"
#include "transformations/op_conversions/convert_gather_downgrade.hpp"
#include "transformations/op_conversions/convert_gather_upgrade.hpp"
#include "transformations/op_conversions/convert_gelu.hpp"
#include "transformations/op_conversions/convert_interpolate11_downgrade.hpp"
#include "transformations/op_conversions/convert_interpolate1_to_interpolate4.hpp"
#include "transformations/op_conversions/convert_maxpool_downgrade.hpp"
#include "transformations/op_conversions/convert_maxpool_upgrade.hpp"
#include "transformations/op_conversions/convert_minimum_to_power_and_max.hpp"
#include "transformations/op_conversions/convert_mod.hpp"
#include "transformations/op_conversions/convert_multiclass_nms_upgrade.hpp"
#include "transformations/op_conversions/convert_pad12_downgrade.hpp"
#include "transformations/op_conversions/convert_pad_to_group_conv.hpp"
#include "transformations/op_conversions/convert_prior_box_v8_to_v0.hpp"
#include "transformations/op_conversions/convert_reduce_to_pooling.hpp"
#include "transformations/op_conversions/convert_roi_align_v3_to_v9.hpp"
#include "transformations/op_conversions/convert_roi_align_v9_to_v3.hpp"
#include "transformations/op_conversions/convert_scatter_elements_update12_downgrade.hpp"
#include "transformations/op_conversions/convert_scatter_nd_update15_downgrade.hpp"
#include "transformations/op_conversions/convert_slice_to_strided_slice.hpp"
#include "transformations/op_conversions/convert_slicescatter.hpp"
#include "transformations/op_conversions/convert_softmax_downgrade.hpp"
#include "transformations/op_conversions/convert_softmax_upgrade.hpp"
#include "transformations/op_conversions/convert_space_to_depth.hpp"
#include "transformations/op_conversions/convert_squeeze15_downgrade.hpp"
#include "transformations/op_conversions/convert_subtract.hpp"
#include "transformations/op_conversions/convert_topk11_downgrade.hpp"
#include "transformations/op_conversions/convert_xor_to_logical_xor.hpp"
#include "transformations/op_conversions/detection_output_downgrade.hpp"
#include "transformations/op_conversions/detection_output_upgrade.hpp"
#include "transformations/op_conversions/einsum_decomposition.hpp"
#include "transformations/op_conversions/eye_decomposition.hpp"
#include "transformations/op_conversions/gelu7_downgrade.hpp"
#include "transformations/op_conversions/group_normalization_decomposition.hpp"
#include "transformations/op_conversions/hsigmoid_decomposition.hpp"
#include "transformations/op_conversions/hswish_decomposition.hpp"
#include "transformations/op_conversions/log_softmax_decomposition.hpp"
#include "transformations/op_conversions/mvn6_decomposition.hpp"
#include "transformations/op_conversions/normalize_l2_decomposition.hpp"
#include "transformations/op_conversions/reduce_l1_decomposition.hpp"
#include "transformations/op_conversions/reduce_l2_decomposition.hpp"
#include "transformations/op_conversions/scaled_dot_product_attention_decomposition.hpp"
#include "transformations/op_conversions/simplify_ctc_greedy_decoder_seq_len.hpp"
#include "transformations/op_conversions/softmax_decomposition.hpp"
#include "transformations/op_conversions/softsign_decomposition.hpp"
#include "transformations/op_conversions/unique_decomposition.hpp"
#include "transformations/symbolic_transformations/symbolic_optimizations.hpp"

bool ov::pass::CommonOptimizations::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_FUNCTION_SCOPE(CommonOptimizations);
    ov::pass::Manager manager(get_pass_config(), "CommonOptimizations");
    manager.set_per_pass_validation(false);

    using namespace ov::pass;
    REGISTER_PASS(manager, DisableDecompressionConvertConstantFolding)
    // MOCTransformations contain StridedSliceOptimization transformation,
    // so we must call SliceToStridedSlice before MOCTransformations call
    REGISTER_PASS(manager, SliceToStridedSlice, true)
    // Disable low_precision_enabled as all plugins handle low-precision sub-graph manually
    // before CommonOptimization pipeline execution
    REGISTER_PASS(manager, MOCTransformations, true, false)

    // Enabling conversion of FP16 IR to legacy representation, each plugin have to disable it
    // after support for FP16 IR is implemented
    REGISTER_PASS(manager, ConvertCompressedOnlyToLegacy)

    REGISTER_PASS(manager, MarkDividesInShapeSubgraphs)
    REGISTER_PASS(manager, WeightsDequantizeToFakeQuantize)

    auto common_fusions = manager.register_pass<GraphRewrite>();
    ADD_MATCHER(common_fusions, SpaceToBatchFusion)
    ADD_MATCHER(common_fusions, BatchToSpaceFusion)
    ADD_MATCHER(common_fusions, InterpolateSequenceFusion)
    ADD_MATCHER(common_fusions, SkipGatherBeforeTransposeAndReshape)
    ADD_MATCHER(common_fusions, ReduceMerge)
    common_fusions->set_name("ov::pass::CommonFusions");

    manager.register_pass<ConcatReduceFusion>();
    REGISTER_DISABLED_PASS(manager, ConvertPadToGroupConvolution)
    REGISTER_DISABLED_PASS(manager, ConvertInterpolate1ToInterpolate4)

    auto decomp = manager.register_pass<GraphRewrite>();
    ADD_MATCHER(decomp, ScaledDotProductAttentionDecomposition)
    ADD_MATCHER(decomp, Gelu7Downgrade)
    ADD_MATCHER(decomp, BidirectionalSequenceDecomposition)
    ADD_MATCHER(decomp, ReduceL1Decomposition)
    ADD_MATCHER(decomp, ReduceL2Decomposition)
    ADD_MATCHER(decomp, HSwishDecomposition)
    ADD_MATCHER(decomp, HSigmoidDecomposition)
    ADD_MATCHER(decomp, LogSoftmaxDecomposition)
    ADD_MATCHER(decomp, ConvertReduceToPooling)
    ADD_MATCHER(decomp, ConvertBroadcastToTiles)
    ADD_MATCHER(decomp, ConvertMod)
    ADD_MATCHER(decomp, ConvertGELU)
    ADD_MATCHER(decomp, ConvertMinimum)
    ADD_MATCHER(decomp, ConvertSubtract)
    ADD_MATCHER(decomp, ConvertDivide)
    ADD_MATCHER(decomp, ConvertDepthToSpace)
    ADD_MATCHER(decomp, ConvertSpaceToDepth)
    ADD_MATCHER(decomp, ConvertConvertLike)
    ADD_MATCHER(decomp, BatchNormDecomposition)
    ADD_MATCHER(decomp, GroupNormalizationDecomposition)
    ADD_MATCHER(decomp, MVN6Decomposition)
    decomp->add_matcher<NormalizeL2Decomposition, false>();
    ADD_MATCHER(decomp, SimplifyCTCGreedyDecoderSeqLen)
    ADD_MATCHER(decomp, EinsumDecomposition)
    decomp->add_matcher<SoftmaxDecomposition, false>();
    ADD_MATCHER(decomp, SoftSignDecomposition)
    ADD_MATCHER(decomp, DropoutWithRandomUniformReplacer)
    ADD_MATCHER(decomp, TransposeReshapeEliminationForMatmul)
    ADD_MATCHER(decomp, EyeDecomposition)
    ADD_MATCHER(decomp, UniqueDecomposition)
    decomp->set_name("ov::pass::CommonDecompositions");

    // CF is required after all decompositions
    REGISTER_PASS(manager, ConstantFolding)

    // LinOpSequenceFusion must be executed after all decompositions
    manager.register_pass<LinOpSequenceFusion>();
    REGISTER_PASS(manager, UnrollIf)

    auto multiply_fusions = manager.register_pass<GraphRewrite>();
    ADD_MATCHER(multiply_fusions, ConvolutionMultiplyFusion)
    ADD_MATCHER(multiply_fusions, GroupConvolutionMultiplyFusion)
    ADD_MATCHER(multiply_fusions, ConvolutionBackpropDataMultiplyFusion)
    ADD_MATCHER(multiply_fusions, GroupConvolutionBackpropDataMultiplyFusion)
    ADD_MATCHER(multiply_fusions, MultiplyConvolutionFusion)
    ADD_MATCHER(multiply_fusions, MultiplyGroupConvolutionFusion)
    ADD_MATCHER(multiply_fusions, MultiplyConvolutionBackpropDataFusion)
    ADD_MATCHER(multiply_fusions, MultiplyGroupConvolutionBackpropDataFusion)
    ADD_MATCHER(multiply_fusions, MatMulMultiplyFusion)
    multiply_fusions->set_name("ov::pass::MultiplyFusions");

    REGISTER_PASS(manager, ConstantFolding)
    REGISTER_PASS(manager, ConvertGather8ToGather7)  // not plugins implemented gather8
    REGISTER_PASS(manager, ConvertGather7ToGather1)  // not plugins implemented gather7

    REGISTER_DISABLED_PASS(manager, ConvertGather1ToGather7)
    REGISTER_DISABLED_PASS(manager, ConvertGather7ToGather8)
    REGISTER_PASS(manager, ConvertDeformableConv8To1)
    REGISTER_PASS(manager, ConvertSoftMax8ToSoftMax1)
    REGISTER_DISABLED_PASS(manager, ConvertSoftMax1ToSoftMax8)
    REGISTER_PASS(manager, ConvertMaxPool14ToMaxPool8)
    REGISTER_PASS(manager, ConvertMaxPool8ToMaxPool1)
    REGISTER_DISABLED_PASS(manager, ConvertMaxPool1ToMaxPool8)
    REGISTER_PASS(manager, ConvertPriorBox8To0)
    REGISTER_DISABLED_PASS(manager, ConvertDetectionOutput1ToDetectionOutput8)
    REGISTER_PASS(manager, ConvertDetectionOutput8ToDetectionOutput1)
    REGISTER_DISABLED_PASS(manager, ConvertROIAlign3To9)
    REGISTER_PASS(manager, ConvertROIAlign9To3)
    REGISTER_PASS(manager, ConvertMulticlassNms8ToMulticlassNms9)
    REGISTER_PASS(manager, ConvertXorToLogicalXor)
    REGISTER_PASS(manager, ConvertTopK11ToTopK3)
    REGISTER_PASS(manager, ConvertInterpolate11ToInterpolate4)
    REGISTER_PASS(manager, ConvertPad12ToPad1)
    REGISTER_PASS(manager, ConvertScatterElementsUpdate12ToScatterElementsUpdate3)
    REGISTER_PASS(manager, ConcatFusion)
    REGISTER_PASS(manager, ConvertAvgPool14ToAvgPool1)
    REGISTER_PASS(manager, ConvertEmbeddingBagOffsets15ToEmbeddingBagOffsetsSum3)
    REGISTER_PASS(manager, ConvertEmbeddingBagPacked15ToEmbeddingBagPackedSum3)
    REGISTER_PASS(manager, ConvertScatterNDUpdate15ToScatterNDUpdate3)
    REGISTER_PASS(manager, ConvertSliceScatter)
    REGISTER_PASS(manager, ConvertSqueeze15ToSqueeze0)

    auto fq_fusions = manager.register_pass<GraphRewrite>();
    ADD_MATCHER(fq_fusions, FakeQuantizeMulFusion)
    ADD_MATCHER(fq_fusions, FakeQuantizeReshapeFusion)
    ADD_MATCHER(fq_fusions, PullTransposeThroughFQUp)
    ADD_MATCHER(fq_fusions, ReluFakeQuantizeFusion)
    ADD_MATCHER(fq_fusions, AddFakeQuantizeFusion)
    ADD_MATCHER(fq_fusions, MulFakeQuantizeFusion)
    fq_fusions->set_name("ov::pass::FakeQuantizeFusions");

    // Temporary transformation to allow for PyTorch frontend to
    // partially support bitwise operators with boolean inputs for plugins
    // that didn't enabled BitwiseOps from opset13 and to allow for constant
    // folding for bool inputs
    REGISTER_PASS(manager, ConvertBitwiseToLogical)

    // StridesOptimization should be at the very end
    // because we cannot insert any MaxPools since they may prevent
    // other optimizations
    REGISTER_PASS(manager, StridesOptimization)
    REGISTER_PASS(manager, Validate)
    manager.run_passes(f);

    // Returning value is false because pass::Manager always apply Validation pass
    // if function was changed. This helps to avoid excess Validations after applying
    // this pass. In future when we will return more meaningful status code it will be
    // replaced with real status reported by manager.run_passes() method call.
    return false;
}
