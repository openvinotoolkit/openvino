// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/passes.hpp"

#include "transformations/common_optimizations/add_fake_quantize_fusion.hpp"
#include "transformations/common_optimizations/batch_to_space_fusion.hpp"
#include "transformations/common_optimizations/conv_mul_fusion.hpp"
#include "transformations/common_optimizations/convert_quantize_dequantize.hpp"
#include "transformations/common_optimizations/depth_to_space_fusion.hpp"
#include "transformations/common_optimizations/dropout_with_random_uniform_replacer.hpp"
#include "transformations/common_optimizations/fq_mul_fusion.hpp"
#include "transformations/common_optimizations/lin_op_sequence_fusion.hpp"
#include "transformations/common_optimizations/moc_transformations.hpp"
#include "transformations/common_optimizations/mul_conv_fusion.hpp"
#include "transformations/common_optimizations/mul_fake_quantize_fusion.hpp"
#include "transformations/common_optimizations/mvn_fusion.hpp"
#include "transformations/common_optimizations/pad_fusion.hpp"
#include "transformations/common_optimizations/pull_through_reduce.hpp"
#include "transformations/common_optimizations/reduce_reshape_fusion.hpp"
#include "transformations/common_optimizations/relu_fake_quantize_fusion.hpp"
#include "transformations/common_optimizations/rms_fusion.hpp"
#include "transformations/common_optimizations/shared_ops_optimization.hpp"
#include "transformations/common_optimizations/shuffle_channels_fusion.hpp"
#include "transformations/common_optimizations/space_to_batch_fusion.hpp"
#include "transformations/common_optimizations/strides_optimization.hpp"
#include "transformations/common_optimizations/transpose_to_reshape.hpp"
#include "transformations/common_optimizations/weights_dequantize_to_fake_quantize.hpp"
#include "transformations/control_flow/unroll_if.hpp"
#include "transformations/control_flow/unroll_tensor_iterator.hpp"
#include "transformations/fp16_compression/mark_decompression_convert_constant_folding.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/low_precision/mark_dequantization_subgraph.hpp"
#include "transformations/op_conversions/batch_norm_decomposition.hpp"
#include "transformations/op_conversions/bidirectional_sequences_decomposition.hpp"
#include "transformations/op_conversions/convert_avgpool_downgrade.hpp"
#include "transformations/op_conversions/convert_broadcast_to_tiles.hpp"
#include "transformations/op_conversions/convert_convertlike.hpp"
#include "transformations/op_conversions/convert_deformable_conv_v8_to_v1.hpp"
#include "transformations/op_conversions/convert_gather_upgrade.hpp"
#include "transformations/op_conversions/convert_interpolate11_downgrade.hpp"
#include "transformations/op_conversions/convert_interpolate1_to_interpolate4.hpp"
#include "transformations/op_conversions/convert_maxpool_downgrade.hpp"
#include "transformations/op_conversions/convert_nms9_to_nms_ie_internal.hpp"
#include "transformations/op_conversions/convert_pad12_downgrade.hpp"
#include "transformations/op_conversions/convert_pad_to_group_conv.hpp"
#include "transformations/op_conversions/convert_previous_nms_to_nms_9.hpp"
#include "transformations/op_conversions/convert_reduce_to_pooling.hpp"
#include "transformations/op_conversions/convert_scatter_elements_update12_downgrade.hpp"
#include "transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp"
#include "transformations/op_conversions/convert_shapeof3.hpp"
#include "transformations/op_conversions/convert_slice_to_strided_slice.hpp"
#include "transformations/op_conversions/convert_softmax_upgrade.hpp"
#include "transformations/op_conversions/convert_topk11_downgrade.hpp"
#include "transformations/op_conversions/detection_output_downgrade.hpp"
#include "transformations/op_conversions/einsum_decomposition.hpp"
#include "transformations/op_conversions/gelu7_downgrade.hpp"
#include "transformations/op_conversions/group_normalization_decomposition.hpp"
#include "transformations/op_conversions/log_softmax_decomposition.hpp"
#include "transformations/op_conversions/normalize_l2_decomposition.hpp"
#include "transformations/op_conversions/scaled_dot_product_attention_decomposition.hpp"
#include "transformations/op_conversions/softmax_decomposition.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"
#include "transformations/utils/utils.hpp"

namespace intel_npu {

void runOVPasses(const std::shared_ptr<ov::Model>& model) {
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();

    ov::element::TypeVector decompression_precisions{ov::element::u4,
                                                     ov::element::i4,
                                                     ov::element::nf4,
                                                     ov::element::u8,
                                                     ov::element::i8,
                                                     ov::element::f8e4m3,
                                                     ov::element::f8e5m2,
                                                     ov::element::f8e8m0};
    manager.register_pass<ov::pass::MarkDequantization>(decompression_precisions, /*fold_subtract_const=*/true);
    manager.register_pass<ov::pass::KeepConstPrecision>(decompression_precisions, /*fold_subtract_const=*/true);
    manager.register_pass<ov::pass::SharedOpOptimization>();
    manager.register_pass<ov::pass::ConvertQuantizeDequantize>();
    manager.register_pass<ov::pass::ConstantFolding>();
    manager.register_pass<ov::pass::ConvertScatterElementsUpdate12ToScatterElementsUpdate3>();
    manager.register_pass<ov::pass::ConvertInterpolate1ToInterpolate4>();
    manager.register_pass<ov::pass::ConvertInterpolate11ToInterpolate4>();
    manager.register_pass<ov::pass::ConvertTopK11ToTopK3>();
    manager.register_pass<ov::pass::ConvertPad12ToPad1>();
    manager.register_pass<ov::pass::ConstantFolding>();
    manager.register_pass<ov::pass::SliceToStridedSlice>(true);
    manager.register_pass<ov::pass::MOCTransformations>(true, false);

    auto pass_config = manager.get_pass_config();
    pass_config->disable<ov::pass::PadFusionConvolution>();
    pass_config->disable<ov::pass::PadFusionGroupConvolution>();
    pass_config->disable<ov::pass::MVNFusionWithConstantsInside>();
    pass_config->disable<ov::pass::PullThroughReduce>();
    pass_config->disable<ov::pass::AddFakeQuantizeFusion>();
    pass_config->disable<ov::pass::FakeQuantizeMulFusion>();
    pass_config->disable<ov::pass::MulFakeQuantizeFusion>();

    manager.register_pass<ov::pass::ConvertNMS1ToNMS9>();
    manager.register_pass<ov::pass::ConvertNMS3ToNMS9>();
    manager.register_pass<ov::pass::ConvertNMS4ToNMS9>();
    manager.register_pass<ov::pass::ConvertNMS5ToNMS9>();

    auto static_shape = manager.register_pass<ov::pass::GraphRewrite>();
    static_shape->add_matcher<ov::pass::ConvertNMS9ToNMSIEInternal>();
    static_shape->set_name("ov::pass::CommonStaticShape");

    auto common_fusions = manager.register_pass<ov::pass::GraphRewrite>();
    common_fusions->add_matcher<ov::pass::DepthToSpaceFusion>();
    common_fusions->add_matcher<ov::pass::ShuffleChannelsFusion>(false);
    common_fusions->add_matcher<ov::pass::SpaceToBatchFusion>();
    common_fusions->add_matcher<ov::pass::BatchToSpaceFusion>();
    common_fusions->add_matcher<ov::pass::TransposeToReshape>();
    common_fusions->add_matcher<ov::pass::RMSFusion>();
    common_fusions->set_name("ov::pass::CommonFusions");

    auto decomp = manager.register_pass<ov::pass::GraphRewrite>();
    decomp->add_matcher<ov::pass::Gelu7Downgrade>();
    decomp->add_matcher<ov::pass::BidirectionalGRUSequenceDecomposition>();
    decomp->add_matcher<ov::pass::BidirectionalRNNSequenceDecomposition>();
    decomp->add_matcher<ov::pass::ConvertBroadcastToTiles>();
    decomp->add_matcher<ov::pass::ConvertConvertLike>();
    decomp->add_matcher<ov::pass::BatchNormDecomposition>();
    decomp->add_matcher<ov::pass::EinsumDecomposition>();
    decomp->add_matcher<ov::pass::DropoutWithRandomUniformReplacer>();
    decomp->add_matcher<ov::pass::ScaledDotProductAttentionDecomposition>();
    decomp->add_matcher<ov::pass::GroupNormalizationDecomposition>();
    decomp->set_name("ov::pass::CommonDecompositions");

    manager.register_pass<ov::pass::ConstantFolding>();
    manager.register_pass<ov::pass::LinOpSequenceFusion>();
    manager.register_pass<ov::pass::UnrollIf>();

    auto conv_fusions = manager.register_pass<ov::pass::GraphRewrite>();
    conv_fusions->add_matcher<ov::pass::ConvolutionMultiplyFusion>();
    conv_fusions->add_matcher<ov::pass::GroupConvolutionMultiplyFusion>();
    conv_fusions->add_matcher<ov::pass::ConvolutionBackpropDataMultiplyFusion>();
    conv_fusions->add_matcher<ov::pass::GroupConvolutionBackpropDataMultiplyFusion>();
    conv_fusions->add_matcher<ov::pass::MultiplyConvolutionFusion>();
    conv_fusions->add_matcher<ov::pass::MultiplyGroupConvolutionFusion>();
    conv_fusions->add_matcher<ov::pass::MultiplyConvolutionBackpropDataFusion>();
    conv_fusions->add_matcher<ov::pass::MultiplyGroupConvolutionBackpropDataFusion>();
    conv_fusions->set_name("ov::pass::ConvFusions");

    manager.register_pass<ov::pass::ConstantFolding>();
    manager.register_pass<ov::pass::ConvertGather1ToGather7>();
    manager.register_pass<ov::pass::ConvertGather7ToGather8>();
    manager.register_pass<ov::pass::ConvertDeformableConv8To1>();
    manager.register_pass<ov::pass::ConvertMaxPool14ToMaxPool8>();
    manager.register_pass<ov::pass::ConvertMaxPool8ToMaxPool1>();
    manager.register_pass<ov::pass::ConvertAvgPool14ToAvgPool1>();
    manager.register_pass<ov::pass::ConvertSoftMax1ToSoftMax8>();
    manager.register_pass<ov::pass::ConvertDetectionOutput8ToDetectionOutput1>();
    manager.register_pass<ov::pass::ConvertShapeOf3>();

    manager.register_pass<ov::pass::StridesOptimization>();
    manager.register_pass<ov::pass::ConvertSoftMax1ToSoftMax8>();

    manager.run_passes(model);
}

}  // namespace intel_npu
