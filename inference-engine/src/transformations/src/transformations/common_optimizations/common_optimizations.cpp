// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "transformations/init_node_info.hpp"
#include "itt.hpp"
#include "transformations/common_optimizations/broadcast_elementwise_fusion.hpp"
#include "transformations/common_optimizations/nop_elimination.hpp"
#include "transformations/common_optimizations/common_optimizations.hpp"
#include "transformations/common_optimizations/conv_mul_fusion.hpp"
#include "transformations/common_optimizations/fq_mul_fusion.hpp"
#include "transformations/common_optimizations/fq_reshape_fusion.hpp"
#include "transformations/common_optimizations/gelu_fusion.hpp"
#include "transformations/common_optimizations/depth_to_space_fusion.hpp"
#include "transformations/common_optimizations/optimize_strided_slice.hpp"
#include "transformations/common_optimizations/softplus_fusion.hpp"
#include "transformations/common_optimizations/softplus_to_mish_fusion.hpp"
#include "transformations/common_optimizations/swish_fusion.hpp"
#include "transformations/common_optimizations/normalize_l2_fusion.hpp"
#include "transformations/common_optimizations/pull_transpose_through_fq.hpp"
#include "transformations/common_optimizations/leaky_relu_fusion.hpp"
#include "transformations/common_optimizations/lin_op_sequence_fusion.hpp"
#include "transformations/common_optimizations/remove_filtering_boxes_by_size.hpp"
#include "transformations/common_optimizations/hsigmoid_fusion.hpp"
#include "transformations/common_optimizations/hswish_fusion.hpp"
#include "transformations/common_optimizations/convert_quantize_dequantize.hpp"
#include "transformations/common_optimizations/relu_fake_quantize_fusion.hpp"
#include "transformations/common_optimizations/add_fake_quantize_fusion.hpp"
#include "transformations/common_optimizations/mul_fake_quantize_fusion.hpp"
#include "transformations/common_optimizations/clamp_fusion.hpp"
#include "transformations/common_optimizations/pad_fusion.hpp"
#include "transformations/common_optimizations/eliminate_unsqueeze_gather.hpp"
#include "transformations/common_optimizations/shuffle_channels_fusion.hpp"
#include "transformations/common_optimizations/softmax_fusion.hpp"
#include "transformations/common_optimizations/mvn_fusion.hpp"
#include "transformations/common_optimizations/binarize_weights.hpp"
#include "transformations/common_optimizations/conv_to_binary_conv.hpp"
#include "transformations/common_optimizations/space_to_batch_fusion.hpp"
#include "transformations/common_optimizations/batch_to_space_fusion.hpp"
#include "transformations/common_optimizations/dilated_convolution_converter.hpp"
#include "transformations/common_optimizations/transpose_sinking.hpp"
#include "transformations/common_optimizations/split_squeeze_concat_fusion.hpp"
#include "transformations/common_optimizations/transpose_to_reshape.hpp"
#include "transformations/common_optimizations/strides_optimization.hpp"
#include "transformations/op_conversions/bidirectional_sequences_decomposition.hpp"
#include "transformations/op_conversions/convert_pad_to_group_conv.hpp"
#include "transformations/op_conversions/convert_divide.hpp"
#include "transformations/op_conversions/convert_gather_v7_to_gather_v1.hpp"
#include "transformations/op_conversions/convert_gather_v1_to_gather_v7.hpp"
#include "transformations/op_conversions/convert_mod.hpp"
#include "transformations/op_conversions/convert_minimum_to_power_and_max.hpp"
#include "transformations/op_conversions/convert_negative.hpp"
#include "transformations/op_conversions/convert_scatter_elements_to_scatter.hpp"
#include "transformations/op_conversions/convert_reduce_to_pooling.hpp"
#include "transformations/op_conversions/convert_subtract.hpp"
#include "transformations/op_conversions/convert_depth_to_space.hpp"
#include "transformations/op_conversions/convert_space_to_depth.hpp"
#include "transformations/op_conversions/convert_broadcast_to_tiles.hpp"
#include "transformations/op_conversions/convert_gelu.hpp"
#include "transformations/op_conversions/convert_interpolate1_to_interpolate4.hpp"
#include "transformations/op_conversions/batch_norm_decomposition.hpp"
#include "transformations/op_conversions/einsum_decomposition.hpp"
#include "transformations/op_conversions/gelu7_downgrade.hpp"
#include "transformations/op_conversions/reduce_l1_decomposition.hpp"
#include "transformations/op_conversions/reduce_l2_decomposition.hpp"
#include "transformations/op_conversions/hswish_decomposition.hpp"
#include "transformations/op_conversions/convert_previous_nms_to_nms_5.hpp"
#include "transformations/op_conversions/hsigmoid_decomposition.hpp"
#include "transformations/op_conversions/log_softmax_decomposition.hpp"
#include "transformations/op_conversions/mvn6_decomposition.hpp"
#include "transformations/op_conversions/simplify_ctc_greedy_decoder_seq_len.hpp"
#include "transformations/op_conversions/gather_normalize_negative_indices.hpp"
#include "transformations/op_conversions/convert_deformable_conv_v8_to_v1.hpp"

#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <transformations/common_optimizations/weights_dequantize_to_fake_quantize.hpp>
#include <transformations/common_optimizations/simplify_shape_of_sub_graph.hpp>

NGRAPH_RTTI_DEFINITION(ov::pass::CommonOptimizations, "CommonOptimizations", 0);

bool ov::pass::CommonOptimizations::run_on_function(std::shared_ptr<ov::Function> f) {
    RUN_ON_FUNCTION_SCOPE(CommonOptimizations);
    ov::pass::Manager manager(get_pass_config());

    // This pass must be called first in pipeline
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::pass::SimplifyShapeOfSubGraph>();
    manager.register_pass<ov::pass::ConstantFolding>();
    manager.register_pass<ov::pass::RemoveFilteringBoxesBySize>(); // Resolves dynamism (replaces NonZero), CF needed

    // TODO: move to KMB
    manager.register_pass<ov::pass::ConvertQuantizeDequantize>();
    manager.register_pass<ov::pass::WeightsDequantizeToFakeQuantize>();

    manager.register_pass<ov::pass::ConstantFolding>();
    manager.register_pass<ov::pass::StridedSliceOptimization>(); // depends on CF
    manager.register_pass<ov::pass::BroadcastElementwiseFusion>();

    auto transpose_sinking = manager.register_pass<ov::pass::GraphRewrite>();
    transpose_sinking->add_matcher<ov::pass::TransposeSinking>();
    // SplitSqueezeConcatFusion should work in same GraphRewrite as TransposesSinking,
    // because it replaces pattern that may contain Transposes which must be optimized before
    // the transformation and it also inserts Transpose that can be optimized by TransposeSinking
    transpose_sinking->add_matcher<ov::pass::SplitSqueezeConcatFusion>();

    auto eliminations = manager.register_pass<ov::pass::GraphRewrite>();
    eliminations->add_matcher<ov::pass::EliminateUnsqueezeGather>();
    eliminations->add_matcher<ov::pass::NopElimination>(); // may introduce fake dynamism
    eliminations->set_name("ov::pass::CommonEliminations");

    manager.register_pass<ov::pass::ConstantFolding>();

    auto common_fusions = manager.register_pass<ov::pass::GraphRewrite>();
    common_fusions->add_matcher<ov::pass::ConvertScatterElementsToScatter>();
    common_fusions->add_matcher<ov::pass::DepthToSpaceFusion>();
    common_fusions->add_matcher<ov::pass::SoftPlusFusion>();
    common_fusions->add_matcher<ov::pass::SoftPlusToMishFusion>();
    common_fusions->add_matcher<ov::pass::SwishFusion>();
    common_fusions->add_matcher<ov::pass::ShuffleChannelsFusion>(false);
    common_fusions->add_matcher<ov::pass::HSwishFusion>();
    common_fusions->add_matcher<ov::pass::HSigmoidFusion>();
    common_fusions->add_matcher<ov::pass::NormalizeL2Fusion>();
    common_fusions->add_matcher<ov::pass::ClampFusion>();
    common_fusions->add_matcher<ov::pass::PadFusion>();
    common_fusions->add_matcher<ov::pass::SoftmaxFusion>();
    common_fusions->add_matcher<ov::pass::MVNFusion>();
    common_fusions->add_matcher<ov::pass::SpaceToBatchFusion>();
    common_fusions->add_matcher<ov::pass::BatchToSpaceFusion>();
    common_fusions->add_matcher<ov::pass::DilatedConvolutionConverter>();
    common_fusions->add_matcher<ov::pass::GeluFusion>();
    common_fusions->add_matcher<ov::pass::TransposeToReshape>();
    common_fusions->add_matcher<ov::pass::LeakyReluFusion>();
    common_fusions->set_name("ov::pass::CommonFusions");

    manager.register_pass<ov::pass::ConvertPadToGroupConvolution, false>();
    manager.register_pass<ov::pass::ConvertInterpolate1ToInterpolate4, false>();
    manager.register_pass<ov::pass::BinarizeWeights>();
    manager.register_pass<ov::pass::ConvToBinaryConv>();

    auto decomp = manager.register_pass<ov::pass::GraphRewrite>();
    decomp->add_matcher<ov::pass::Gelu7Downgrade>();
    decomp->add_matcher<ov::pass::BidirectionalSequenceDecomposition>();
    decomp->add_matcher<ov::pass::ReduceL1Decomposition>();
    decomp->add_matcher<ov::pass::ReduceL2Decomposition>();
    decomp->add_matcher<ov::pass::HSwishDecomposition>();
    decomp->add_matcher<ov::pass::HSigmoidDecomposition>();
    decomp->add_matcher<ov::pass::LogSoftmaxDecomposition>();
    decomp->add_matcher<ov::pass::ConvertReduceToPooling>();
    decomp->add_matcher<ov::pass::ConvertBroadcastToTiles>();
    decomp->add_matcher<ov::pass::ConvertMod>();
    decomp->add_matcher<ov::pass::ConvertGELU>();
    decomp->add_matcher<ov::pass::ConvertMinimum>();
    decomp->add_matcher<ov::pass::ConvertSubtract>();
    decomp->add_matcher<ov::pass::ConvertDivide>();
    decomp->add_matcher<ov::pass::ConvertNegative>();
    decomp->add_matcher<ov::pass::ConvertDepthToSpace>();
    decomp->add_matcher<ov::pass::ConvertSpaceToDepth>();
    decomp->add_matcher<ov::pass::BatchNormDecomposition>();
    decomp->add_matcher<ov::pass::MVN6Decomposition>();
    decomp->add_matcher<ov::pass::SimplifyCTCGreedyDecoderSeqLen>();
    decomp->add_matcher<ov::pass::EinsumDecomposition>();
    decomp->add_matcher<ov::pass::GatherNegativeConstIndicesNormalize>();
    decomp->set_name("ov::pass::CommonDecompositions");

    // CF is required after all decompositions
    manager.register_pass<ov::pass::ConstantFolding>();

    // LinOpSequenceFusion must be executed after all decompositions
    manager.register_pass<ov::pass::LinOpSequenceFusion>();

    auto conv_fusions = manager.register_pass<ov::pass::GraphRewrite>();
    conv_fusions->add_matcher<ov::pass::ConvolutionMultiplyFusion>();
    conv_fusions->add_matcher<ov::pass::GroupConvolutionMultiplyFusion>();
    conv_fusions->add_matcher<ov::pass::ConvolutionBackpropDataMultiplyFusion>();
    conv_fusions->add_matcher<ov::pass::GroupConvolutionBackpropDataMultiplyFusion>();
    conv_fusions->set_name("ov::pass::ConvFusions");

    manager.register_pass<ov::pass::ConstantFolding>();
    manager.register_pass<ov::pass::ConvertGather7ToGather1>();
    manager.register_pass<ov::pass::ConvertGather1ToGather7, false>();
    manager.register_pass<ov::pass::ConvertDeformableConv8To1>();

    auto fq_fusions = manager.register_pass<ov::pass::GraphRewrite>();
    fq_fusions->add_matcher<ov::pass::FakeQuantizeMulFusion>();
    fq_fusions->add_matcher<ov::pass::FakeQuantizeReshapeFusion>();
    fq_fusions->add_matcher<ov::pass::PullTransposeThroughFQUp>();
    fq_fusions->add_matcher<ov::pass::ReluFakeQuantizeFusion>();
    fq_fusions->set_name("ov::pass::FakeQuantizeFusions");

    // StridesOptimization should be at the very end
    // because we cannot insert any MaxPools since they may prevent
    // other optimizations
    manager.register_pass<ov::pass::StridesOptimization>();

    manager.run_passes(f);

    // Returning value is false because pass::Manager always apply Validation pass
    // if function was changed. This helps to avoid excess Validations after applying
    // this pass. In future when we will return more meaningful status code it will be
    // replaced with real status reported by manager.run_passes() method call.
    return false;
}
