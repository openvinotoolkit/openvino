// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "transformations/init_node_info.hpp"
#include "itt.hpp"
#include "transformations/common_optimizations/algebraic_simplification.hpp"
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

#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <transformations/common_optimizations/weights_dequantize_to_fake_quantize.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::CommonOptimizations, "CommonOptimizations", 0);

bool ngraph::pass::CommonOptimizations::run_on_function(std::shared_ptr<ngraph::Function> f) {
    RUN_ON_FUNCTION_SCOPE(CommonOptimizations);
    ngraph::pass::Manager manager(get_pass_config());

    // This pass must be called first in pipeline
    manager.register_pass<ngraph::pass::InitNodeInfo>();
    manager.register_pass<ngraph::pass::ConstantFolding>();
    manager.register_pass<ngraph::pass::RemoveFilteringBoxesBySize>(); // Resolves dynamism (replaces NonZero), CF needed

    // TODO: move to KMB
    manager.register_pass<ngraph::pass::ConvertQuantizeDequantize>();
    manager.register_pass<ngraph::pass::WeightsDequantizeToFakeQuantize>();

    manager.register_pass<ngraph::pass::ConstantFolding>();
    manager.register_pass<ngraph::pass::StridedSliceOptimization>(); // depends on CF
    manager.register_pass<ngraph::pass::BroadcastElementwiseFusion>();

    auto transpose_sinking = manager.register_pass<ngraph::pass::GraphRewrite>();
    transpose_sinking->add_matcher<ngraph::pass::TransposeSinking>();
    // SplitSqueezeConcatFusion should work in same GraphRewrite as TransposesSinking,
    // because it replaces pattern that may contain Transposes which must be optimized before
    // the transformation and it also inserts Transpose that can be optimized by TransposeSinking
    transpose_sinking->add_matcher<ngraph::pass::SplitSqueezeConcatFusion>();

    auto eliminations = manager.register_pass<ngraph::pass::GraphRewrite>();
    eliminations->add_matcher<ngraph::pass::EliminateUnsqueezeGather>();
    eliminations->add_matcher<ngraph::pass::AlgebraicSimplification>(); // may introduce fake dynamism
    eliminations->add_matcher<ngraph::pass::NopElimination>(); // may introduce fake dynamism
    eliminations->set_name("ngraph::pass::CommonEliminations");

    manager.register_pass<ngraph::pass::ConstantFolding>();

    auto common_fusions = manager.register_pass<ngraph::pass::GraphRewrite>();
    common_fusions->add_matcher<ngraph::pass::ConvertScatterElementsToScatter>();
    common_fusions->add_matcher<ngraph::pass::DepthToSpaceFusion>();
    common_fusions->add_matcher<ngraph::pass::SoftPlusFusion>();
    common_fusions->add_matcher<ngraph::pass::SoftPlusToMishFusion>();
    common_fusions->add_matcher<ngraph::pass::SwishFusion>();
    common_fusions->add_matcher<ngraph::pass::ShuffleChannelsFusion>(false);
    common_fusions->add_matcher<ngraph::pass::HSwishFusion>();
    common_fusions->add_matcher<ngraph::pass::HSigmoidFusion>();
    common_fusions->add_matcher<ngraph::pass::NormalizeL2Fusion>();
    common_fusions->add_matcher<ngraph::pass::ClampFusion>();
    common_fusions->add_matcher<ngraph::pass::PadFusion>();
    common_fusions->add_matcher<ngraph::pass::SoftmaxFusion>();
    common_fusions->add_matcher<ngraph::pass::MVNFusion>();
    common_fusions->add_matcher<ngraph::pass::SpaceToBatchFusion>();
    common_fusions->add_matcher<ngraph::pass::BatchToSpaceFusion>();
    common_fusions->add_matcher<ngraph::pass::DilatedConvolutionConverter>();
    common_fusions->add_matcher<ngraph::pass::GeluFusion>();
    common_fusions->add_matcher<ngraph::pass::TransposeToReshape>();
    common_fusions->set_name("ngraph::pass::CommonFusions");

    manager.register_pass<ngraph::pass::ConvertPadToGroupConvolution, false>();
    manager.register_pass<ngraph::pass::ConvertInterpolate1ToInterpolate4, false>();
    manager.register_pass<ngraph::pass::BinarizeWeights>();
    manager.register_pass<ngraph::pass::ConvToBinaryConv>();

    auto decomp = manager.register_pass<ngraph::pass::GraphRewrite>();
    decomp->add_matcher<ngraph::pass::Gelu7Downgrade>();
    decomp->add_matcher<ngraph::pass::BidirectionalSequenceDecomposition>();
    decomp->add_matcher<ngraph::pass::ReduceL1Decomposition>();
    decomp->add_matcher<ngraph::pass::ReduceL2Decomposition>();
    decomp->add_matcher<ngraph::pass::HSwishDecomposition>();
    decomp->add_matcher<ngraph::pass::HSigmoidDecomposition>();
    decomp->add_matcher<ngraph::pass::LogSoftmaxDecomposition>();
    decomp->add_matcher<ngraph::pass::ConvertReduceToPooling>();
    decomp->add_matcher<ngraph::pass::ConvertBroadcastToTiles>();
    decomp->add_matcher<ngraph::pass::ConvertMod>();
    decomp->add_matcher<ngraph::pass::ConvertGELU>();
    decomp->add_matcher<ngraph::pass::ConvertMinimum>();
    decomp->add_matcher<ngraph::pass::ConvertSubtract>();
    decomp->add_matcher<ngraph::pass::ConvertDivide>();
    decomp->add_matcher<ngraph::pass::ConvertNegative>();
    decomp->add_matcher<ngraph::pass::ConvertDepthToSpace>();
    decomp->add_matcher<ngraph::pass::ConvertSpaceToDepth>();
    decomp->add_matcher<ngraph::pass::BatchNormDecomposition>();
    decomp->add_matcher<ngraph::pass::MVN6Decomposition>();
    decomp->add_matcher<ngraph::pass::SimplifyCTCGreedyDecoderSeqLen>();
    decomp->add_matcher<ngraph::pass::EinsumDecomposition>();
    decomp->add_matcher<ngraph::pass::GatherNegativeConstIndicesNormalize>();
    decomp->set_name("ngraph::pass::CommonDecompositions");

    // CF is required after all decompositions
    manager.register_pass<ngraph::pass::ConstantFolding>();

    // LinOpSequenceFusion must be executed after all decompositions
    manager.register_pass<ngraph::pass::LinOpSequenceFusion>();

    auto conv_fusions = manager.register_pass<ngraph::pass::GraphRewrite>();
    conv_fusions->add_matcher<ngraph::pass::ConvolutionMultiplyFusion>();
    conv_fusions->add_matcher<ngraph::pass::GroupConvolutionMultiplyFusion>();
    conv_fusions->add_matcher<ngraph::pass::ConvolutionBackpropDataMultiplyFusion>();
    conv_fusions->add_matcher<ngraph::pass::GroupConvolutionBackpropDataMultiplyFusion>();
    conv_fusions->set_name("ngraph::pass::ConvFusions");

    manager.register_pass<ngraph::pass::ConstantFolding>();
    manager.register_pass<ngraph::pass::ConvertGather7ToGather1>();
    manager.register_pass<ngraph::pass::ConvertGather1ToGather7, false>();

    auto fq_fusions = manager.register_pass<ngraph::pass::GraphRewrite>();
    fq_fusions->add_matcher<ngraph::pass::FakeQuantizeMulFusion>();
    fq_fusions->add_matcher<ngraph::pass::FakeQuantizeReshapeFusion>();
    fq_fusions->add_matcher<ngraph::pass::PullTransposeThroughFQUp>();
    fq_fusions->add_matcher<ngraph::pass::ReluFakeQuantizeFusion>();
    fq_fusions->set_name("ngraph::pass::FakeQuantizeFusions");

    manager.run_passes(f);

    // Returning value is false because pass::Manager always apply Validation pass
    // if function was changed. This helps to avoid excess Validations after applying
    // this pass. In future when we will return more meaningful status code it will be
    // replaced with real status reported by manager.run_passes() method call.
    return false;
}
