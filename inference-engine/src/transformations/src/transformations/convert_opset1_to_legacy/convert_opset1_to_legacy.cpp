// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_opset1_to_legacy/convert_opset1_to_legacy.hpp"

#include <transformations/convert_broadcast_to_tiles.hpp>
#include <transformations/convert_opset1_to_legacy/convert_convolutions.hpp>
#include <transformations/convert_divide.hpp>
#include <transformations/convert_mod.hpp>
#include <transformations/itt.hpp>
#include <transformations/convert_opset1_to_legacy/convert_cells_to_cells_ie.hpp>
#include <transformations/convert_opset1_to_legacy/convert_gather_to_gather_ie.hpp>
#include <transformations/convert_opset1_to_legacy/convert_gathertree_to_gathertree_ie.hpp>
#include <transformations/convert_opset1_to_legacy/convert_interpolate_to_interp_or_resample.hpp>
#include <transformations/convert_opset1_to_legacy/convert_lrn_to_lrn_ie.hpp>
#include <transformations/convert_opset1_to_legacy/convert_matmul_to_fc_or_gemm.hpp>
#include <transformations/convert_minimum_to_power_and_max.hpp>
#include <transformations/convert_opset1_to_legacy/convert_mul_add_to_scaleshift_or_power.hpp>
#include <transformations/convert_opset1_to_legacy/convert_mul_or_add_finally.hpp>
#include <transformations/convert_negative.hpp>
#include <transformations/convert_opset1_to_legacy/convert_nms_to_nms_ie.hpp>
#include <transformations/convert_opset1_to_legacy/convert_nms_4_to_legacy.hpp>
#include <transformations/convert_opset1_to_legacy/convert_normalizel2_to_normalize_ie.hpp>
#include <transformations/convert_opset1_to_legacy/convert_one_hot_to_one_hot_ie.hpp>
#include <transformations/convert_opset1_to_legacy/convert_pad_to_pad_ie.hpp>
#include <transformations/convert_opset1_to_legacy/convert_sqrt_to_power_ie.hpp>
#include <transformations/convert_opset1_to_legacy/convert_power_to_power_ie.hpp>
#include <transformations/convert_opset1_to_legacy/convert_prelu_to_relu_ie.hpp>
#include <transformations/convert_opset1_to_legacy/convert_proposal_to_proposal_ie.hpp>
#include <transformations/convert_reduce_to_pooling.hpp>
#include <transformations/convert_opset1_to_legacy/convert_strided_slice_to_crop.hpp>
#include <transformations/convert_subtract.hpp>
#include <transformations/convert_opset1_to_legacy/convert_selu_to_selu_ie.hpp>
#include <transformations/convert_opset1_to_legacy/convert_sequences_to_sequences_ie.hpp>
#include <transformations/convert_opset1_to_legacy/convert_swish_to_swish_ie.hpp>
#include <transformations/convert_opset1_to_legacy/convert_tile_to_ie_tile.hpp>
#include <transformations/convert_opset1_to_legacy/convert_topk_to_topk_ie.hpp>
#include <transformations/convert_depth_to_space.hpp>
#include <transformations/convert_space_to_depth.hpp>
#include <transformations/batch_norm_decomposition.hpp>
#include <transformations/convert_opset1_to_legacy/conv_bias_fusion.hpp>
#include <transformations/convert_opset1_to_legacy/fc_bias_fusion.hpp>
#include <transformations/convert_opset1_to_legacy/reshape_fc_fusion.hpp>
#include <transformations/convert_opset1_to_legacy/reshape_1d_ops.hpp>
#include <transformations/convert_opset1_to_legacy/reshape_fully_connected.hpp>
#include <transformations/pull_transpose_through_fq.hpp>
#include <transformations/convert_opset1_to_legacy/convert_hard_sigmoid_to_hard_sigmoid_ie.hpp>
#include <transformations/lin_op_sequence_fusoin.hpp>
#include <transformations/common_optimizations/conv_mul_fusion.hpp>
#include <transformations/hswish_decomposition.hpp>
#include <transformations/reduce_l1_decomposition.hpp>
#include <transformations/reduce_l2_decomposition.hpp>
#include <transformations/common_optimizations/fq_mul_fusion.hpp>
#include <transformations/common_optimizations/fq_reshape_fusion.hpp>

#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/manager.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

#include <memory>
#include <vector>

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertOpSet1ToLegacy, "ConvertOpSet1ToLegacy", 0);

bool ngraph::pass::ConvertOpSet1ToLegacy::run_on_function(std::shared_ptr<ngraph::Function> f) {
    OV_ITT_SCOPED_TASK(itt::domains::IETransform, "ngraph::pass::ConvertOpSet1ToLegacy");

    ngraph::pass::Manager manager;

    manager.register_pass<ngraph::pass::ConstantFolding>();

    // Some passes before ConvertOpSet1ToLegacy can produce some of this
    // operations. So for convenience we decompose this operations here and
    // in CommonOptimizations.
    auto decomp = manager.register_pass<ngraph::pass::GraphRewrite>();
    decomp->add_matcher<ngraph::pass::ConvertMod>();
    decomp->add_matcher<ngraph::pass::ConvertMinimum>();
    decomp->add_matcher<ngraph::pass::ConvertSubtract>();
    decomp->add_matcher<ngraph::pass::ConvertDivide>();
    decomp->add_matcher<ngraph::pass::ConvertNegative>();
    decomp->set_name("ngraph::pass::LegacyDecompositions");

    auto convert_matmul = manager.register_pass<ngraph::pass::GraphRewrite>();
    convert_matmul->add_matcher<ngraph::pass::ConvertMatMulToFC>();
    convert_matmul->add_matcher<ngraph::pass::PullTransposeThroughFQUp>();
    convert_matmul->add_matcher<ngraph::pass::ConvertMatMulToGemm>();
    convert_matmul->set_name("ngraph::pass::ConvertMatMul");

    manager.register_pass<ngraph::pass::ConstantFolding>();

    // Convolution/Deconvolution/FullyConnected fusions
    auto convert_convolutions = manager.register_pass<ngraph::pass::GraphRewrite>();
    convert_convolutions->add_matcher<ngraph::pass::ConvertConvolution>();
    convert_convolutions->add_matcher<ngraph::pass::ConvertGroupConvolution>();
    convert_convolutions->add_matcher<ngraph::pass::ConvertDeconvolution>();
    convert_convolutions->add_matcher<ngraph::pass::ConvertGroupDeconvolution>();
    convert_convolutions->set_name("ngraph::pass::ConvertConvolutions");

    // Convolution/Deconvolution/FullyConnected fusions
    auto fusion = manager.register_pass<ngraph::pass::GraphRewrite>();
    fusion->add_matcher<ngraph::pass::ConvAddFusion>();
    fusion->add_matcher<ngraph::pass::DeconvAddFusion>();
    fusion->add_matcher<ngraph::pass::FullyConnectedBiasFusion>();
    fusion->set_name("ngraph::pass::BiasFusions");

    // CF is required after fusions
    manager.register_pass<ngraph::pass::ConstantFolding>();

    // List of passes that convert opset1 operations to legacy
    // plus transformations that are required by InferenceEngine
    // All this transformations can be executed simultaneously
    auto anchor = manager.register_pass<ngraph::pass::GraphRewrite>();
    anchor->add_matcher<ngraph::pass::ReshapeFullyConnected>();
    anchor->add_matcher<ngraph::pass::Reshape1DConvolution>();
    anchor->add_matcher<ngraph::pass::Reshape1DAvgPool>();
    anchor->add_matcher<ngraph::pass::Reshape1DMaxPool>();
    anchor->add_matcher<ngraph::pass::ConvertNormalizeL2WithMulToNormalizeIE>();
    anchor->add_matcher<ngraph::pass::ConvertHardSigmoidToLegacyMatcher>();
    anchor->add_matcher<ngraph::pass::ConvertProposalToLegacyMatcher>();
    anchor->add_matcher<ngraph::pass::ConvertProposal4ToLegacyMatcher>();
    anchor->add_matcher<ngraph::pass::ConvertBroadcastToTiles>();
    anchor->add_matcher<ngraph::pass::ConvertTileToLegacyMatcher>();
    anchor->add_matcher<ngraph::pass::ConvertLRNToLegacyMatcher>();
    anchor->add_matcher<ngraph::pass::ConvertPadToLegacyMatcher>();
    anchor->add_matcher<ngraph::pass::ConvertLSTMCellMatcher>();
    anchor->add_matcher<ngraph::pass::ConvertRNNCellMatcher>();
    anchor->add_matcher<ngraph::pass::ConvertGRUCellMatcher>();
    anchor->add_matcher<ngraph::pass::ConvertInterpolateToInterpOrResampleMatcher>();
    anchor->add_matcher<ngraph::pass::ConvertStridedSliceToCropMatcher>();
    anchor->add_matcher<ngraph::pass::ConvertPowerToPowerIEMatcher>();
    anchor->add_matcher<ngraph::pass::ConvertSqrtToPowerIEMatcher>();
    anchor->add_matcher<ngraph::pass::ConvertPReLUToReLUIE>();
    anchor->add_matcher<ngraph::pass::ConvertGatherToGatherIEMatcher>();
    anchor->add_matcher<ngraph::pass::ConvertSeluToSeluIEMatcher>();
    anchor->add_matcher<ngraph::pass::ConvertSwishToSwishIEMatcher>();
    anchor->add_matcher<ngraph::pass::ConvertOneHotToOneHotIEMatcher>()->detect_output_type(f);
    anchor->add_matcher<ngraph::pass::ConvertGatherTreeToGatherTreeIEMatcher>();
    anchor->add_matcher<ngraph::pass::ConvertTopKToTopKIEMatcher>();
    anchor->add_matcher<ngraph::pass::ConvertNMSToNMSIEMatcher>();
    anchor->add_matcher<ngraph::pass::ConvertNMS4ToLegacyMatcher>();
    anchor->add_matcher<ngraph::pass::ConvertGRUSequenceMatcher>();
    anchor->add_matcher<ngraph::pass::ConvertRNNSequenceMatcher>();
    anchor->add_matcher<ngraph::pass::ConvertLSTMSequenceMatcher>();
    anchor->set_name("ngraph::pass::LegacyConversions");

    // List of final conversion transformations that must to be executed
    // after previous group of transformations
    manager.register_pass<ngraph::pass::ReshapeFullyConnectedFusion>();
    manager.register_pass<ngraph::pass::ConvertNormalizeL2ToLegacyMatcher>();
    manager.register_pass<ngraph::pass::ConvertMulAddToScaleShiftOrPower>();
    manager.register_pass<ngraph::pass::ConvertMulOrAddFinally>();

    manager.register_pass<ngraph::pass::ConstantFolding>();

    manager.set_callback(m_transformation_callback);
    manager.run_passes(f);
    return true;
}
