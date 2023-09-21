// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "legacy/transformations/convert_opset1_to_legacy/convert_opset1_to_legacy.hpp"

#include <memory>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/common_optimizations/fq_mul_fusion.hpp>
#include <transformations/common_optimizations/fq_reshape_fusion.hpp>
#include <transformations/common_optimizations/pull_transpose_through_fq.hpp>
#include <transformations/op_conversions/convert_broadcast_to_tiles.hpp>
#include <transformations/op_conversions/convert_divide.hpp>
#include <transformations/op_conversions/convert_minimum_to_power_and_max.hpp>
#include <transformations/op_conversions/convert_mod.hpp>
#include <transformations/op_conversions/convert_negative.hpp>
#include <transformations/op_conversions/convert_previous_nms_to_nms_5.hpp>
#include <transformations/op_conversions/convert_subtract.hpp>
#include <vector>

#include "legacy/transformations/convert_opset1_to_legacy/conv_bias_fusion.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/convert_cells_to_cells_ie.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/convert_convolutions.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/convert_gather_to_gather_ie.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/convert_gathertree_to_gathertree_ie.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/convert_hard_sigmoid_to_hard_sigmoid_ie.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/convert_interpolate_to_interp_or_resample.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/convert_lrn_to_lrn_ie.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/convert_matmul_to_fc_or_gemm.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/convert_mul_add_to_scaleshift_or_power.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/convert_mul_or_add_finally.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/convert_nms_5_to_legacy.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/convert_normalizel2_to_normalize_ie.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/convert_one_hot_to_one_hot_ie.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/convert_pad_to_pad_ie.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/convert_power_to_power_ie.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/convert_prelu_to_relu_ie.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/convert_proposal_to_proposal_ie.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/convert_selu_to_selu_ie.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/convert_sequences_to_sequences_ie.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/convert_sqrt_to_power_ie.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/convert_strided_slice_to_crop.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/convert_swish_to_swish_ie.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/convert_tile_to_ie_tile.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/convert_topk_to_topk_ie.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/fc_bias_fusion.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/reshape_1d_ops.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/reshape_fc_fusion.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/reshape_fully_connected.hpp"

bool ngraph::pass::ConvertOpSet1ToLegacy::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    ngraph::pass::Manager manager(get_pass_config());

    manager.register_pass<ngraph::pass::ConstantFolding>();

    // Some passes before ConvertOpSet1ToLegacy can produce some of this
    // operations. So for convenience we decompose this operations here and
    // in CommonOptimizations.
    auto decomp = manager.register_pass<ov::pass::GraphRewrite>();
    decomp->add_matcher<ov::pass::ConvertMod>();
    decomp->add_matcher<ov::pass::ConvertMinimum>();
    decomp->add_matcher<ov::pass::ConvertSubtract>();
    decomp->add_matcher<ov::pass::ConvertDivide>();
    decomp->add_matcher<ov::pass::ConvertNegative>();
    decomp->set_name("ngraph::pass::LegacyDecompositions");

    auto convert_matmul = manager.register_pass<ov::pass::GraphRewrite>();
    convert_matmul->add_matcher<ngraph::pass::ConvertMatMulToFC>();
    convert_matmul->add_matcher<ov::pass::PullTransposeThroughFQUp>();
    convert_matmul->add_matcher<ngraph::pass::ConvertMatMulToGemm>();
    convert_matmul->set_name("ngraph::pass::ConvertMatMul");

    manager.register_pass<ngraph::pass::ConstantFolding>();

    // Convolution/Deconvolution/FullyConnected fusions
    manager.register_pass<ngraph::pass::ConvertConvolutions>();

    // Convolution/Deconvolution/FullyConnected fusions
    auto fusion = manager.register_pass<ov::pass::GraphRewrite>();
    fusion->add_matcher<ngraph::pass::ConvAddFusion>();
    fusion->add_matcher<ngraph::pass::DeconvAddFusion>();
    fusion->add_matcher<ngraph::pass::FullyConnectedBiasFusion>();
    fusion->set_name("ngraph::pass::BiasFusions");

    // CF is required after fusions
    manager.register_pass<ngraph::pass::ConstantFolding>();

    // List of passes that convert opset1 operations to legacy
    // plus transformations that are required by InferenceEngine
    // All this transformations can be executed simultaneously
    auto anchor = manager.register_pass<ov::pass::GraphRewrite>();
    anchor->add_matcher<ngraph::pass::ReshapeFullyConnected>();
    anchor->add_matcher<ngraph::pass::Reshape1DConvolution>();
    anchor->add_matcher<ngraph::pass::Reshape1DAvgPool>();
    anchor->add_matcher<ngraph::pass::Reshape1DMaxPool>();
    anchor->add_matcher<ngraph::pass::ConvertNormalizeL2WithMulToNormalizeIE>();
    anchor->add_matcher<ngraph::pass::ConvertHardSigmoidToLegacyMatcher>();
    anchor->add_matcher<ngraph::pass::ConvertProposalToLegacyMatcher>();
    anchor->add_matcher<ngraph::pass::ConvertProposal4ToLegacyMatcher>();
    anchor->add_matcher<ov::pass::ConvertBroadcastToTiles>();
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
    anchor->add_matcher<ov::pass::ConvertNMS1ToNMS5>();
    anchor->add_matcher<ov::pass::ConvertNMS3ToNMS5>();
    anchor->add_matcher<ov::pass::ConvertNMS4ToNMS5>();
    anchor->add_matcher<ngraph::pass::ConvertNMS5ToLegacyMatcher>();
    anchor->add_matcher<ngraph::pass::ConvertGRUSequenceMatcher>();
    anchor->add_matcher<ngraph::pass::ConvertRNNSequenceMatcher>();
    anchor->add_matcher<ngraph::pass::ConvertLSTMSequenceMatcher>();
    anchor->set_name("ngraph::pass::LegacyConversions");

    // List of final conversion transformations that must to be executed
    // after previous group of transformations

    if (!ov::op::util::has_op_with_type<ngraph::op::FakeQuantize>(f)) {
        manager.register_pass<ngraph::pass::ReshapeFullyConnectedFusion>();
    }
    manager.register_pass<ngraph::pass::ConvertNormalizeL2ToLegacyMatcher>();
    manager.register_pass<ngraph::pass::ConvertMulAddToScaleShiftOrPower>();
    manager.register_pass<ngraph::pass::ConvertMulOrAddFinally>();

    manager.register_pass<ngraph::pass::ConstantFolding>();

    manager.run_passes(f);

    // Returning value is false because pass::Manager always apply Validation pass
    // if function was changed. This helps to avoid excess Validations after applying
    // this pass. In future when we will return more meaningful status code it will be
    // replaced with real status reported by manager.run_passes() method call.
    return false;
}
