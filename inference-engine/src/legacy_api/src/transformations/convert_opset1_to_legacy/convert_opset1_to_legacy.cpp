// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "legacy/transformations/convert_opset1_to_legacy/convert_opset1_to_legacy.hpp"

#include "legacy/transformations/convert_opset1_to_legacy/convert_cells_to_cells_ie.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/convert_gather_to_gather_ie.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/convert_gathertree_to_gathertree_ie.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/convert_interpolate_to_interp_or_resample.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/convert_lrn_to_lrn_ie.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/convert_matmul_to_fc_or_gemm.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/convert_mul_add_to_scaleshift_or_power.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/convert_mul_or_add_finally.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/convert_nms_5_to_legacy.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/convert_normalizel2_to_normalize_ie.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/convert_one_hot_to_one_hot_ie.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/convert_pad_to_pad_ie.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/convert_sqrt_to_power_ie.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/convert_power_to_power_ie.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/convert_prelu_to_relu_ie.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/convert_proposal_to_proposal_ie.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/convert_strided_slice_to_crop.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/convert_selu_to_selu_ie.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/convert_sequences_to_sequences_ie.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/convert_swish_to_swish_ie.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/convert_tile_to_ie_tile.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/convert_topk_to_topk_ie.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/fc_bias_fusion.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/reshape_fc_fusion.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/reshape_1d_ops.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/reshape_fully_connected.hpp"
#include "legacy/transformations/convert_opset1_to_legacy/convert_hard_sigmoid_to_hard_sigmoid_ie.hpp"

#include <transformations/common_optimizations/fq_mul_fusion.hpp>
#include <transformations/common_optimizations/fq_reshape_fusion.hpp>
#include <transformations/common_optimizations/pull_transpose_through_fq.hpp>
#include <transformations/op_conversions/convert_negative.hpp>
#include <transformations/op_conversions/convert_broadcast_to_tiles.hpp>
#include <transformations/op_conversions/convert_mod.hpp>
#include <transformations/op_conversions/convert_divide.hpp>
#include <transformations/op_conversions/convert_minimum_to_power_and_max.hpp>
#include <transformations/op_conversions/convert_subtract.hpp>

#include <transformations/common_optimizations/conv_bias_fusion.hpp>
#include <transformations/op_conversions/convert_convolutions.hpp>
#include <transformations/op_conversions/convert_previous_nms_to_nms_5.hpp>

#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/manager.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

#include <memory>
#include <vector>

NGRAPH_RTTI_DEFINITION(ov::pass::ConvertOpSet1ToLegacy, "ConvertOpSet1ToLegacy", 0);

bool ov::pass::ConvertOpSet1ToLegacy::run_on_function(std::shared_ptr<ov::Function> f) {
    ov::pass::Manager manager(get_pass_config());

    manager.register_pass<ov::pass::ConstantFolding>();

    // Some passes before ConvertOpSet1ToLegacy can produce some of this
    // operations. So for convenience we decompose this operations here and
    // in CommonOptimizations.
    auto decomp = manager.register_pass<ov::pass::GraphRewrite>();
    decomp->add_matcher<ov::pass::ConvertMod>();
    decomp->add_matcher<ov::pass::ConvertMinimum>();
    decomp->add_matcher<ov::pass::ConvertSubtract>();
    decomp->add_matcher<ov::pass::ConvertDivide>();
    decomp->add_matcher<ov::pass::ConvertNegative>();
    decomp->set_name("ov::pass::LegacyDecompositions");

    auto convert_matmul = manager.register_pass<ov::pass::GraphRewrite>();
    convert_matmul->add_matcher<ov::pass::ConvertMatMulToFC>();
    convert_matmul->add_matcher<ov::pass::PullTransposeThroughFQUp>();
    convert_matmul->add_matcher<ov::pass::ConvertMatMulToGemm>();
    convert_matmul->set_name("ov::pass::ConvertMatMul");

    manager.register_pass<ov::pass::ConstantFolding>();

    // Convolution/Deconvolution/FullyConnected fusions
    manager.register_pass<ov::pass::ConvertConvolutions>();

    // Convolution/Deconvolution/FullyConnected fusions
    auto fusion = manager.register_pass<ov::pass::GraphRewrite>();
    fusion->add_matcher<ov::pass::ConvAddFusion>();
    fusion->add_matcher<ov::pass::DeconvAddFusion>();
    fusion->add_matcher<ov::pass::FullyConnectedBiasFusion>();
    fusion->set_name("ov::pass::BiasFusions");

    // CF is required after fusions
    manager.register_pass<ov::pass::ConstantFolding>();

    // List of passes that convert opset1 operations to legacy
    // plus transformations that are required by InferenceEngine
    // All this transformations can be executed simultaneously
    auto anchor = manager.register_pass<ov::pass::GraphRewrite>();
    anchor->add_matcher<ov::pass::ReshapeFullyConnected>();
    anchor->add_matcher<ov::pass::Reshape1DConvolution>();
    anchor->add_matcher<ov::pass::Reshape1DAvgPool>();
    anchor->add_matcher<ov::pass::Reshape1DMaxPool>();
    anchor->add_matcher<ov::pass::ConvertNormalizeL2WithMulToNormalizeIE>();
    anchor->add_matcher<ov::pass::ConvertHardSigmoidToLegacyMatcher>();
    anchor->add_matcher<ov::pass::ConvertProposalToLegacyMatcher>();
    anchor->add_matcher<ov::pass::ConvertProposal4ToLegacyMatcher>();
    anchor->add_matcher<ov::pass::ConvertBroadcastToTiles>();
    anchor->add_matcher<ov::pass::ConvertTileToLegacyMatcher>();
    anchor->add_matcher<ov::pass::ConvertLRNToLegacyMatcher>();
    anchor->add_matcher<ov::pass::ConvertPadToLegacyMatcher>();
    anchor->add_matcher<ov::pass::ConvertLSTMCellMatcher>();
    anchor->add_matcher<ov::pass::ConvertRNNCellMatcher>();
    anchor->add_matcher<ov::pass::ConvertGRUCellMatcher>();
    anchor->add_matcher<ov::pass::ConvertInterpolateToInterpOrResampleMatcher>();
    anchor->add_matcher<ov::pass::ConvertStridedSliceToCropMatcher>();
    anchor->add_matcher<ov::pass::ConvertPowerToPowerIEMatcher>();
    anchor->add_matcher<ov::pass::ConvertSqrtToPowerIEMatcher>();
    anchor->add_matcher<ov::pass::ConvertPReLUToReLUIE>();
    anchor->add_matcher<ov::pass::ConvertGatherToGatherIEMatcher>();
    anchor->add_matcher<ov::pass::ConvertSeluToSeluIEMatcher>();
    anchor->add_matcher<ov::pass::ConvertSwishToSwishIEMatcher>();
    anchor->add_matcher<ov::pass::ConvertOneHotToOneHotIEMatcher>()->detect_output_type(f);
    anchor->add_matcher<ov::pass::ConvertGatherTreeToGatherTreeIEMatcher>();
    anchor->add_matcher<ov::pass::ConvertTopKToTopKIEMatcher>();
    anchor->add_matcher<ov::pass::ConvertNMS1ToNMS5>();
    anchor->add_matcher<ov::pass::ConvertNMS3ToNMS5>();
    anchor->add_matcher<ov::pass::ConvertNMS4ToNMS5>();
    anchor->add_matcher<ov::pass::ConvertNMS5ToLegacyMatcher>();
    anchor->add_matcher<ov::pass::ConvertGRUSequenceMatcher>();
    anchor->add_matcher<ov::pass::ConvertRNNSequenceMatcher>();
    anchor->add_matcher<ov::pass::ConvertLSTMSequenceMatcher>();
    anchor->set_name("ov::pass::LegacyConversions");

    // List of final conversion transformations that must to be executed
    // after previous group of transformations

    if (!ov::op::util::has_op_with_type<ov::op::FakeQuantize>(f)) {
        manager.register_pass<ov::pass::ReshapeFullyConnectedFusion>();
    }
    manager.register_pass<ov::pass::ConvertNormalizeL2ToLegacyMatcher>();
    manager.register_pass<ov::pass::ConvertMulAddToScaleShiftOrPower>();
    manager.register_pass<ov::pass::ConvertMulOrAddFinally>();

    manager.register_pass<ov::pass::ConstantFolding>();

    manager.run_passes(f);

    // Returning value is false because pass::Manager always apply Validation pass
    // if function was changed. This helps to avoid excess Validations after applying
    // this pass. In future when we will return more meaningful status code it will be
    // replaced with real status reported by manager.run_passes() method call.
    return false;
}
