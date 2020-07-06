// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_opset1_to_legacy/convert_opset1_to_legacy.hpp"

#include <transformations/constant_eltwise_reduction.hpp>
#include <transformations/convert_broadcast_to_tiles.hpp>
#include <transformations/convert_opset1_to_legacy/convert_convolutions.hpp>
#include <transformations/convert_divide.hpp>
#include <transformations/convert_mod.hpp>
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
#include <transformations/convert_opset1_to_legacy/convert_tile_to_ie_tile.hpp>
#include <transformations/convert_opset1_to_legacy/convert_topk_to_topk_ie.hpp>
#include <transformations/convert_depth_to_space.hpp>
#include <transformations/convert_space_to_depth.hpp>
#include <transformations/batch_norm_decomposition.hpp>
#include <transformations/convert_opset1_to_legacy/conv_bias_fusion.hpp>
#include <transformations/convert_opset1_to_legacy/fc_bias_fusion.hpp>
#include <transformations/mul_add_squence_fusion.hpp>
#include <transformations/mul_add_verification.hpp>
#include <transformations/convert_opset1_to_legacy/reshape_fc_fusion.hpp>
#include <transformations/convert_opset1_to_legacy/reshape_1d_ops.hpp>
#include <transformations/convert_opset1_to_legacy/reshape_fully_connected.hpp>
#include <transformations/pull_transpose_through_fq.hpp>
#include <transformations/convert_opset1_to_legacy/convert_hard_sigmoid_to_hard_sigmoid_ie.hpp>
#include <transformations/lin_op_sequence_fusoin.hpp>

#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/manager.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

#include <memory>
#include <vector>

bool ngraph::pass::ConvertOpSet1ToLegacy::run_on_function(std::shared_ptr<ngraph::Function> f) {
    ngraph::pass::Manager manager;
    std::vector<std::shared_ptr<ngraph::pass::PassBase> > transforms;

    manager.register_pass<ngraph::pass::ConstantFolding>();
    manager.register_pass<ngraph::pass::ConvertReduceToPooling>();
    manager.register_pass<ngraph::pass::ConvertMod>();
    manager.register_pass<ngraph::pass::ConvertMinimum>();
    manager.register_pass<ngraph::pass::ConvertSubtract>();
    manager.register_pass<ngraph::pass::ConvertDivide>();
    manager.register_pass<ngraph::pass::ConvertNegative>();
    manager.register_pass<ngraph::pass::ConvertDepthToSpace>();
    manager.register_pass<ngraph::pass::ConvertSpaceToDepth>();
    manager.register_pass<ngraph::pass::ConvertConvolutions>();
    manager.register_pass<ngraph::pass::BatchNormDecomposition>();
    manager.register_pass<ngraph::pass::ConstantFolding>();
    manager.register_pass<ngraph::pass::LinOpSequenceFusion>();
    manager.register_pass<ngraph::pass::ConvertMatMulToFCorGemm>();
    manager.register_pass<ngraph::pass::PullTransposeThroughFQUp>();
    manager.register_pass<ngraph::pass::ConstantFolding>();
    manager.register_pass<ngraph::pass::ConvFusion>();
    manager.register_pass<ngraph::pass::FullyConnectedBiasFusion>();
    manager.register_pass<ngraph::pass::ConstantFolding>();
    manager.register_pass<ngraph::pass::ReshapeFullyConnected>();
    manager.register_pass<ngraph::pass::ReshapeFullyConnectedFusion>();
    manager.register_pass<ngraph::pass::Reshape1DOps>();
    manager.register_pass<ngraph::pass::ConvertNormalizeL2WithMulToNormalizeIE>();
    manager.register_pass<ngraph::pass::ConstantEltwiseReduction>();
    manager.register_pass<ngraph::pass::ConvertMulAddToScaleShiftOrPower>();
    manager.register_pass<ngraph::pass::ConvertMulOrAddFinally>();
    manager.register_pass<ngraph::pass::ConstantFolding>();
    manager.register_pass<ngraph::pass::ConvertBroadcastToTiles>();

    auto anchor = manager.register_pass<ngraph::pass::GraphRewrite>();
    anchor->add_matcher<ngraph::pass::ConvertNormalizeL2ToLegacyMatcher>();
    anchor->add_matcher<ngraph::pass::ConvertHardSigmoidToLegacyMatcher>();
    anchor->add_matcher<ngraph::pass::ConvertProposalToLegacyMatcher>();
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
    anchor->add_matcher<ngraph::pass::ConvertPReLUToReLUIEMatcher>();
    anchor->add_matcher<ngraph::pass::ConvertGatherToGatherIEMatcher>();
    anchor->add_matcher<ngraph::pass::ConvertSeluToSeluIEMatcher>();
    anchor->add_matcher<ngraph::pass::ConvertOneHotToOneHotIEMatcher>()->detect_output_type(f);
    anchor->add_matcher<ngraph::pass::ConvertGatherTreeToGatherTreeIEMatcher>();
    anchor->add_matcher<ngraph::pass::ConvertTopKToTopKIEMatcher>();
    anchor->add_matcher<ngraph::pass::ConvertNMSToNMSIEMatcher>();
    anchor->add_matcher<ngraph::pass::ConvertNMS4ToLegacyMatcher>();
    anchor->set_name("ngraph::pass::ConvertOpSet1ToLegacy");

    manager.register_pass<ngraph::pass::ConstantFolding>();

    manager.set_callback(m_transformation_callback);
    manager.run_passes(f);
    return true;
}