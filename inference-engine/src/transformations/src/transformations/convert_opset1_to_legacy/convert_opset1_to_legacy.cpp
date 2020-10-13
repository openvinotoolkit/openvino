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
    IETRANSFORM_SCOPE(ConvertOpSet1ToLegacy,
        ngraph::pass::Manager manager;

        REGISTER_PASS(manager, ConstantFolding);

        // Some passes before ConvertOpSet1ToLegacy can produce some of this
        // operations. So for convenience we decompose this operations here and
        // in CommonOptimizations.
        auto decomp = manager.register_pass<GraphRewrite>();
        ADD_MATCHER(decomp, ConvertMod);
        ADD_MATCHER(decomp, ConvertMinimum);
        ADD_MATCHER(decomp, ConvertSubtract);
        ADD_MATCHER(decomp, ConvertDivide);
        ADD_MATCHER(decomp, ConvertNegative);
        decomp->set_name("ngraph::pass::LegacyDecompositions");

        auto convert_matmul = manager.register_pass<GraphRewrite>();
        ADD_MATCHER(convert_matmul, ConvertMatMulToFC);
        ADD_MATCHER(convert_matmul, PullTransposeThroughFQUp);
        ADD_MATCHER(convert_matmul, ConvertMatMulToGemm);
        convert_matmul->set_name("ngraph::pass::ConvertMatMul");

        REGISTER_PASS(manager, ConstantFolding);

        // Convolution/Deconvolution/FullyConnected fusions
        auto convert_convolutions = manager.register_pass<GraphRewrite>();
        ADD_MATCHER(convert_convolutions, ConvertConvolution);
        ADD_MATCHER(convert_convolutions, ConvertGroupConvolution);
        ADD_MATCHER(convert_convolutions, ConvertDeconvolution);
        ADD_MATCHER(convert_convolutions, ConvertGroupDeconvolution);
        convert_convolutions->set_name("ngraph::pass::ConvertConvolutions");

        // Convolution/Deconvolution/FullyConnected fusions
        auto fusion = manager.register_pass<GraphRewrite>();
        ADD_MATCHER(fusion, ConvAddFusion);
        ADD_MATCHER(fusion, DeconvAddFusion);
        ADD_MATCHER(fusion, FullyConnectedBiasFusion);

        fusion->set_name("ngraph::pass::BiasFusions");

        // CF is required after fusions
        REGISTER_PASS(manager, ConstantFolding);

        // List of passes that convert opset1 operations to legacy
        // plus transformations that are required by InferenceEngine
        // All this transformations can be executed simultaneously
        auto anchor = manager.register_pass<GraphRewrite>();
        ADD_MATCHER(anchor, ReshapeFullyConnected);
        ADD_MATCHER(anchor, Reshape1DConvolution);
        ADD_MATCHER(anchor, Reshape1DAvgPool);
        ADD_MATCHER(anchor, Reshape1DMaxPool);
        ADD_MATCHER(anchor, ConvertNormalizeL2WithMulToNormalizeIE);
        ADD_MATCHER(anchor, ConvertHardSigmoidToLegacyMatcher);
        ADD_MATCHER(anchor, ConvertProposalToLegacyMatcher);
        ADD_MATCHER(anchor, ConvertProposal4ToLegacyMatcher);
        ADD_MATCHER(anchor, ConvertBroadcastToTiles);
        ADD_MATCHER(anchor, ConvertTileToLegacyMatcher);
        ADD_MATCHER(anchor, ConvertLRNToLegacyMatcher);
        ADD_MATCHER(anchor, ConvertPadToLegacyMatcher);
        ADD_MATCHER(anchor, ConvertLSTMCellMatcher);
        ADD_MATCHER(anchor, ConvertRNNCellMatcher);
        ADD_MATCHER(anchor, ConvertGRUCellMatcher);
        ADD_MATCHER(anchor, ConvertInterpolateToInterpOrResampleMatcher);
        ADD_MATCHER(anchor, ConvertStridedSliceToCropMatcher);
        ADD_MATCHER(anchor, ConvertPowerToPowerIEMatcher);
        ADD_MATCHER(anchor, ConvertSqrtToPowerIEMatcher);
        ADD_MATCHER(anchor, ConvertPReLUToReLUIE);
        ADD_MATCHER(anchor, ConvertGatherToGatherIEMatcher);
        ADD_MATCHER(anchor, ConvertSeluToSeluIEMatcher);
        ADD_MATCHER(anchor, ConvertSwishToSwishIEMatcher);
        auto convert_one_hot = ADD_MATCHER(anchor, ConvertOneHotToOneHotIEMatcher);
        if (convert_one_hot)
            convert_one_hot->detect_output_type(f);
        ADD_MATCHER(anchor, ConvertGatherTreeToGatherTreeIEMatcher);
        ADD_MATCHER(anchor, ConvertTopKToTopKIEMatcher);
        ADD_MATCHER(anchor, ConvertNMSToNMSIEMatcher);
        ADD_MATCHER(anchor, ConvertNMS4ToLegacyMatcher);
        ADD_MATCHER(anchor, ConvertGRUSequenceMatcher);
        ADD_MATCHER(anchor, ConvertRNNSequenceMatcher);
        ADD_MATCHER(anchor, ConvertLSTMSequenceMatcher);
        anchor->set_name("ngraph::pass::LegacyConversions");

        // List of final conversion transformations that must to be executed
        // after previous group of transformations
        REGISTER_PASS(manager, ReshapeFullyConnectedFusion);
        REGISTER_PASS(manager, ConvertNormalizeL2ToLegacyMatcher);
        REGISTER_PASS(manager, ConvertMulAddToScaleShiftOrPower);
        REGISTER_PASS(manager, ConvertMulOrAddFinally);
        REGISTER_PASS(manager, ConstantFolding);

        manager.set_callback(m_transformation_callback);
        manager.run_passes(f);
        return true;
    )
    NGRAPH_CHECK(false, "nGraph pass is not included into the selective build.");
}
