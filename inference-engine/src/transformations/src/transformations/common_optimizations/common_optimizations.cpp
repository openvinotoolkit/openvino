// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "transformations/common_optimizations/algebraic_simplification.hpp"
#include "transformations/common_optimizations/nop_elimination.hpp"
#include "transformations/common_optimizations/common_optimizations.hpp"
#include "transformations/common_optimizations/conv_mul_fusion.hpp"
#include "transformations/common_optimizations/fq_mul_fusion.hpp"
#include "transformations/common_optimizations/fq_reshape_fusion.hpp"
#include "transformations/depth_to_space_fusion.hpp"
#include "transformations/optimize_strided_slice.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/itt.hpp"
#include "transformations/mish_fusion.hpp"
#include "transformations/softplus_fusion.hpp"
#include "transformations/softplus_to_mish_fusion.hpp"
#include "transformations/swish_fusion.hpp"
#include "transformations/normalize_l2_fusion.hpp"
#include "transformations/bidirectional_sequences_decomposition.hpp"
#include "transformations/convert_pad_to_group_conv.hpp"
#include "transformations/convert_divide.hpp"
#include "transformations/convert_quantize_dequantize.hpp"
#include "transformations/convert_mod.hpp"
#include "transformations/convert_minimum_to_power_and_max.hpp"
#include "transformations/convert_negative.hpp"
#include "transformations/convert_scatter_elements_to_scatter.hpp"
#include "transformations/convert_reduce_to_pooling.hpp"
#include "transformations/convert_subtract.hpp"
#include "transformations/convert_depth_to_space.hpp"
#include "transformations/convert_space_to_depth.hpp"
#include "transformations/convert_broadcast_to_tiles.hpp"
#include "transformations/convert_gelu.hpp"
#include "transformations/batch_norm_decomposition.hpp"
#include "transformations/pull_transpose_through_fq.hpp"
#include "transformations/lin_op_sequence_fusoin.hpp"
#include "transformations/reduce_l1_decomposition.hpp"
#include "transformations/reduce_l2_decomposition.hpp"
#include "transformations/remove_filtering_boxes_by_size.hpp"
#include "transformations/hswish_decomposition.hpp"
#include "transformations/hswish_fusion.hpp"
#include "transformations/itt.hpp"

#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/constant_folding.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::CommonOptimizations, "CommonOptimizations", 0);

bool ngraph::pass::CommonOptimizations::run_on_function(std::shared_ptr<ngraph::Function> f) {
    IETRANSFORM_SCOPE(CommonOptimizations,
        ngraph::pass::Manager manager;

        // This pass must be called first in pipeline
        REGISTER_PASS(manager, InitNodeInfo);
        REGISTER_PASS(manager, RemoveFilteringBoxesBySize); // Resolves dynamism (replaces NonZero), CF needed
        REGISTER_PASS(manager, RemoveFilteringBoxesBySize);
        REGISTER_PASS(manager, ConvertQuantizeDequantize);
        REGISTER_PASS(manager, ConstantFolding);
        REGISTER_PASS(manager, StridedSliceOptimization); // depends on CF
        REGISTER_PASS(manager, NopElimination); // may introduce fake dynamism
        REGISTER_PASS(manager, AlgebraicSimplification); // may introduce fake dynamism
        REGISTER_PASS(manager, ConstantFolding);
        REGISTER_PASS(manager, ConvertScatterElementsToScatter); // partially depends on CF
        REGISTER_PASS(manager, DepthToSpaceFusion);
        REGISTER_PASS(manager, MishFusion);
        REGISTER_PASS(manager, SoftPlusFusion);
        REGISTER_PASS(manager, SoftPlusToMishFusion);
        REGISTER_PASS(manager, SwishFusion);
        REGISTER_PASS(manager, HSwishFusion);
        REGISTER_PASS(manager, ConvertPadToGroupConvolution);
        REGISTER_PASS(manager, NormalizeL2Fusion);
        REGISTER_PASS(manager, BidirectionalLSTMSequenceDecomposition);
        REGISTER_PASS(manager, BidirectionalRNNSequenceDecomposition);
        REGISTER_PASS(manager, BidirectionalGRUSequenceDecomposition);


        auto decomp = manager.register_pass<GraphRewrite>();
        ADD_MATCHER(decomp, ReduceL1Decomposition);
        ADD_MATCHER(decomp, ReduceL2Decomposition);
        ADD_MATCHER(decomp, HSwishDecomposition);
        ADD_MATCHER(decomp, ConvertReduceMeanToPooling);
        ADD_MATCHER(decomp, ConvertReduceMaxToPooling);
        ADD_MATCHER(decomp, ConvertReduceSumToPooling);
        ADD_MATCHER(decomp, ConvertBroadcastToTiles);
        ADD_MATCHER(decomp, ConvertMod);
        ADD_MATCHER(decomp, ConvertGELU);
        ADD_MATCHER(decomp, ConvertMinimum);
        ADD_MATCHER(decomp, ConvertSubtract);
        ADD_MATCHER(decomp, ConvertDivide);
        ADD_MATCHER(decomp, ConvertNegative);
        ADD_MATCHER(decomp, ConvertDepthToSpace);
        ADD_MATCHER(decomp, ConvertSpaceToDepth);
        ADD_MATCHER(decomp, BatchNormDecomposition);
        decomp->set_name("ngraph::pass::CommonDecompositions");

        // CF is required after all decompositions
        REGISTER_PASS(manager, ConstantFolding);

        // LinOpSequenceFusion must be executed after all decompositions
        REGISTER_PASS(manager, LinOpSequenceFusion);
        REGISTER_PASS(manager, ConvolutionMultiplyFusion);
        REGISTER_PASS(manager, GroupConvolutionMultiplyFusion);
        REGISTER_PASS(manager, ConvolutionBackpropDataMultiplyFusion);
        REGISTER_PASS(manager, GroupConvolutionBackpropDataMultiplyFusion);
        REGISTER_PASS(manager, ConstantFolding);

        auto fq_fusions = manager.register_pass<GraphRewrite>();
        ADD_MATCHER(fq_fusions, FakeQuantizeMulFusion);
        ADD_MATCHER(fq_fusions, FakeQuantizeReshapeFusion);
        ADD_MATCHER(fq_fusions, PullTransposeThroughFQUp);
        fq_fusions->set_name("ngraph::pass::FakeQuantizeFusions");

        manager.set_callback(m_transformation_callback);
        manager.run_passes(f);
        return true;
    )
    NGRAPH_CHECK(false, "nGraph pass is not included into the selective build.");
}
