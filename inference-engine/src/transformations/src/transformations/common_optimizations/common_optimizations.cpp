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

#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/constant_folding.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::CommonOptimizations, "CommonOptimizations", 0);

bool ngraph::pass::CommonOptimizations::run_on_function(std::shared_ptr<ngraph::Function> f) {
    OV_ITT_SCOPED_TASK(itt::domains::IETransform, "ngraph::pass::CommonOptimizations");

    ngraph::pass::Manager manager;

    // This pass must be called first in pipeline
    manager.register_pass<ngraph::pass::InitNodeInfo>();
    manager.register_pass<ngraph::pass::RemoveFilteringBoxesBySize>(); // Resolves dynamism (replaces NonZero), CF needed
    manager.register_pass<ngraph::pass::ConvertQuantizeDequantize>();
    manager.register_pass<ngraph::pass::ConstantFolding>();
    manager.register_pass<ngraph::pass::StridedSliceOptimization>(); // depends on CF
    manager.register_pass<ngraph::pass::NopElimination>(); // may introduce fake dynamism
    manager.register_pass<ngraph::pass::AlgebraicSimplification>(); // may introduce fake dynamism
    manager.register_pass<ngraph::pass::ConstantFolding>();
    manager.register_pass<ngraph::pass::ConvertScatterElementsToScatter>(); // partially depends on CF
    manager.register_pass<ngraph::pass::DepthToSpaceFusion>();
    manager.register_pass<ngraph::pass::MishFusion>();
    manager.register_pass<ngraph::pass::SoftPlusFusion>();
    manager.register_pass<ngraph::pass::SoftPlusToMishFusion>();
    manager.register_pass<ngraph::pass::SwishFusion>();
    manager.register_pass<ngraph::pass::HSwishFusion>();
    manager.register_pass<ngraph::pass::ConvertPadToGroupConvolution>();
    manager.register_pass<ngraph::pass::NormalizeL2Fusion>();
    manager.register_pass<ngraph::pass::BidirectionalLSTMSequenceDecomposition>();
    manager.register_pass<ngraph::pass::BidirectionalRNNSequenceDecomposition>();
    manager.register_pass<ngraph::pass::BidirectionalGRUSequenceDecomposition>();

    auto decomp = manager.register_pass<ngraph::pass::GraphRewrite>();
    decomp->add_matcher<ngraph::pass::ReduceL1Decomposition>();
    decomp->add_matcher<ngraph::pass::ReduceL2Decomposition>();
    decomp->add_matcher<ngraph::pass::HSwishDecomposition>();
    decomp->add_matcher<ngraph::pass::ConvertReduceMeanToPooling>();
    decomp->add_matcher<ngraph::pass::ConvertReduceMaxToPooling>();
    decomp->add_matcher<ngraph::pass::ConvertReduceSumToPooling>();
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
    decomp->set_name("ngraph::pass::CommonDecompositions");

    // CF is required after all decompositions
    manager.register_pass<ngraph::pass::ConstantFolding>();

    // LinOpSequenceFusion must be executed after all decompositions
    manager.register_pass<ngraph::pass::LinOpSequenceFusion>();

    manager.register_pass<ngraph::pass::ConvolutionMultiplyFusion>();
    manager.register_pass<ngraph::pass::GroupConvolutionMultiplyFusion>();
    manager.register_pass<ngraph::pass::ConvolutionBackpropDataMultiplyFusion>();
    manager.register_pass<ngraph::pass::GroupConvolutionBackpropDataMultiplyFusion>();
    manager.register_pass<ngraph::pass::ConstantFolding>();

    auto fq_fusions = manager.register_pass<ngraph::pass::GraphRewrite>();
    fq_fusions->add_matcher<ngraph::pass::FakeQuantizeMulFusion>();
    fq_fusions->add_matcher<ngraph::pass::FakeQuantizeReshapeFusion>();
    fq_fusions->add_matcher<ngraph::pass::PullTransposeThroughFQUp>();
    fq_fusions->set_name("ngraph::pass::FakeQuantizeFusions");

    manager.set_callback(m_transformation_callback);
    manager.run_passes(f);
    return true;
}
