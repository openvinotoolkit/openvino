// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "transformations/init_node_info.hpp"
#include "transformations/itt.hpp"
#include "transformations/common_optimizations/algebraic_simplification.hpp"
#include "transformations/common_optimizations/nop_elimination.hpp"
#include "transformations/common_optimizations/common_optimizations.hpp"
#include "transformations/common_optimizations/conv_mul_fusion.hpp"
#include "transformations/common_optimizations/fq_mul_fusion.hpp"
#include "transformations/common_optimizations/fq_reshape_fusion.hpp"
#include "transformations/common_optimizations/depth_to_space_fusion.hpp"
#include "transformations/common_optimizations/optimize_strided_slice.hpp"
#include "transformations/common_optimizations/mish_fusion.hpp"
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
#include "transformations/op_conversions/bidirectional_sequences_decomposition.hpp"
#include "transformations/op_conversions/convert_pad_to_group_conv.hpp"
#include "transformations/op_conversions/convert_divide.hpp"
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
#include "transformations/op_conversions/reduce_l1_decomposition.hpp"
#include "transformations/op_conversions/reduce_l2_decomposition.hpp"
#include "transformations/op_conversions/hswish_decomposition.hpp"
#include "transformations/op_conversions/convert_previous_nms_to_nms_5.hpp"
#include "transformations/op_conversions/hsigmoid_decomposition.hpp"
#include "transformations/op_conversions/log_softmax_decomposition.hpp"

#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/constant_folding.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::CommonOptimizations, "CommonOptimizations", 0);

bool ngraph::pass::CommonOptimizations::run_on_function(std::shared_ptr<ngraph::Function> f) {
    ngraph::pass::Manager manager(get_pass_config());

    // This pass must be called first in pipeline
    manager.register_pass<ngraph::pass::InitNodeInfo>();
    manager.register_pass<ngraph::pass::RemoveFilteringBoxesBySize>(); // Resolves dynamism (replaces NonZero), CF needed
    manager.register_pass<ngraph::pass::ConvertQuantizeDequantize>();
    manager.register_pass<ngraph::pass::ConstantFolding>();
    manager.register_pass<ngraph::pass::StridedSliceOptimization>(); // depends on CF
    manager.register_pass<ngraph::pass::AlgebraicSimplification>(); // may introduce fake dynamism
    manager.register_pass<ngraph::pass::NopElimination>(); // may introduce fake dynamism
    manager.register_pass<ngraph::pass::ConstantFolding>();
    manager.register_pass<ngraph::pass::ConvertScatterElementsToScatter>(); // partially depends on CF
    manager.register_pass<ngraph::pass::DepthToSpaceFusion>();
    manager.register_pass<ngraph::pass::MishFusion>();
    manager.register_pass<ngraph::pass::SoftPlusFusion>();
    manager.register_pass<ngraph::pass::SoftPlusToMishFusion>();
    manager.register_pass<ngraph::pass::SwishFusion>();
    manager.register_pass<ngraph::pass::HSwishFusion>();
    manager.register_pass<ngraph::pass::HSigmoidFusion>();
    manager.register_pass<ngraph::pass::ConvertPadToGroupConvolution, false>();
    manager.register_pass<ngraph::pass::NormalizeL2Fusion>();

    auto decomp = manager.register_pass<ngraph::pass::GraphRewrite>();
    decomp->add_matcher<ngraph::pass::BidirectionalLSTMSequenceDecomposition>();
    decomp->add_matcher<ngraph::pass::BidirectionalRNNSequenceDecomposition>();
    decomp->add_matcher<ngraph::pass::BidirectionalGRUSequenceDecomposition>();
    decomp->add_matcher<ngraph::pass::ReduceL1Decomposition>();
    decomp->add_matcher<ngraph::pass::ReduceL2Decomposition>();
    decomp->add_matcher<ngraph::pass::HSwishDecomposition>();
    decomp->add_matcher<ngraph::pass::HSigmoidDecomposition>();
    decomp->add_matcher<ngraph::pass::LogSoftmaxDecomposition>();
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
    decomp->add_matcher<ngraph::pass::BatchNormV5Decomposition>();
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
    manager.register_pass<ngraph::pass::ConvertInterpolate1ToInterpolate4, false>();

    manager.register_pass<ngraph::pass::ConvertPreviousNMSToNMS5>();

    auto fq_fusions = manager.register_pass<ngraph::pass::GraphRewrite>();
    fq_fusions->add_matcher<ngraph::pass::FakeQuantizeMulFusion>();
    fq_fusions->add_matcher<ngraph::pass::FakeQuantizeReshapeFusion>();
    fq_fusions->add_matcher<ngraph::pass::PullTransposeThroughFQUp>();
    fq_fusions->set_name("ngraph::pass::FakeQuantizeFusions");

    manager.run_passes(f);
    return true;
}
