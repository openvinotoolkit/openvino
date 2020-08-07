// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "transformations/common_optimizations/common_optimizations.hpp"
#include "transformations/depth_to_space_fusion.hpp"
#include "transformations/optimize_strided_slice.hpp"
#include "transformations/convert_scatter_elements_to_scatter.hpp"
#include "transformations/remove_filtering_boxes_by_size.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/mul_add_squence_fusion.hpp"

#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/nop_elimination.hpp>
#include <ngraph/pass/algebraic_simplification.hpp>
#include <ngraph/pass/constant_folding.hpp>

#include <transformations/convert_broadcast_to_tiles.hpp>
#include <transformations/convert_divide.hpp>
#include <transformations/convert_mod.hpp>
#include <transformations/convert_minimum_to_power_and_max.hpp>
#include <transformations/convert_negative.hpp>
#include <transformations/convert_reduce_to_pooling.hpp>
#include <transformations/convert_subtract.hpp>
#include <transformations/convert_depth_to_space.hpp>
#include <transformations/convert_space_to_depth.hpp>
#include <transformations/batch_norm_decomposition.hpp>
#include <transformations/pull_transpose_through_fq.hpp>
#include <transformations/lin_op_sequence_fusoin.hpp>
#include <transformations/convert_opset1_to_legacy/conv_bias_fusion.hpp>


bool ngraph::pass::CommonOptimizations::run_on_function(std::shared_ptr<ngraph::Function> f) {
    ngraph::pass::Manager manager;

    // This pass must be called first in pipeline
    manager.register_pass<ngraph::pass::InitNodeInfo>();
    manager.register_pass<ngraph::pass::RemoveFilteringBoxesBySize>(); // Resolves dynamism (replaces NonZero), CF needed
    manager.register_pass<ngraph::pass::ConstantFolding>();
    manager.register_pass<ngraph::pass::StridedSliceOptimization>(); // depends on CF
    manager.register_pass<ngraph::pass::NopElimination>(); // may introduce fake dynamism
    manager.register_pass<ngraph::pass::AlgebraicSimplification>(); // may introduce fake dynamism
    manager.register_pass<ngraph::pass::ConstantFolding>();
    manager.register_pass<ngraph::pass::ConvertScatterElementsToScatter>(); // partially depends on CF
    manager.register_pass<ngraph::pass::DepthToSpaceFusion>();

    auto decomp = manager.register_pass<ngraph::pass::GraphRewrite>();

    decomp->add_matcher<ngraph::pass::ConvertBroadcastToTiles>();
    decomp->add_matcher<ngraph::pass::ConvertReduceMeanToPooling>();
    decomp->add_matcher<ngraph::pass::ConvertReduceMaxToPooling>();
    decomp->add_matcher<ngraph::pass::ConvertReduceSumToPooling>();
    decomp->add_matcher<ngraph::pass::ConvertMod>();
    decomp->add_matcher<ngraph::pass::ConvertMinimum>();
    decomp->add_matcher<ngraph::pass::ConvertSubtract>();
    decomp->add_matcher<ngraph::pass::ConvertDivide>();
    decomp->add_matcher<ngraph::pass::ConvertNegative>();
    decomp->add_matcher<ngraph::pass::ConvertDepthToSpace>();
    decomp->add_matcher<ngraph::pass::ConvertSpaceToDepth>();
    decomp->add_matcher<ngraph::pass::BatchNormDecomposition>();
    decomp->add_matcher<ngraph::pass::PullTransposeThroughFQUp>();

    // LinOpSequenceFusion must be executed after all decompositions
    manager.register_pass<ngraph::pass::LinOpSequenceFusion>();

    // CF is required after all decompositions
    manager.register_pass<ngraph::pass::ConstantFolding>();

    manager.register_pass<ngraph::pass::ConvMultiplyFusion>();

    manager.set_callback(m_transformation_callback);
    manager.run_passes(f);
    return true;
}
