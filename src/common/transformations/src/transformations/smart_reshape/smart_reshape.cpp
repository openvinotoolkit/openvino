// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <ngraph/pass/manager.hpp>
#include <transformations/common_optimizations/remove_concat_zero_dim_input.hpp>
#include <transformations/common_optimizations/remove_multi_subgraph_op_dangling_params.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/smart_reshape/broadcast_const_range_replacement.hpp>
#include <transformations/smart_reshape/matmul_sr.hpp>
#include <transformations/smart_reshape/mimic_set_batch_size.hpp>
#include <transformations/smart_reshape/proposal_scales_stridedslice.hpp>
#include <transformations/smart_reshape/reshape_to_1D.hpp>
#include <transformations/smart_reshape/smart_reshape.hpp>
#include <transformations/smart_reshape/strided_slice_squeeze.hpp>

#include "itt.hpp"

bool ngraph::pass::SmartReshape::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    // TODO: enable conditional compile
    // RUN_ON_FUNCTION_SCOPE(SmartReshape);
    ngraph::pass::Manager static_manager;
    // This pass must be called first in pipeline
    static_manager.register_pass<ngraph::pass::InitNodeInfo>();
    static_manager.register_pass<ngraph::pass::ReshapeTo1D>();
    static_manager.register_pass<ngraph::pass::Proposal1Scales>();
    static_manager.register_pass<ngraph::pass::Proposal4Scales>();
    static_manager.register_pass<ngraph::pass::SharedSqueeze>();
    static_manager.register_pass<ngraph::pass::SqueezeStridedSlice>();
    static_manager.register_pass<ngraph::pass::StridedSliceSqueeze>();
    static_manager.register_pass<ngraph::pass::ReshapeTo1D>();
    static_manager.register_pass<ngraph::pass::TransposeMatMul>();
    static_manager.register_pass<ngraph::pass::BroadcastConstRangeReplacement>();
    static_manager.register_pass<ov::pass::RemoveConcatZeroDimInput>();
    static_manager.register_pass<ov::pass::RemoveMultiSubGraphOpDanglingParams>();
    static_manager.run_passes(f);

    ngraph::pass::Manager dynamic_manager;
    // function revalidation will cause "fake" dynamism due to ShapeOf ops insertions
    // we turn it off to have access to originally static shapes
    dynamic_manager.set_per_pass_validation(false);
    dynamic_manager.register_pass<ngraph::pass::ReshapeAMatMul>();
    dynamic_manager.register_pass<ngraph::pass::ReshapeBMatMul>();
    dynamic_manager.run_passes(f);
    return true;
}
