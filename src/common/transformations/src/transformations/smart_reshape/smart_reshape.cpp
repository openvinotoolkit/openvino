// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/smart_reshape/smart_reshape.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/common_optimizations/shared_ops_optimization.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/smart_reshape/broadcast_const_range_replacement.hpp"
#include "transformations/smart_reshape/lstm_states_broadcast.hpp"
#include "transformations/smart_reshape/matmul_sr.hpp"
#include "transformations/smart_reshape/proposal_scales_stridedslice.hpp"
#include "transformations/smart_reshape/reshape_sinking.hpp"
#include "transformations/smart_reshape/reshape_to_1D.hpp"
#include "transformations/smart_reshape/shape_of_const_folding.hpp"
#include "transformations/smart_reshape/strided_slice_squeeze.hpp"

bool ov::pass::SmartReshape::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_FUNCTION_SCOPE(SmartReshape);
    ov::pass::Manager static_manager("SmartReshape:static");
    // This pass must be called first in pipeline
    static_manager.register_pass<ov::pass::InitNodeInfo>();
    static_manager.register_pass<ov::pass::ReshapeTo1D>();
    static_manager.register_pass<ov::pass::Proposal1Scales>();
    static_manager.register_pass<ov::pass::Proposal4Scales>();
    static_manager.register_pass<ov::pass::SharedOpOptimization>();
    static_manager.register_pass<ov::pass::SqueezeStridedSlice>();
    static_manager.register_pass<ov::pass::StridedSliceSqueeze>();
    static_manager.register_pass<ov::pass::ReshapeTo1D>();
    static_manager.register_pass<ov::pass::TransposeMatMul>();
    static_manager.register_pass<ov::pass::BroadcastConstRangeReplacement>();
    static_manager.register_pass<ov::pass::LSTMStatesBroadcast>();
    static_manager.register_pass<ov::pass::ReshapeSinkingMatMul>();
    static_manager.run_passes(f);

    ov::pass::Manager dynamic_manager("SmartReshape:dynamic");
    // function revalidation will cause "fake" dynamism due to ShapeOf ops insertions
    // we turn it off to have access to originally static shapes
    dynamic_manager.set_per_pass_validation(false);
    dynamic_manager.register_pass<ov::pass::ReshapeAMatMul>();
    dynamic_manager.register_pass<ov::pass::ReshapeBMatMul>();
    dynamic_manager.register_pass<ov::pass::ShapeOfConstFolding>();
    dynamic_manager.run_passes(f);

    return true;
}
