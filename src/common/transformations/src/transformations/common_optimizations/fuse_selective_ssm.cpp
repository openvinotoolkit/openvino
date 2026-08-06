// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fuse_selective_ssm.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/selective_ssm.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/symbolic_transformations/symbolic_optimizations.hpp"

namespace pattern = ov::pass::pattern;
namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;

namespace {

bool match_selective_ssm_body(const std::shared_ptr<ov::Node>& node) {
    auto loop = ov::as_type_ptr<ov::op::v5::Loop>(node);
    if (!loop || loop->get_input_size() != 8 || loop->get_output_size() != 2) {
        return false;
    }

    auto dA_t = pattern::any_input();
    auto dtB_t = pattern::any_input();
    auto x_t = pattern::any_input();
    auto C_t = pattern::any_input();
    auto last_state = pattern::any_input();
    auto core_out = pattern::any_input();
    auto step_index = pattern::any_input();

    auto axis1 = pattern::any_input();
    auto dtB_sq = pattern::wrap_type<v0::Squeeze>({dtB_t, axis1});
    auto x_sq = pattern::wrap_type<v0::Squeeze>({x_t, axis1});
    auto C_sq = pattern::wrap_type<v0::Squeeze>({C_t, axis1});
    auto dA_sq = pattern::wrap_type<v0::Squeeze>({dA_t, axis1});

    auto dA_4d = pattern::wrap_type<v0::Unsqueeze>({pattern::wrap_type<v0::Unsqueeze>({dA_sq, -1}), -1});
    auto dBx = pattern::wrap_type<v1::Multiply>({pattern::wrap_type<v0::Unsqueeze>({dtB_sq, -2}),
                                                 pattern::wrap_type<v0::Unsqueeze>({x_sq, -1})});
    auto state_decay = pattern::wrap_type<v1::Multiply>({last_state, dA_4d});
    auto state_new = pattern::wrap_type<v1::Add>({state_decay, dBx});
    auto weighted_output =
        pattern::wrap_type<v1::Multiply>({state_new, pattern::wrap_type<v0::Unsqueeze>({C_sq, -2})});
    auto output_reduce_sum = pattern::wrap_type<v1::ReduceSum>({weighted_output, -1}, {{"keep_dims", false}});
    auto output_unsqueeze = pattern::wrap_type<v0::Unsqueeze>({output_reduce_sum, 1});
    auto output_unsqueeze_conv = pattern::optional<v0::Convert>({output_unsqueeze});
    auto step_index_unsqueeze = pattern::wrap_type<v0::Unsqueeze>({step_index, 0});
    auto scatter_update_output =
        pattern::wrap_type<ov::op::v3::ScatterUpdate>({core_out, step_index_unsqueeze, output_unsqueeze_conv, 1});
    auto output_result = pattern::wrap_type<v0::Result>({scatter_update_output});
    auto state_new_conv = pattern::optional<v0::Convert>({state_new});
    auto state_result = pattern::wrap_type<v0::Result>({state_new_conv});

    auto body = loop->get_function();
    const auto& body_results = body->get_results();
    if (body_results.size() < 3) {
        return false;
    }

    ov::pass::pattern::Matcher loop_output_matcher(output_result);
    if (!loop_output_matcher.match(body_results[2]->output(0))) {
        return false;
    }
    ov::pass::pattern::Matcher loop_state_matcher(state_result);
    return loop_state_matcher.match(body_results[1]->output(0));
}

}  // namespace

namespace ov::pass {

RemoveConcatSliceAfterLoopSelectiveSSM::RemoveConcatSliceAfterLoopSelectiveSSM() {
    auto x = pattern::any_input(pattern::shape_matches("[?, ?, head_num, head_dim]"));
    auto init_state = pattern::any_input(pattern::rank_equals(4));

    auto loop_inputs = ov::OutputVector{pattern::any_input(),
                                        pattern::any_input(),
                                        pattern::any_input(),
                                        pattern::any_input(),
                                        x,
                                        pattern::any_input(),
                                        init_state,
                                        pattern::any_input()};

    auto loop_output0 = pattern::wrap_type<ov::op::v5::Loop>(loop_inputs, pattern::output_index_matches(0));
    auto loop_output1 = pattern::wrap_type<ov::op::v5::Loop>(loop_inputs, pattern::output_index_matches(1));

    auto reshape_output = pattern::wrap_type<v1::Reshape>({loop_output0, {-1}});
    auto reshape_state = pattern::wrap_type<v1::Reshape>({loop_output1, {-1}});
    auto concat_loop = pattern::wrap_type<v0::Concat>({reshape_output, reshape_state}, {{"axis", 0}});
    auto out_numel = pattern::any_input(pattern::has_static_shape());
    auto slice_output = pattern::wrap_type<ov::op::v8::Slice>({concat_loop, {0}, out_numel, {1}, {0}});
    auto restored_output = pattern::wrap_type<v1::Reshape>({slice_output, pattern::any_input()},
                                                           pattern::shape_matches("[?, ?, head_num, head_dim]"));
    auto slice_state = pattern::wrap_type<ov::op::v8::Slice>({concat_loop, out_numel, pattern::any_input(), {1}, {0}});
    auto restored_state = pattern::wrap_type<v1::Reshape>({slice_state, pattern::any_input()},
                                                          pattern::shape_matches("[?, head_num, head_dim, state_size]"));

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        bool changed = false;
        auto loop_node = pattern_map.at(loop_output0).get_node_shared_ptr();
        if (pattern_map.count(restored_output)) {
            auto restored_output_out = pattern_map.at(restored_output);
            if (!ov::replace_output_update_name(restored_output_out, loop_node->output(0))) {
                restored_output_out.replace(loop_node->output(0));
            }
            changed = true;
        }
        if (pattern_map.count(restored_state)) {
            auto restored_state_out = pattern_map.at(restored_state);
            if (!ov::replace_output_update_name(restored_state_out, loop_node->output(1))) {
                restored_state_out.replace(loop_node->output(1));
            }
            changed = true;
        }
        return changed;
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(restored_output | restored_state,
                                                                "RemoveConcatSliceAfterLoopSelectiveSSM");
    register_matcher(matcher, callback);
}

FuseSelectiveSSMLoop::FuseSelectiveSSMLoop() {
    auto A = pattern::any_input(pattern::shape_matches("[head_num]"));
    auto dt = pattern::any_input(pattern::shape_matches("[?, ?, head_num]"));
    auto B = pattern::any_input(pattern::shape_matches("[?, ?, group_num, state_size]"));
    auto x = pattern::any_input(pattern::shape_matches("[?, ?, head_num, head_dim]"));
    auto C = pattern::any_input(pattern::shape_matches("[?, ?, group_num, state_size]"));
    auto init_state = pattern::any_input(pattern::shape_matches("[?, head_num, head_dim, state_size]"));

    auto B_expanded = pattern::wrap_type<v1::Reshape>(
        {pattern::wrap_type<v0::Tile>({pattern::wrap_type<v0::Unsqueeze>({B, 3}), pattern::any_input()}),
         pattern::any_input()});
    auto C_expanded = pattern::wrap_type<v1::Reshape>(
        {pattern::wrap_type<v0::Tile>({pattern::wrap_type<v0::Unsqueeze>({C, 3}), pattern::any_input()}),
         pattern::any_input()});
    auto dA = pattern::wrap_type<v0::Exp>({pattern::wrap_type<v1::Multiply>({A, dt})});
    auto dtB = pattern::wrap_type<v1::Multiply>({pattern::wrap_type<v0::Unsqueeze>({dt, -1}), B_expanded});

    auto loop_output = pattern::wrap_type<ov::op::v5::Loop>(
        ov::OutputVector{pattern::any_input(),
                         pattern::any_input(),
                         dA,
                         dtB,
                         x,
                         C_expanded,
                         init_state,
                         pattern::any_input()},
        [](const std::shared_ptr<ov::Node>& node) {
            return match_selective_ssm_body(node);
        });

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto loop_node = pattern_map.at(loop_output).get_node_shared_ptr();

        auto selective_ssm = std::make_shared<ov::op::internal::SelectiveSSM>(ov::OutputVector{pattern_map.at(A),
                                                                                                 pattern_map.at(dt),
                                                                                                 pattern_map.at(B),
                                                                                                 pattern_map.at(x),
                                                                                                 pattern_map.at(C),
                                                                                                 pattern_map.at(init_state)});
        selective_ssm->set_friendly_name(loop_node->get_friendly_name());
        ov::copy_runtime_info(loop_node, selective_ssm);
        ov::replace_node(loop_node, selective_ssm);
        return true;
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(loop_output, "FuseSelectiveSSMLoop");
    register_matcher(matcher, callback);
}

bool SelectiveSSMFusion::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(SelectiveSSMFusion);
    ov::pass::SymbolicOptimizations symbolic_optimizations(false, get_pass_config());
    auto symbolic_ctx_manager = symbolic_optimizations.get_manager();
    symbolic_ctx_manager->register_pass<ov::pass::RemoveConcatSliceAfterLoopSelectiveSSM>();
    symbolic_ctx_manager->register_pass<ov::pass::FuseSelectiveSSMLoop>();
    return symbolic_optimizations.run_on_model(model);
}

}  // namespace ov::pass
