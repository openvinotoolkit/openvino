// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fuse_mamba2.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/mamba2.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/symbolic_transformations/symbolic_optimizations.hpp"

namespace pattern = ov::pass::pattern;
namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;

namespace {

bool matches_mamba2_loop(const std::shared_ptr<ov::Node>& node) {
    auto loop = ov::as_type_ptr<ov::op::v5::Loop>(node);
    if (!loop) {
        return false;
    }

    // External inputs: trip_count, exec_cond, dA, dBx, C, recurrent_state, output_buffer.
    // External outputs: output, output_recurrent_state.
    if (loop->get_input_size() != 7 || loop->get_output_size() != 2) {
        return false;
    }

    // Per-step body inputs (sliced over the sequence axis).
    auto output_buffer = pattern::any_input(pattern::shape_matches("[?, head_num, ?, head_dim]"));
    auto last_state = pattern::any_input(pattern::shape_matches("[?, head_num, head_dim, state_size]"));
    auto dA_t = pattern::any_input(pattern::shape_matches("[?, head_num, 1, 1, 1]"));
    auto dBx_t = pattern::any_input(pattern::shape_matches("[?, head_num, 1, head_dim, state_size]"));
    auto C_t = pattern::any_input(pattern::shape_matches("[?, head_num, 1, state_size]"));
    auto step_index = pattern::any_input();

    auto step_index_unsqueeze = pattern::wrap_type<v0::Unsqueeze>({step_index, 0});

    auto dA_squeezed = pattern::wrap_type<v0::Squeeze>({dA_t, {2}});
    auto dBx_squeezed = pattern::wrap_type<v0::Squeeze>({dBx_t, {2}});
    auto C_squeezed = pattern::wrap_type<v0::Squeeze>({C_t, {2}});

    // state_t = state_{t-1} * dA_t + dBx_t
    auto state_decay = pattern::wrap_type<v1::Multiply>({last_state, dA_squeezed});
    auto state_new = pattern::wrap_type<v1::Add>({state_decay, dBx_squeezed});

    // y_t = reduce_sum(state_t * unsqueeze(C_t), axis=state_size)
    auto C_unsqueeze = pattern::wrap_type<v0::Unsqueeze>({C_squeezed, {-2}});
    auto weighted_output = pattern::wrap_type<v1::Multiply>({state_new, C_unsqueeze});
    auto output_reduce_sum = pattern::wrap_type<v1::ReduceSum>({weighted_output, {-1}}, {{"keep_dims", false}});
    auto output_unsqueeze = pattern::wrap_type<v0::Unsqueeze>({output_reduce_sum, {2}});
    auto output_unsqueeze_conv = pattern::optional<v0::Convert>({output_unsqueeze});

    auto scatter_update_output =
        pattern::wrap_type<ov::op::v3::ScatterUpdate>({output_buffer, step_index_unsqueeze, output_unsqueeze_conv, 2});
    auto output_result = pattern::wrap_type<v0::Result>({scatter_update_output});

    auto state_new_conv = pattern::optional<v0::Convert>({state_new});
    auto state_result = pattern::wrap_type<v0::Result>({state_new_conv});

    ov::pass::pattern::Matcher loop_output_matcher(output_result);
    ov::pass::pattern::Matcher loop_state_matcher(state_result);
    auto body = loop->get_function();
    const auto& body_results = body->get_results();
    if (body_results.size() < 3) {
        return false;
    }

    // body_results: [0] = exec_condition, [1] = updated state, [2] = scattered output.
    if (!loop_output_matcher.match(body_results[2]->output(0))) {
        return false;
    }
    if (!loop_state_matcher.match(body_results[1]->output(0))) {
        return false;
    }
    return true;
}

}  // namespace

ov::pass::RemoveConcatSliceAfterLoopMamba2::RemoveConcatSliceAfterLoopMamba2() {
    auto dBx = pattern::any_input(pattern::shape_matches("[?, head_num, ?, head_dim, state_size]"));
    auto init_state = pattern::any_input(pattern::rank_equals(4));

    auto loop_inputs = ov::OutputVector{pattern::any_input(),
                                        pattern::any_input(),
                                        pattern::any_input(),
                                        dBx,
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
                                                           pattern::shape_matches("[?, head_num, ?, head_dim]"));
    auto slice_state = pattern::wrap_type<ov::op::v8::Slice>({concat_loop, out_numel, pattern::any_input(), {1}, {0}});
    auto restored_state =
        pattern::wrap_type<v1::Reshape>({slice_state, pattern::any_input()},
                                        pattern::shape_matches("[?, head_num, head_dim, state_size]"));

    auto restored_root = restored_output | restored_state;

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

    auto m = std::make_shared<ov::pass::pattern::Matcher>(restored_root, "RemoveConcatSliceAfterLoopMamba2");
    register_matcher(m, callback);
}

ov::pass::FuseMamba2Loop::FuseMamba2Loop() {
    auto dA = pattern::any_input(pattern::shape_matches("[?, head_num, ?, 1, 1]"));
    auto dBx = pattern::any_input(pattern::shape_matches("[?, head_num, ?, head_dim, state_size]"));
    auto C = pattern::any_input(pattern::shape_matches("[?, head_num, ?, state_size]"));
    auto init_state = pattern::any_input(pattern::shape_matches("[?, head_num, head_dim, state_size]"));

    auto loop_output =
        pattern::wrap_type<ov::op::v5::Loop>(ov::OutputVector{pattern::any_input(),  // trip count
                                                              pattern::any_input(),  // execution condition
                                                              dA,
                                                              dBx,
                                                              C,
                                                              init_state,
                                                              pattern::any_input()},  // output accumulator buffer
                                             [](std::shared_ptr<ov::Node> node) -> bool {
                                                 return matches_mamba2_loop(node);
                                             });

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto loop_node = pattern_map.at(loop_output).get_node_shared_ptr();

        ov::OutputVector inputs = {
            pattern_map.at(dA),         // dA
            pattern_map.at(dBx),        // dBx
            pattern_map.at(C),          // C
            pattern_map.at(init_state)  // recurrent_state
        };

        auto mamba2 = std::make_shared<ov::op::internal::Mamba2>(inputs);
        mamba2->set_friendly_name(loop_node->get_friendly_name());

        ov::copy_runtime_info(loop_node, mamba2);
        ov::replace_node(loop_node, mamba2);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(loop_output, "FuseMamba2Loop");
    register_matcher(m, callback);
}

bool ov::pass::Mamba2Fusion::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(Mamba2Fusion);
    ov::pass::SymbolicOptimizations symbolic_optimizations(false, get_pass_config());
    auto symbolic_ctx_manager = symbolic_optimizations.get_manager();
    symbolic_ctx_manager->register_pass<ov::pass::RemoveConcatSliceAfterLoopMamba2>();
    symbolic_ctx_manager->register_pass<ov::pass::FuseMamba2Loop>();
    return symbolic_optimizations.run_on_model(model);
}
