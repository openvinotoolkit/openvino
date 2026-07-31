// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fuse_mamba2.hpp"

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
#include "openvino/op/mamba2.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scatter_update.hpp"
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

// Result of matching the per-step Mamba2 recurrence inside a Loop body. On success `A` carries the
// per-head log-decay operand feeding `dA_t = exp(A * dt_t)`, so it can be materialized as the `A`
// input of the fused op.
struct Mamba2BodyMatch {
    bool matched = false;
    ov::Output<ov::Node> A;
};

// Matches the time-major Mamba2 recurrence body (discretization performed inside the loop).
// Per-step body semantics (H = num_heads, P = head_dim, N = state_size):
//   dt_t = Squeeze(dt_t, 1); B_t = Squeeze(B_t, 1); x_t = Squeeze(x_t, 1); C_t = Squeeze(C_t, 1)
//   dA_t    = exp(Unsqueeze(A, 0) * dt_t)
//   dBx_t   = Unsqueeze(dt_t * B_t, -2) * Unsqueeze(x_t, -1)
//   state_t = state_{t-1} * dA_t + dBx_t
//   y_t     = reduce_sum(state_t * Unsqueeze(C_t, -2), axis=N)
//   ScatterUpdate(core_out, t, y_t, axis=1)
Mamba2BodyMatch match_mamba2_body(const std::shared_ptr<ov::Node>& node) {
    Mamba2BodyMatch result;

    auto loop = ov::as_type_ptr<ov::op::v5::Loop>(node);
    if (!loop) {
        return result;
    }

    // External inputs: trip_count, exec_cond, dt, B, x, C, recurrent_state, output_buffer.
    // External outputs: output, output_recurrent_state.
    if (loop->get_input_size() != 8 || loop->get_output_size() != 2) {
        return result;
    }

    auto dt_t = pattern::any_input();
    auto B_t = pattern::any_input();
    auto x_t = pattern::any_input();
    auto C_t = pattern::any_input();
    auto last_state = pattern::any_input();
    auto core_out = pattern::any_input();
    auto step_index = pattern::any_input();
    auto A = pattern::any_input();

    // Drop the singleton sequence axis introduced by slicing.
    auto dt_squeezed = pattern::wrap_type<v0::Squeeze>({dt_t, 1});
    auto B_squeezed = pattern::wrap_type<v0::Squeeze>({B_t, 1});
    auto x_squeezed = pattern::wrap_type<v0::Squeeze>({x_t, 1});
    auto C_squeezed = pattern::wrap_type<v0::Squeeze>({C_t, 1});

    // dA_t = exp(A * dt_t) -> [B, H, 1, 1]
    auto A_unsqueeze = pattern::wrap_type<v0::Unsqueeze>({A, 0});
    auto dA = pattern::wrap_type<v1::Multiply>({A_unsqueeze, dt_squeezed});
    auto dA_exp = pattern::wrap_type<v0::Exp>({dA});
    auto dA_4d = pattern::wrap_type<v0::Unsqueeze>({pattern::wrap_type<v0::Unsqueeze>({dA_exp, -1}), -1});

    // dBx_t = (dt_t * B_t) outer x_t -> [B, H, P, N]
    auto dt_B = pattern::wrap_type<v1::Multiply>({pattern::wrap_type<v0::Unsqueeze>({dt_squeezed, -1}), B_squeezed});
    auto dBx = pattern::wrap_type<v1::Multiply>({pattern::wrap_type<v0::Unsqueeze>({dt_B, -2}),
                                                 pattern::wrap_type<v0::Unsqueeze>({x_squeezed, -1})});

    // state_t = state_{t-1} * dA_t + dBx_t
    auto state_decay = pattern::wrap_type<v1::Multiply>({last_state, dA_4d});
    auto state_new = pattern::wrap_type<v1::Add>({state_decay, dBx});

    // y_t = reduce_sum(state_t * unsqueeze(C_t), axis=N)
    auto C_unsqueeze = pattern::wrap_type<v0::Unsqueeze>({C_squeezed, -2});
    auto weighted_output = pattern::wrap_type<v1::Multiply>({state_new, C_unsqueeze});
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
        return result;
    }

    // body_results: [0] = exec_condition, [1] = updated state, [2] = scattered output.
    ov::pass::pattern::Matcher loop_output_matcher(output_result);
    if (!loop_output_matcher.match(body_results[2]->output(0))) {
        return result;
    }
    ov::pass::pattern::Matcher loop_state_matcher(state_result);
    if (!loop_state_matcher.match(body_results[1]->output(0))) {
        return result;
    }

    const auto& output_map = loop_output_matcher.get_pattern_value_map();
    auto a_it = output_map.find(A);
    if (a_it == output_map.end()) {
        return result;
    }
    result.A = a_it->second;
    result.matched = true;
    return result;
}

}  // namespace

ov::pass::RemoveConcatSliceAfterLoopMamba2::RemoveConcatSliceAfterLoopMamba2() {
    auto x = pattern::any_input(pattern::shape_matches("[?, ?, head_num, head_dim]"));
    auto init_state = pattern::any_input(pattern::rank_equals(4));

    // External inputs: trip_count, exec_cond, dt, B, x, C, recurrent_state, output_buffer.
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
    auto dt = pattern::any_input(pattern::shape_matches("[?, ?, head_num]"));
    auto x = pattern::any_input(pattern::shape_matches("[?, ?, head_num, head_dim]"));
    auto init_state = pattern::any_input(pattern::shape_matches("[?, head_num, head_dim, state_size]"));

    // The loop consumes B/C already expanded from groups to heads via Unsqueeze -> Tile -> Reshape.
    // Capture the per-group operands so they can be fed to the op, which broadcasts them internally.
    auto B = pattern::any_input(pattern::shape_matches("[?, ?, group_num, state_size]"));
    auto B_expanded = pattern::wrap_type<v1::Reshape>(
        {pattern::wrap_type<v0::Tile>({pattern::wrap_type<v0::Unsqueeze>({B, 3}), pattern::any_input()}),
         pattern::any_input()});
    auto C = pattern::any_input(pattern::shape_matches("[?, ?, group_num, state_size]"));
    auto C_expanded = pattern::wrap_type<v1::Reshape>(
        {pattern::wrap_type<v0::Tile>({pattern::wrap_type<v0::Unsqueeze>({C, 3}), pattern::any_input()}),
         pattern::any_input()});

    auto loop_output =
        pattern::wrap_type<ov::op::v5::Loop>(ov::OutputVector{pattern::any_input(),  // trip count
                                                              pattern::any_input(),  // execution condition
                                                              dt,
                                                              B_expanded,
                                                              x,
                                                              C_expanded,
                                                              init_state,
                                                              pattern::any_input()});  // output accumulator buffer

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto loop_node = pattern_map.at(loop_output).get_node_shared_ptr();

        auto body_match = match_mamba2_body(loop_node);
        if (!body_match.matched) {
            return false;
        }

        // `A` is a foldable constant embedded in the loop body
        auto A_const = ov::util::get_constant_from_source(body_match.A);
        if (!A_const) {
            return false;
        }

        ov::OutputVector inputs = {
            A_const,                     // A
            pattern_map.at(dt),          // dt
            pattern_map.at(B),           // B (per group)
            pattern_map.at(x),           // x
            pattern_map.at(C),           // C (per group)
            pattern_map.at(init_state),  // recurrent_state
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
