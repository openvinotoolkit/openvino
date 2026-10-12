// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fuse_ssm.hpp"

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

// Matches the time-major SSM recurrence body. All discretization is performed outside the loop
// (dA = exp(A * dt), dBx = (dt * B) outer x, C reshaped), so the body consumes the fully discretized
// per-step tensors directly. Per-step body semantics (H = num_heads, P = head_dim, N = state_size):
//   dA_t = Squeeze(dA_t, 1)   -> [B, H, 1, 1]
//   dBx_t = Squeeze(dBx_t, 1) -> [B, H, P, N]
//   C_t = Squeeze(C_t, 1)     -> [B, H, 1, N]
//   state_t = state_{t-1} * dA_t + dBx_t
//   y_t     = reduce_sum(state_t * C_t, axis=N)
//   ScatterUpdate(core_out, t, y_t, axis=1)
bool match_ssm_body(const std::shared_ptr<ov::Node>& node) {
    auto loop = ov::as_type_ptr<ov::op::v5::Loop>(node);
    if (!loop)
        return false;

    // External inputs: trip_count, exec_cond, dA, dBx, C, recurrent_state, output_buffer.
    // External outputs: output, output_recurrent_state.
    if (loop->get_input_size() != 7 || loop->get_output_size() != 2)
        return false;

    // body_results: [0] = exec_condition, [1] = updated state, [2] = scattered output.
    auto body = loop->get_function();
    const auto& body_results = body->get_results();
    if (body_results.size() < 3)
        return false;

    auto dA_t = pattern::any_input();
    auto dBx_t = pattern::any_input();
    auto last_state = pattern::any_input();

    auto dA_squeezed = pattern::wrap_type<v0::Squeeze>({dA_t, 1});    // [B, H, 1, 1]
    auto dBx_squeezed = pattern::wrap_type<v0::Squeeze>({dBx_t, 1});  // [B, H, P, N]

    // state_t = state_{t-1} * dA_t + dBx_t
    auto state_decay = pattern::wrap_type<v1::Multiply>({last_state, dA_squeezed});
    auto state_new = pattern::wrap_type<v1::Add>({state_decay, dBx_squeezed});
    auto state_result = pattern::wrap_type<v0::Result>(pattern::optional<v0::Convert>({state_new}));

    ov::pass::pattern::Matcher loop_state_matcher(state_result);
    if (!loop_state_matcher.match(body_results[1]->output(0))) {
        return false;
    }

    auto core_out = pattern::any_input();
    auto step_index = pattern::any_input();
    auto C_t = pattern::any_input();
    auto C_squeezed = pattern::wrap_type<v0::Squeeze>({C_t, 1});  // [B, H, 1, N]

    // y_t = reduce_sum(state_t * C_t, axis=N)
    auto weighted_output = pattern::wrap_type<v1::Multiply>({state_new, C_squeezed});
    auto output_reduce_sum = pattern::wrap_type<v1::ReduceSum>({weighted_output, -1}, {{"keep_dims", false}});
    auto output_unsqueeze = pattern::wrap_type<v0::Unsqueeze>({output_reduce_sum, 1});

    auto step_index_unsqueeze = pattern::wrap_type<v0::Unsqueeze>({step_index, 0});
    auto scatter_update_output = pattern::wrap_type<ov::op::v3::ScatterUpdate>(
        {core_out, step_index_unsqueeze, pattern::optional<v0::Convert>({output_unsqueeze}), 1});
    auto output_result = pattern::wrap_type<v0::Result>({scatter_update_output});

    ov::pass::pattern::Matcher loop_output_matcher(output_result);
    if (!loop_output_matcher.match(body_results[2]->output(0))) {
        return false;
    }

    return true;
}

}  // namespace

ov::pass::RemoveConcatSliceAfterLoopSSM::RemoveConcatSliceAfterLoopSSM() {
    auto init_state = pattern::any_input(pattern::rank_equals(4));

    // External inputs: trip_count, exec_cond, dA, dBx, C, recurrent_state, output_buffer.
    auto loop_inputs = ov::OutputVector{pattern::any_input(),
                                        pattern::any_input(),
                                        pattern::any_input(),
                                        pattern::any_input(),
                                        pattern::any_input(),
                                        init_state,
                                        pattern::any_input()};

    auto loop = pattern::wrap_type_strict_index<ov::op::v5::Loop>(loop_inputs);

    auto reshape_output = pattern::wrap_type<v1::Reshape>({loop->output(0), {-1}});
    auto reshape_state = pattern::wrap_type<v1::Reshape>({loop->output(1), {-1}});
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
        auto loop_node = m.get_pattern_map().at(loop);
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

    auto m = std::make_shared<ov::pass::pattern::Matcher>(restored_root, "RemoveConcatSliceAfterLoopSSM");
    register_matcher(m, callback);
}

ov::pass::FuseSSMLoop::FuseSSMLoop(size_t& fused_count) {
    auto dt = pattern::any_input(pattern::shape_matches("[?, ?, head_num]"));
    auto x = pattern::any_input(pattern::shape_matches("[?, ?, head_num, head_dim]"));
    auto init_state = pattern::any_input(pattern::shape_matches("[?, head_num, head_dim, state_size]"));

    // All discretization is performed outside the loop and consumed as 5D per-step slices:
    //   dA  = Reshape(exp(A * dt), [B, T, H, 1, 1])                       -> Loop input 2
    //   dBx = Unsqueeze(dt*B, -2) * Unsqueeze(x, -1)  -> [B, T, H, P, N]  -> Loop input 3
    //   C   = Unsqueeze(B/C-expanded, -2)             -> [B, T, H, 1, N]  -> Loop input 4
    auto A = pattern::any_input(pattern::shape_matches("[head_num]"));
    auto dA = pattern::wrap_type<v1::Reshape>(
        {pattern::wrap_type<v0::Exp>({pattern::wrap_type<v1::Multiply>({A, dt})}), pattern::any_input()});

    auto B = pattern::any_input(pattern::shape_matches("[?, ?, group_num, state_size]"));
    auto B_expanded = pattern::wrap_type<v1::Reshape>(
        {pattern::wrap_type<v0::Tile>(
             {pattern::wrap_type<v0::Unsqueeze>({B, pattern::any_input()}), pattern::any_input()}),
         pattern::any_input()});

    // dB = Unsqueeze(dt, -1) * B_expanded ; dBx = Unsqueeze(dB, -2) * Unsqueeze(x, -1).
    auto dB =
        pattern::wrap_type<v1::Multiply>({pattern::wrap_type<v0::Unsqueeze>({dt, pattern::any_input()}), B_expanded});
    auto dBx = pattern::wrap_type<v1::Multiply>({pattern::wrap_type<v0::Unsqueeze>({dB, pattern::any_input()}),
                                                 pattern::wrap_type<v0::Unsqueeze>({x, pattern::any_input()})});

    auto C = pattern::any_input(pattern::shape_matches("[?, ?, group_num, state_size]"));
    auto C_expanded = pattern::wrap_type<v1::Reshape>(
        {pattern::wrap_type<v0::Tile>(
             {pattern::wrap_type<v0::Unsqueeze>({C, pattern::any_input()}), pattern::any_input()}),
         pattern::any_input()});
    auto C_5d = pattern::wrap_type<v0::Unsqueeze>({C_expanded, pattern::any_input()});

    auto loop_output =
        pattern::wrap_type<ov::op::v5::Loop>(ov::OutputVector{pattern::any_input(),  // trip count
                                                              pattern::any_input(),  // execution condition
                                                              dA,
                                                              dBx,
                                                              C_5d,
                                                              init_state,
                                                              pattern::any_input()});  // output accumulator buffer

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS, &fused_count](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto loop_node = pattern_map.at(loop_output).get_node_shared_ptr();

        if (!match_ssm_body(loop_node)) {
            return false;
        }

        ov::OutputVector inputs = {
            pattern_map.at(A),           // A
            pattern_map.at(dt),          // dt
            pattern_map.at(B),           // B (per group)
            pattern_map.at(x),           // x
            pattern_map.at(C),           // C (per group)
            pattern_map.at(init_state),  // recurrent_state
        };

        auto selective_ssm = std::make_shared<ov::op::internal::SelectiveSSM>(inputs);
        selective_ssm->set_friendly_name(loop_node->get_friendly_name());

        ov::copy_runtime_info(m.get_matched_nodes(), selective_ssm);
        ov::replace_node(loop_node, selective_ssm);
        ++fused_count;
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(loop_output, "FuseSSMLoop");
    register_matcher(m, callback);
}

bool ov::pass::SelectiveSSMFusion::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(SelectiveSSMFusion);
    ov::pass::SymbolicOptimizations symbolic_optimizations(false, get_pass_config());
    auto symbolic_ctx_manager = symbolic_optimizations.get_manager();
    symbolic_ctx_manager->register_pass<ov::pass::RemoveConcatSliceAfterLoopSSM>();
    symbolic_ctx_manager->register_pass<ov::pass::FuseSSMLoop>(m_fused_count);
    return symbolic_optimizations.run_on_model(model);
}
