// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fuse_gated_delta_net.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/gated_delta_net.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/common_optimizations/transpose_sinking.hpp"
#include "transformations/symbolic_transformations/symbolic_optimizations.hpp"
#include "transformations/utils/gen_pattern.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::pass;
using namespace ov::gen_pattern;
using namespace ov::pass::pattern;
namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;

namespace {

bool matches_linear_attention_loop(const std::shared_ptr<ov::Node>& node) {
    auto loop = ov::as_type_ptr<ov::op::v5::Loop>(node);
    if (!loop) {
        return false;
    }

    if (loop->get_input_size() < 9 || loop->get_output_size() != 2) {
        return false;
    }

    auto output_attn_buffer = any_input(shape_matches("[?, head_num, ?, v_head_size]"));
    auto recurrent_state = any_input(shape_matches("[?, head_num, k_head_size, v_head_size]"));
    auto beta = any_input(shape_matches("[?, head_num, 1]"));
    auto gate = any_input(shape_matches("[?, head_num, 1]"));
    auto value = any_input(shape_matches("[?, head_num, 1, value_head_size]"));
    auto key = any_input(shape_matches("[?, head_num, 1, k_head_size]"));
    auto query = any_input(shape_matches("[?, head_num, 1, k_head_size]"));
    auto step_index = any_input();

    auto step_index_unsqueeze = wrap_type<v0::Unsqueeze>({step_index, 0});
    auto gate_f32 = pattern::optional<v0::Convert>({gate});

    auto exp_gate = wrap_type<v0::Exp>({gate_f32});
    auto exp_gate_unsqueeze = wrap_type<v0::Unsqueeze>({exp_gate, {-1}});
    auto gated_state = wrap_type<v1::Multiply>({recurrent_state, exp_gate_unsqueeze});

    auto key_squeezed = wrap_type<v0::Squeeze>({key, {2}});
    auto key_unsqueeze = wrap_type<v0::Unsqueeze>({key_squeezed, {-1}});

    auto value_squeezed = wrap_type<v0::Squeeze>({value, {2}});

    auto projected_value = wrap_type<v1::Multiply>({gated_state, key_unsqueeze});
    auto projected_sum = wrap_type<v1::ReduceSum>({projected_value, {-2}}, {{"keep_dims", false}});
    auto delta = wrap_type<v1::Subtract>({value_squeezed, projected_sum});

    auto scaled_delta = wrap_type<v1::Multiply>({delta, beta});
    auto scaled_delta_unsqueeze = wrap_type<v0::Unsqueeze>({scaled_delta, {-2}});
    auto outer_update = wrap_type<v1::Multiply>({key_unsqueeze, scaled_delta_unsqueeze});
    auto updated_state = wrap_type<v1::Add>({gated_state, outer_update});

    auto query_squeezed = wrap_type<v0::Squeeze>({query, 2});
    auto query_unsqueeze = wrap_type<v0::Unsqueeze>({query_squeezed, {-1}});
    auto weighted_output = wrap_type<v1::Multiply>({updated_state, query_unsqueeze});

    auto output_reduce_sum = wrap_type<v1::ReduceSum>({weighted_output, {-2}}, {{"keep_dims", true}});
    auto output_reduce_sum_fp16 = pattern::optional<v0::Convert>({output_reduce_sum});
    auto scatter_update_output =
        wrap_type<ov::op::v3::ScatterUpdate>({output_attn_buffer, step_index_unsqueeze, output_reduce_sum_fp16, 2});
    auto output_result = wrap_type<v0::Result>({scatter_update_output});

    auto updated_state_fp16 = pattern::optional<v0::Convert>({updated_state});
    auto state_result = wrap_type<v0::Result>({updated_state_fp16});

    ov::pass::pattern::Matcher loop_output_matcher(output_result);
    ov::pass::pattern::Matcher loop_state_matcher(state_result);
    auto body = loop->get_function();
    const auto& body_results = body->get_results();

    // match output
    if (!loop_output_matcher.match(body_results[2]->output(0))) {
        return false;
    }
    // match state
    if (!loop_state_matcher.match(body_results[1]->output(0))) {
        return false;
    }
    return true;
}

}  // namespace

ov::pass::RemoveConcatSliceAfterLoop::RemoveConcatSliceAfterLoop() {
    auto value = any_input(shape_matches("[?, head_num, ?, v_head_size]"));
    auto init_state = any_input(rank_equals((4)));

    auto loop_inputs = OutputVector{any_input(),
                                    any_input(),
                                    any_input(),
                                    any_input(),
                                    value,
                                    any_input(),
                                    any_input(),
                                    init_state,
                                    any_input()};

    auto loop_output0 = wrap_type<ov::op::v5::Loop>(loop_inputs, pattern::output_index_matches(0));
    auto loop_output1 = wrap_type<ov::op::v5::Loop>(loop_inputs, pattern::output_index_matches(1));

    auto reshape_core_attn = pattern::wrap_type<v1::Reshape>({loop_output0, {-1}});
    auto reshape_core_state = pattern::wrap_type<v1::Reshape>({loop_output1, {-1}});
    auto concat_loop = makeOP<v0::Concat>({reshape_core_attn, reshape_core_state}, {{"axis", 0}});
    auto out_numel = any_input(has_static_shape());
    auto slice_attn = pattern::wrap_type<ov::op::v8::Slice>({concat_loop, {0}, out_numel, {1}, {0}});
    auto reshape_attn = pattern::wrap_type<v1::Reshape>({slice_attn, any_input()},
                                                        pattern::shape_matches("[?, head_num, ?, v_head_size]"));
    auto slice_state = pattern::wrap_type<ov::op::v8::Slice>({concat_loop, out_numel, any_input(), {1}, {0}});
    auto reshape_state =
        pattern::wrap_type<v1::Reshape>({slice_state, any_input()},
                                        pattern::shape_matches("[?, head_num, k_head_size, v_head_size]"));
    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        bool changed = false;
        auto loop_node = pattern_map.at(loop_output0).get_node_shared_ptr();
        if (pattern_map.count(reshape_attn)) {
            auto reshape_attn_out = pattern_map.at(reshape_attn);
            if (!ov::replace_output_update_name(pattern_map.at(reshape_attn), loop_node->output(0))) {
                reshape_attn_out.replace(loop_node->output(0));
            }
            changed = true;
        }

        if (pattern_map.count(reshape_state)) {
            auto reshape_state_out = pattern_map.at(reshape_state);
            if (!ov::replace_output_update_name(reshape_state_out, loop_node->output(1))) {
                reshape_state_out.replace(loop_node->output(1));
            }
            changed = true;
        }
        return changed;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(reshape_attn | reshape_state, "RemoveConcatSliceAfterLoop");
    register_matcher(m, callback);
}

ov::pass::FuseGDNLoop::FuseGDNLoop() {
    // fuse standalone loop into a single GatedDeltaNet op
    auto query = ov::pass::pattern::any_input(shape_matches("[?, head_num, ?, qk_head_size]"));
    auto key = ov::pass::pattern::any_input(shape_matches("[?, head_num, ?, qk_head_size]"));
    auto value = ov::pass::pattern::any_input(shape_matches("[?, head_num, ?, v_head_size]"));
    auto init_state = ov::pass::pattern::any_input(rank_equals(4));
    auto gate = ov::pass::pattern::any_input(shape_matches("[?, head_num, ?]"));
    auto beta = ov::pass::pattern::any_input(shape_matches("[?, head_num, ?]"));

    auto shape_head_size = any_input(shape_matches("[?, ?, ?, qk_head_size]"));
    auto shape_of_head_size = pattern::wrap_type<op::v3::ShapeOf>({shape_head_size});
    auto gather_index = pattern::optional<op::v8::Gather>({shape_of_head_size, {0, 2, 1, 3}, 0}, {{"batch_dims", 0}});
    auto gather = pattern::wrap_type<op::v8::Gather>({gather_index, 3, 0}, {{"batch_dims", 0}});

    auto head_size_f32 = pattern::optional<v0::Convert>({gather});

    auto const_half = pattern::wrap_type<v0::Constant>(value_matches("0.5"));
    auto convert_half = pattern::optional<v0::Convert>({const_half});

    auto power_head_size = pattern::wrap_type<v1::Power>({head_size_f32, convert_half});
    auto attn_scale = power_head_size;

    auto q_scale = pattern::wrap_type<v1::Divide>({query, attn_scale});
    // optional convert after q_scale for fp16
    auto q_convert = pattern::optional<v0::Convert>({q_scale});

    auto loop_output = ov::pass::pattern::wrap_type<ov::op::v5::Loop>(
        OutputVector{any_input(), any_input(), q_convert, key, value, gate, beta, init_state, any_input()},
        [](std::shared_ptr<ov::Node> node) -> bool {
            return matches_linear_attention_loop(node);
        });

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto loop_node = pattern_map.at(loop_output).get_node_shared_ptr();

        auto perm_bhls_to_blhs = v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3});
        auto perm_bhl_to_blh = v0::Constant::create(ov::element::i64, {3}, {0, 2, 1});
        ov::Output<ov::Node> q_in = pattern_map.at(query);
        // fuse q_scale into GDN if convert exists
        if (pattern_map.count(q_convert)) {
            auto q_convert_node = pattern_map.at(q_convert).get_node_shared_ptr();
            auto new_convert_node = q_convert_node->clone_with_new_inputs({q_in});
            ov::copy_runtime_info(q_convert_node, new_convert_node);
            ov::replace_node(q_convert_node, new_convert_node);
            q_in = new_convert_node;
        }
        // layout transpose to [batch, seq_len, head_num, head_size] to align with GDN spec
        auto q_blhs = std::make_shared<v1::Transpose>(q_in, perm_bhls_to_blhs);
        auto k_blhs = std::make_shared<v1::Transpose>(pattern_map.at(key), perm_bhls_to_blhs);
        auto v_blhs = std::make_shared<v1::Transpose>(pattern_map.at(value), perm_bhls_to_blhs);
        auto g_blh = std::make_shared<v1::Transpose>(pattern_map.at(gate), perm_bhl_to_blh);
        auto beta_blh = std::make_shared<v1::Transpose>(pattern_map.at(beta), perm_bhl_to_blh);

        ov::copy_runtime_info(m.get_matched_nodes(),
                              {perm_bhls_to_blhs, perm_bhl_to_blh, q_blhs, k_blhs, v_blhs, g_blh, beta_blh});

        ov::OutputVector inputs = {
            q_blhs,                      // query
            k_blhs,                      // key
            v_blhs,                      // value
            pattern_map.at(init_state),  // initial_state
            g_blh,                       // g
            beta_blh                     // beta
        };

        auto linear_attn = std::make_shared<ov::op::GatedDeltaNet>(inputs);

        linear_attn->set_friendly_name(loop_node->get_friendly_name());

        ov::op::GatedDeltaNet::Config config;
        config.fuse_qk_l2norm = false;
        linear_attn->set_config(config);
        ov::copy_runtime_info(loop_node, linear_attn);
        ov::replace_node(loop_node, linear_attn);
        auto consumers = linear_attn->output(0).get_target_inputs();

        auto out_transposed = std::make_shared<v1::Transpose>(linear_attn->output(0), perm_bhls_to_blhs);

        for (auto input : consumers) {
            input.replace_source_output(out_transposed);
        }
        register_new_node(out_transposed);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(loop_output, "FuseGDNLoop");
    register_matcher(m, callback);
}

ov::pass::FuseL2NormIntoGDN::FuseL2NormIntoGDN() {
    auto query = ov::pass::pattern::any_input(has_static_rank());
    auto key = ov::pass::pattern::any_input(has_static_rank());
    auto value = ov::pass::pattern::any_input(has_static_rank());
    auto init_state = ov::pass::pattern::any_input(has_static_rank());
    auto gate = ov::pass::pattern::any_input(has_static_rank());
    auto beta = ov::pass::pattern::any_input(has_static_rank());
    auto eps_q_const = pattern::wrap_type<v0::Constant>();
    auto eps_q = pattern::optional<v0::Convert>({eps_q_const});
    auto eps_k_const = pattern::wrap_type<v0::Constant>();
    auto eps_k = pattern::optional<v0::Convert>({eps_k_const});

    auto l2_norm = [](const ov::Output<ov::Node>& data, const ov::Output<ov::Node>& eps) {
        auto input_convert = pattern::optional<v0::Convert>({data});
        auto mul = pattern::wrap_type<v1::Multiply>({input_convert, input_convert});
        auto axis_const = pattern::wrap_type<v0::Constant>(value_matches("-1") || value_matches("3"));
        auto axis = pattern::optional<v0::Convert>({axis_const});
        auto reduce_sum = pattern::wrap_type<v1::ReduceSum>({mul, axis}, {{"keep_dims", true}});
        auto add = pattern::wrap_type<v1::Add>({reduce_sum, eps});
        auto sqrt = pattern::wrap_type<v0::Sqrt>({add});
        auto const_one = pattern::wrap_type<v0::Constant>(value_matches("1"));
        auto convert_one = pattern::optional<v0::Convert>({const_one});
        auto div = pattern::wrap_type<v1::Divide>({convert_one, sqrt});
        auto multiply = pattern::wrap_type<v1::Multiply>({input_convert, div});
        return multiply;
    };

    auto normalized_query = l2_norm(query, eps_q);
    auto normalized_key = l2_norm(key, eps_k);
    // optional transpose maybe inserted by FuseGDNLoop
    auto input_query = pattern::optional<v0::Convert>({normalized_query});
    auto input_key = pattern::optional<v0::Convert>({normalized_key});
    auto transpose_query = pattern::optional<v1::Transpose>({input_query, any_input()});
    auto transpose_key = pattern::optional<v1::Transpose>({input_key, any_input()});

    auto gdn =
        pattern::wrap_type<ov::op::GatedDeltaNet>({transpose_query, transpose_key, value, init_state, gate, beta});

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto gdn_node = ov::as_type_ptr<ov::op::GatedDeltaNet>(pattern_map.at(gdn).get_node_shared_ptr());

        auto config = gdn_node->get_config();
        config.fuse_qk_l2norm = true;

        auto l2_norm_q_eps_node = pattern_map.at(eps_q_const).get_node_shared_ptr();
        auto l2_norm_k_eps_node = pattern_map.at(eps_k_const).get_node_shared_ptr();

        config.q_l2_norm_eps = ov::as_type_ptr<v0::Constant>(l2_norm_q_eps_node)->cast_vector<float>()[0];
        config.k_l2_norm_eps = ov::as_type_ptr<v0::Constant>(l2_norm_k_eps_node)->cast_vector<float>()[0];

        ov::Output<ov::Node> gdn_input_query = pattern_map.at(query);
        ov::Output<ov::Node> gdn_input_key = pattern_map.at(key);
        // q/k->l2_norm->transpose->GDN => q/k->tranpose->GDN,
        bool q_transposed = pattern_map.count(transpose_query);
        if (q_transposed) {
            auto transpose_q = pattern_map.at(transpose_query).get_node_shared_ptr();
            auto out_transposed = transpose_q->clone_with_new_inputs({gdn_input_query, transpose_q->input_value(1)});
            out_transposed->set_friendly_name(transpose_q->get_friendly_name());
            ov::copy_runtime_info(transpose_q, out_transposed);
            ov::replace_node(transpose_q, out_transposed);
            gdn_input_query = gdn_node->input_value(0);
        }
        bool k_transposed = pattern_map.count(transpose_key);
        if (k_transposed) {
            auto transpose_k = pattern_map.at(transpose_key).get_node_shared_ptr();
            auto out_transposed = transpose_k->clone_with_new_inputs({gdn_input_key, transpose_k->input_value(1)});
            out_transposed->set_friendly_name(transpose_k->get_friendly_name());
            ov::copy_runtime_info(transpose_k, out_transposed);
            ov::replace_node(transpose_k, out_transposed);
            gdn_input_key = gdn_node->input_value(1);
        }
        gdn_node->set_config(config);
        auto new_gdn = gdn_node->clone_with_new_inputs({gdn_input_query,
                                                        gdn_input_key,
                                                        pattern_map.at(value),
                                                        pattern_map.at(init_state),
                                                        pattern_map.at(gate),
                                                        pattern_map.at(beta)});
        ov::copy_runtime_info(gdn_node, new_gdn);
        new_gdn->set_friendly_name(gdn_node->get_friendly_name());
        ov::replace_node(gdn_node, new_gdn);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(gdn, "FuseL2NormIntoGDN");
    register_matcher(m, callback);
}

bool ov::pass::GatedDeltaNetFusion::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(GatedDeltaNetFusion);
    ov::pass::SymbolicOptimizations symbolic_optimizations(false, get_pass_config());
    auto symbolic_ctx_manager = symbolic_optimizations.get_manager();
    symbolic_ctx_manager->register_pass<ov::pass::RemoveConcatSliceAfterLoop>();
    symbolic_ctx_manager->register_pass<ov::pass::FuseGDNLoop>();
    // remove redundant transpose after loop fusion, which are inserted by FuseGDNLoop
    symbolic_ctx_manager->register_pass<ov::pass::TransposeFuse>();
    symbolic_ctx_manager->register_pass<ov::pass::FuseL2NormIntoGDN>();
    return symbolic_optimizations.run_on_model(model);
}