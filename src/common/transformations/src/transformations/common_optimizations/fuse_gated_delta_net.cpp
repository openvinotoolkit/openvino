// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fuse_gated_delta_net.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/gated_delta_net.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/common_optimizations/transpose_sinking.hpp"
#include "transformations/symbolic_transformations/symbolic_optimizations.hpp"
#include "transformations/utils/utils.hpp"

namespace pattern = ov::pass::pattern;
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

    auto output_attn_buffer = pattern::any_input(pattern::shape_matches("[?, head_num, ?, v_head_size]"));
    auto recurrent_state = pattern::any_input(pattern::shape_matches("[?, head_num, k_head_size, v_head_size]"));
    auto beta = pattern::any_input(pattern::shape_matches("[?, head_num, 1]"));
    auto gate = pattern::any_input(pattern::shape_matches("[?, head_num, 1]"));
    auto value = pattern::any_input(pattern::shape_matches("[?, head_num, 1, value_head_size]"));
    auto key = pattern::any_input(pattern::shape_matches("[?, head_num, 1, k_head_size]"));
    auto query = pattern::any_input(pattern::shape_matches("[?, head_num, 1, k_head_size]"));
    auto step_index = pattern::any_input();

    auto step_index_unsqueeze = pattern::wrap_type<v0::Unsqueeze>({step_index, 0});
    auto gate_f32 = pattern::optional<v0::Convert>({gate});

    auto exp_gate = pattern::wrap_type<v0::Exp>({gate_f32});
    auto exp_gate_unsqueeze = pattern::wrap_type<v0::Unsqueeze>({exp_gate, {-1}});
    auto gated_state = pattern::wrap_type<v1::Multiply>({recurrent_state, exp_gate_unsqueeze});

    auto key_squeezed = pattern::wrap_type<v0::Squeeze>({key, {2}});
    auto key_unsqueeze = pattern::wrap_type<v0::Unsqueeze>({key_squeezed, {-1}});

    auto value_squeezed = pattern::wrap_type<v0::Squeeze>({value, {2}});

    auto projected_value = pattern::wrap_type<v1::Multiply>({gated_state, key_unsqueeze});
    auto projected_sum = pattern::wrap_type<v1::ReduceSum>({projected_value, {-2}}, {{"keep_dims", false}});
    auto delta = pattern::wrap_type<v1::Subtract>({value_squeezed, projected_sum});

    auto scaled_delta = pattern::wrap_type<v1::Multiply>({delta, beta});
    auto scaled_delta_unsqueeze = pattern::wrap_type<v0::Unsqueeze>({scaled_delta, {-2}});
    auto outer_update = pattern::wrap_type<v1::Multiply>({key_unsqueeze, scaled_delta_unsqueeze});
    auto updated_state = pattern::wrap_type<v1::Add>({gated_state, outer_update});

    auto query_squeezed = pattern::wrap_type<v0::Squeeze>({query, 2});
    auto query_unsqueeze = pattern::wrap_type<v0::Unsqueeze>({query_squeezed, {-1}});
    auto weighted_output = pattern::wrap_type<v1::Multiply>({updated_state, query_unsqueeze});

    auto output_reduce_sum = pattern::wrap_type<v1::ReduceSum>({weighted_output, {-2}}, {{"keep_dims", true}});
    auto output_reduce_sum_fp16 = pattern::optional<v0::Convert>({output_reduce_sum});
    auto scatter_update_output = pattern::wrap_type<ov::op::v3::ScatterUpdate>(
        {output_attn_buffer, step_index_unsqueeze, output_reduce_sum_fp16, 2});
    auto output_result = pattern::wrap_type<v0::Result>({scatter_update_output});

    auto updated_state_fp16 = pattern::optional<v0::Convert>({updated_state});
    auto state_result = pattern::wrap_type<v0::Result>({updated_state_fp16});

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
    auto value = pattern::any_input(pattern::shape_matches("[?, head_num, ?, v_head_size]"));
    auto init_state = pattern::any_input(pattern::rank_equals((4)));

    auto loop_inputs = ov::OutputVector{pattern::any_input(),
                                        pattern::any_input(),
                                        pattern::any_input(),
                                        pattern::any_input(),
                                        value,
                                        pattern::any_input(),
                                        pattern::any_input(),
                                        init_state,
                                        pattern::any_input()};

    auto loop_output0 = pattern::wrap_type<ov::op::v5::Loop>(loop_inputs, pattern::output_index_matches(0));
    auto loop_output1 = pattern::wrap_type<ov::op::v5::Loop>(loop_inputs, pattern::output_index_matches(1));

    auto reshape_core_attn = pattern::wrap_type<v1::Reshape>({loop_output0, {-1}});
    auto reshape_core_state = pattern::wrap_type<v1::Reshape>({loop_output1, {-1}});
    auto concat_loop = pattern::wrap_type<v0::Concat>({reshape_core_attn, reshape_core_state}, {{"axis", 0}});
    auto out_numel = pattern::any_input(pattern::has_static_shape());
    auto slice_attn = pattern::wrap_type<ov::op::v8::Slice>({concat_loop, {0}, out_numel, {1}, {0}});
    auto reshape_attn = pattern::wrap_type<v1::Reshape>({slice_attn, pattern::any_input()},
                                                        pattern::shape_matches("[?, head_num, ?, v_head_size]"));
    auto slice_state = pattern::wrap_type<ov::op::v8::Slice>({concat_loop, out_numel, pattern::any_input(), {1}, {0}});
    auto reshape_state =
        pattern::wrap_type<v1::Reshape>({slice_state, pattern::any_input()},
                                        pattern::shape_matches("[?, head_num, k_head_size, v_head_size]"));
    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
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
    auto query = pattern::any_input(pattern::shape_matches("[?, head_num, ?, qk_head_size]"));
    auto key = pattern::any_input(pattern::shape_matches("[?, head_num, ?, qk_head_size]"));
    auto value = pattern::any_input(pattern::shape_matches("[?, head_num, ?, v_head_size]"));
    auto init_state = pattern::any_input(pattern::rank_equals(4));
    auto gate = pattern::any_input(pattern::shape_matches("[?, head_num, ?]"));
    auto beta = pattern::any_input(pattern::shape_matches("[?, head_num, ?]"));
    // Check if the q_scale is sqrt(qk_head_size) or not. GDN spec assumes it's sqrt(qk_head_size).
    // TODO: CVS-183262 simplify the scale subgraph with constant.
    auto shape_head_size = pattern::any_input(pattern::shape_matches("[?, ?, ?, qk_head_size]"));
    auto shape_of_head_size = pattern::wrap_type<op::v3::ShapeOf>({shape_head_size});
    auto gather_index = pattern::optional<op::v8::Gather>({shape_of_head_size, {0, 2, 1, 3}, 0}, {{"batch_dims", 0}});
    auto gather = pattern::wrap_type<op::v8::Gather>({gather_index, 3, 0}, {{"batch_dims", 0}});

    auto head_size_f32 = pattern::optional<v0::Convert>({gather});

    auto const_half = pattern::wrap_type<v0::Constant>(pattern::value_matches("0.5"));
    auto convert_half = pattern::optional<v0::Convert>({const_half});

    auto attn_scale = pattern::wrap_type<v1::Power>({head_size_f32, convert_half});

    auto q_scale = pattern::wrap_type<v1::Divide>({query, attn_scale});
    // optional convert after q_scale for fp16
    auto q_convert = pattern::optional<v0::Convert>({q_scale});

    auto loop_output = ov::pass::pattern::wrap_type<ov::op::v5::Loop>(ov::OutputVector{pattern::any_input(),
                                                                                       pattern::any_input(),
                                                                                       q_convert,
                                                                                       key,
                                                                                       value,
                                                                                       gate,
                                                                                       beta,
                                                                                       init_state,
                                                                                       pattern::any_input()},
                                                                      [](std::shared_ptr<ov::Node> node) -> bool {
                                                                          return matches_linear_attention_loop(node);
                                                                      });

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
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

        auto linear_attn = std::make_shared<ov::op::internal::GatedDeltaNet>(inputs);

        linear_attn->set_friendly_name(loop_node->get_friendly_name());

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
    auto query = pattern::any_input(pattern::has_static_rank());
    auto key = pattern::any_input(pattern::has_static_rank());
    auto value = pattern::any_input(pattern::has_static_rank());
    auto init_state = pattern::any_input(pattern::has_static_rank());
    auto gate = pattern::any_input(pattern::has_static_rank());
    auto beta = pattern::any_input(pattern::has_static_rank());
    auto eps_q_const = pattern::wrap_type<v0::Constant>();
    auto eps_q = pattern::optional<v0::Convert>({eps_q_const});
    auto eps_k_const = pattern::wrap_type<v0::Constant>();
    auto eps_k = pattern::optional<v0::Convert>({eps_k_const});

    auto l2_norm = [](const ov::Output<ov::Node>& data, const ov::Output<ov::Node>& eps) {
        auto input_convert = pattern::optional<v0::Convert>({data});
        auto mul = pattern::wrap_type<v1::Multiply>({input_convert, input_convert});
        auto axis_const = pattern::wrap_type<v0::Constant>(pattern::value_matches("-1") || pattern::value_matches("3"));
        auto axis = pattern::optional<v0::Convert>({axis_const});
        auto reduce_sum = pattern::wrap_type<v1::ReduceSum>({mul, axis}, {{"keep_dims", true}});
        auto add = pattern::wrap_type<v1::Add>({reduce_sum, eps});
        auto sqrt = pattern::wrap_type<v0::Sqrt>({add});
        auto const_one = pattern::wrap_type<v0::Constant>(pattern::value_matches("1"));
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
    auto transpose_query = pattern::optional<v1::Transpose>({input_query, pattern::any_input()});
    auto transpose_key = pattern::optional<v1::Transpose>({input_key, pattern::any_input()});

    auto gdn = pattern::wrap_type<ov::op::internal::GatedDeltaNet>(
        {transpose_query, transpose_key, value, init_state, gate, beta});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto gdn_node = ov::as_type_ptr<ov::op::internal::GatedDeltaNet>(pattern_map.at(gdn).get_node_shared_ptr());

        auto l2_norm_q_eps_node = pattern_map.at(eps_q_const).get_node_shared_ptr();
        auto l2_norm_k_eps_node = pattern_map.at(eps_k_const).get_node_shared_ptr();

        float q_l2_norm_eps = ov::as_type_ptr<v0::Constant>(l2_norm_q_eps_node)->cast_vector<float>()[0];
        float k_l2_norm_eps = ov::as_type_ptr<v0::Constant>(l2_norm_k_eps_node)->cast_vector<float>()[0];

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

        auto new_gdn = std::make_shared<ov::op::internal::GatedDeltaNet>(gdn_input_query,
                                                                         gdn_input_key,
                                                                         pattern_map.at(value),
                                                                         pattern_map.at(init_state),
                                                                         pattern_map.at(gate),
                                                                         pattern_map.at(beta),
                                                                         true,
                                                                         q_l2_norm_eps,
                                                                         k_l2_norm_eps);
        ov::copy_runtime_info(gdn_node, new_gdn);
        new_gdn->set_friendly_name(gdn_node->get_friendly_name());
        ov::replace_node(gdn_node, new_gdn);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(gdn, "FuseL2NormIntoGDN");
    register_matcher(m, callback);
}

namespace {

bool is_allowed_qk_path_node(const std::shared_ptr<ov::Node>& node) {
    return ov::is_type<v1::Reshape>(node) || ov::is_type<v1::Transpose>(node) ||
           ov::is_type<ov::op::v8::Gather>(node) || ov::is_type<ov::op::v3::Broadcast>(node) ||
           ov::is_type<v0::Unsqueeze>(node) || ov::is_type<v0::Squeeze>(node) ||
           ov::is_type<ov::op::v5::GatherND>(node) || ov::is_type<ov::op::v8::GatherND>(node);
}

bool is_qk_anchor_node(const std::shared_ptr<ov::Node>& node) {
    return ov::is_type<ov::op::v1::Split>(node) || ov::is_type<ov::op::v1::VariadicSplit>(node) ||
           ov::is_type<ov::op::v8::Slice>(node) || ov::is_type<ov::op::v0::Concat>(node);
}

struct QKPathInfo {
    ov::Output<ov::Node> anchor_output;
    std::vector<std::shared_ptr<ov::Node>> visited_path;
};

QKPathInfo analyze_qk_path(const ov::Output<ov::Node>& start) {
    QKPathInfo info;

    // Traverse backward along input(0) only, collecting nodes and finding anchor.
    ov::Output<ov::Node> current = start;
    while (true) {
        const auto node = current.get_node_shared_ptr();
        if (!node) {
            break;
        }
        if (is_qk_anchor_node(node)) {
            info.anchor_output = current;
            break;
        }
        if (!is_allowed_qk_path_node(node) || node->inputs().empty()) {
            break;
        }
        info.visited_path.push_back(node);
        current = node->input_value(0);
    }

    if (!info.anchor_output.get_node_shared_ptr()) {
        info.visited_path.clear();
    }
    return info;
}

bool has_valid_path(const QKPathInfo& info) {
    return info.anchor_output.get_node_shared_ptr() && !info.visited_path.empty();
}

bool have_same_anchor(const QKPathInfo& q_info, const QKPathInfo& k_info, const QKPathInfo& v_info) {
    return q_info.anchor_output.get_node_shared_ptr() && k_info.anchor_output.get_node_shared_ptr() &&
           v_info.anchor_output.get_node_shared_ptr() &&
           q_info.anchor_output.get_node_shared_ptr() == k_info.anchor_output.get_node_shared_ptr() &&
           k_info.anchor_output.get_node_shared_ptr() == v_info.anchor_output.get_node_shared_ptr();
}

ov::Output<ov::Node> align_to_reference_shape(const ov::Output<ov::Node>& src,
                                              const ov::Output<ov::Node>& reference,
                                              ov::pass::MatcherPass* pass) {
    const auto& src_ps = src.get_partial_shape();
    const auto& ref_ps = reference.get_partial_shape();
    if (src_ps.compatible(ref_ps)) {
        return src;
    }

    if (src_ps.is_static() && ref_ps.is_static()) {
        const auto src_shape = src_ps.to_shape();
        const auto ref_shape = ref_ps.to_shape();
        if (ov::shape_size(src_shape) != ov::shape_size(ref_shape)) {
            return {};
        }
    }

    // Accumulate product of all static source dimensions: n.
    uint64_t n = 1;
    bool has_static_dims = false;
    for (const auto& dim : src_ps) {
        if (!dim.is_static()) {
            continue;
        }
        has_static_dims = true;
        n *= static_cast<uint64_t>(dim.get_length());
    }

    const auto ref_rank = ref_ps.rank();
    if (has_static_dims && ref_rank.is_static() && ref_rank.get_length() >= 2) {
        const auto last_dim_idx = ref_rank.get_length() - 1;
        const auto replace_dim_idx = ref_rank.get_length() - 2;
        const auto& last_dim = ref_ps[last_dim_idx];

        if (last_dim.is_static()) {
            const auto last_dim_value = static_cast<uint64_t>(last_dim.get_length());
            if (last_dim_value == 0 || n % last_dim_value != 0) {
                return {};
            }

            const auto n_div_last_dim = static_cast<int64_t>(n / last_dim_value);
            auto ref_shape = std::make_shared<ov::op::v3::ShapeOf>(reference);
            auto indices = v0::Constant::create(ov::element::i64, {1}, {replace_dim_idx});
            auto updates = v0::Constant::create(ov::element::i64, {1}, {n_div_last_dim});
            auto axis = v0::Constant::create(ov::element::i64, {}, {0});
            auto updated_shape = std::make_shared<ov::op::v3::ScatterUpdate>(ref_shape, indices, updates, axis);
            auto reshaped = std::make_shared<v1::Reshape>(src, updated_shape, false);

            pass->register_new_node(ref_shape);
            pass->register_new_node(updated_shape);
            pass->register_new_node(reshaped);
            return reshaped;
        }
    }

    auto ref_shape = std::make_shared<ov::op::v3::ShapeOf>(reference);
    auto reshaped = std::make_shared<v1::Reshape>(src, ref_shape, false);
    pass->register_new_node(ref_shape);
    pass->register_new_node(reshaped);
    return reshaped;
}

}  // namespace

ov::pass::FuseGroupedQueryIntoGDN::FuseGroupedQueryIntoGDN() {
    auto gdn = pattern::wrap_type<ov::op::internal::GatedDeltaNet>();

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto gdn_node = ov::as_type_ptr<ov::op::internal::GatedDeltaNet>(pattern_map.at(gdn).get_node_shared_ptr());
        if (!gdn_node) {
            return false;
        }

        const auto q_path = analyze_qk_path(gdn_node->input_value(0));
        const auto k_path = analyze_qk_path(gdn_node->input_value(1));
        const auto v_path = analyze_qk_path(gdn_node->input_value(2));

        // Step 2. Compare anchors and visited nodes collected for all Q/K/V inputs.
        if (!have_same_anchor(q_path, k_path, v_path)) {
            return false;
        }

        if (!has_valid_path(q_path) || !has_valid_path(k_path) || !has_valid_path(v_path)) {
            return false;
        }

        // If already directly connected from anchor outputs, skip.
        if (gdn_node->input_value(0).get_node_shared_ptr() == q_path.anchor_output.get_node_shared_ptr() &&
            gdn_node->input_value(1).get_node_shared_ptr() == k_path.anchor_output.get_node_shared_ptr() &&
            gdn_node->input_value(2).get_node_shared_ptr() == v_path.anchor_output.get_node_shared_ptr()) {
            return false;
        }

        auto q_aligned = align_to_reference_shape(q_path.anchor_output, gdn_node->input_value(0), this);
        auto k_aligned = align_to_reference_shape(k_path.anchor_output, gdn_node->input_value(1), this);
        auto v_aligned = align_to_reference_shape(v_path.anchor_output, gdn_node->input_value(2), this);

        if (!q_aligned.get_node_shared_ptr() || !k_aligned.get_node_shared_ptr() || !v_aligned.get_node_shared_ptr()) {
            return false;
        }

        auto new_gdn = std::make_shared<ov::op::internal::GatedDeltaNet>(q_aligned,
                                                                         k_aligned,
                                                                         v_aligned,
                                                                         gdn_node->input_value(3),
                                                                         gdn_node->input_value(4),
                                                                         gdn_node->input_value(5),
                                                                         gdn_node->get_fuse_qk_l2norm(),
                                                                         gdn_node->get_q_l2_norm_eps(),
                                                                         gdn_node->get_k_l2_norm_eps());
        new_gdn->set_friendly_name(gdn_node->get_friendly_name());
        ov::copy_runtime_info(gdn_node, new_gdn);
        ov::replace_node(gdn_node, new_gdn);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(gdn, "FuseGroupedQueryIntoGDN");
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
    symbolic_ctx_manager->register_pass<ov::pass::FuseGroupedQueryIntoGDN>();
    return symbolic_optimizations.run_on_model(model);
}