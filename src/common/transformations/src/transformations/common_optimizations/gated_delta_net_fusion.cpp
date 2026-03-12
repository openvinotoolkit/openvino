// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/gated_delta_net_fusion.hpp"

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"
#include "ov_ops/gated_delta_net.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::pass {

namespace {
namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v3 = ov::op::v3;
namespace v5 = ov::op::v5;
namespace v8 = ov::op::v8;

using MergedD = op::util::MultiSubGraphOp::MergedInputDescription;
using SlicedD = op::util::MultiSubGraphOp::SliceInputDescription;

bool is_constant_vector(const std::shared_ptr<Node>& node, const std::vector<int64_t>& expected) {
    const auto constant = ov::as_type_ptr<v0::Constant>(node);
    if (!constant || shape_size(constant->get_shape()) != expected.size()) {
        return false;
    }

    return constant->cast_vector<int64_t>() == expected;
}

bool is_true_constant(const std::shared_ptr<Node>& node) {
    const auto constant = ov::as_type_ptr<v0::Constant>(node);
    if (!constant || shape_size(constant->get_shape()) != 1) {
        return false;
    }

    return constant->cast_vector<char>()[0] != 0;
}

bool is_reduce_sum(const std::shared_ptr<Node>& node, int64_t axis, bool keep_dims) {
    const auto rs = ov::as_type_ptr<v1::ReduceSum>(node);
    if (!rs || rs->get_keep_dims() != keep_dims) {
        return false;
    }

    return is_constant_vector(rs->get_input_node_shared_ptr(1), {axis});
}

bool is_flatten_reshape(const std::shared_ptr<Node>& node) {
    const auto reshape = ov::as_type_ptr<v1::Reshape>(node);
    if (!reshape || reshape->get_special_zero()) {
        return false;
    }

    return is_constant_vector(reshape->get_input_node_shared_ptr(1), {-1});
}

bool is_exporter_zero_broadcast(const Output<Node>& value, const Output<Node>& init_attn) {
    const auto broadcast = ov::as_type_ptr<v3::Broadcast>(init_attn.get_node_shared_ptr());
    if (!broadcast) {
        return false;
    }

    const auto zero = ov::as_type_ptr<v0::Constant>(broadcast->get_input_node_shared_ptr(0));
    const auto value_shape = ov::as_type_ptr<v3::ShapeOf>(broadcast->get_input_node_shared_ptr(1));
    if (!zero || !value_shape) {
        return false;
    }

    if (shape_size(zero->get_shape()) != 1 || zero->cast_vector<float>()[0] != 0.0f) {
        return false;
    }

    return value_shape->input_value(0) == value;
}

struct LoopMatch {
    Output<Node> query;
    Output<Node> key;
    Output<Node> value;
    Output<Node> gate;
    Output<Node> beta;
    Output<Node> recurrent_state;
};

struct LoopBodyMatch {
    std::shared_ptr<v0::Parameter> timestep_param;
    std::shared_ptr<v0::Parameter> query_param;
    std::shared_ptr<v0::Parameter> key_param;
    std::shared_ptr<v0::Parameter> value_param;
    std::shared_ptr<v0::Parameter> gate_param;
    std::shared_ptr<v0::Parameter> beta_param;
    std::shared_ptr<v0::Parameter> recurrent_state_param;
    std::shared_ptr<v0::Parameter> core_attn_state_param;
    std::shared_ptr<v1::Add> state_out;
    std::shared_ptr<v3::ScatterUpdate> scatter;
};

bool is_expected_sliced_input(const std::shared_ptr<SlicedD>& desc) {
    return desc && desc->m_axis == 2 && desc->m_start == 0 && desc->m_stride == 1 && desc->m_part_size == 1 &&
           desc->m_end == -1;
}

bool match_loop_signature(const std::shared_ptr<v5::Loop>& loop) {
    auto loop_pattern = pattern::wrap_type<v5::Loop>([](const Output<Node>& out) {
        const auto loop_node = ov::as_type_ptr<v5::Loop>(out.get_node_shared_ptr());
        if (!loop_node) {
            return false;
        }

        const auto& body = loop_node->get_function();
        if (!body) {
            return false;
        }

        const auto& body_parameters = body->get_parameters();
        const auto& body_results = body->get_results();
        const auto& special_ports = loop_node->get_special_body_ports();
        return body_parameters.size() == 8 && body_results.size() == 3 && special_ports.current_iteration_input_idx == 0 &&
               special_ports.body_condition_output_idx == 0;
    });

    pattern::Matcher loop_matcher(loop_pattern, "GatedDeltaNetFusionLoopMatcher");
    return loop_matcher.match(std::static_pointer_cast<Node>(loop));
}

std::shared_ptr<Node> make_const_pattern(const std::vector<int64_t>& values) {
    return pattern::wrap_type<v0::Constant>([values](const Output<Node>& out) {
        return is_constant_vector(out.get_node_shared_ptr(), values);
    });
}

bool match_loop_body_pattern(const std::shared_ptr<v5::Loop>& loop, LoopBodyMatch& body_match) {
    const auto& body = loop->get_function();
    const auto& body_results = body->get_results();
    if (body_results.size() != 3 || !is_true_constant(body_results[0]->get_input_node_shared_ptr(0))) {
        return false;
    }

    auto c_m1 = make_const_pattern({-1});
    auto c_m2 = make_const_pattern({-2});
    auto c_0 = make_const_pattern({0});
    auto c_2 = make_const_pattern({2});

    auto timestep_param = pattern::wrap_type<v0::Parameter>();
    auto query_param = pattern::wrap_type<v0::Parameter>();
    auto key_param = pattern::wrap_type<v0::Parameter>();
    auto value_param = pattern::wrap_type<v0::Parameter>();
    auto gate_param = pattern::wrap_type<v0::Parameter>();
    auto beta_param = pattern::wrap_type<v0::Parameter>();
    auto recurrent_state_param = pattern::wrap_type<v0::Parameter>();
    auto core_attn_state_param = pattern::wrap_type<v0::Parameter>();

    auto gate_exp = pattern::wrap_type<v0::Exp>({gate_param});
    auto gate_unsqueeze = pattern::wrap_type<v0::Unsqueeze>({gate_exp, c_m1});
    auto state_decay = pattern::wrap_type<v1::Multiply>({recurrent_state_param, gate_unsqueeze});

    auto key_squeeze = pattern::wrap_type<v0::Squeeze>({key_param, c_2});
    auto key_unsqueeze_inner = pattern::wrap_type<v0::Unsqueeze>({key_squeeze, c_m1});
    auto kv_mul = pattern::wrap_type<v1::Multiply>({state_decay, key_unsqueeze_inner});
    auto kv_reduce = pattern::wrap_type<v1::ReduceSum>({kv_mul, c_m2});

    auto value_squeeze = pattern::wrap_type<v0::Squeeze>({value_param, c_2});
    auto value_minus_kv = pattern::wrap_type<v1::Subtract>({value_squeeze, kv_reduce});
    auto delta = pattern::wrap_type<v1::Multiply>({value_minus_kv, beta_param});
    auto delta_unsqueeze = pattern::wrap_type<v0::Unsqueeze>({delta, c_m2});
    auto key_unsqueeze_outer = pattern::wrap_type<v0::Unsqueeze>({key_squeeze, c_m1});
    auto state_delta = pattern::wrap_type<v1::Multiply>({key_unsqueeze_outer, delta_unsqueeze});
    auto state_out = pattern::wrap_type<v1::Add>({state_decay, state_delta});

    auto query_squeeze = pattern::wrap_type<v0::Squeeze>({query_param, c_2});
    auto query_unsqueeze = pattern::wrap_type<v0::Unsqueeze>({query_squeeze, c_m1});
    auto core_attn_mul = pattern::wrap_type<v1::Multiply>({state_out, query_unsqueeze});
    auto core_attn_update = pattern::wrap_type<v1::ReduceSum>({core_attn_mul, c_m2});
    auto timestep_unsqueeze = pattern::wrap_type<v0::Unsqueeze>({timestep_param, c_0});
    auto scatter = pattern::wrap_type<v3::ScatterUpdate>({core_attn_state_param, timestep_unsqueeze, core_attn_update, c_2});

    pattern::Matcher body_matcher(scatter, "GatedDeltaNetFusionBodyMatcher");
    if (!body_matcher.match(body_results[2]->input_value(0))) {
        return false;
    }

    const auto& p_map = body_matcher.get_pattern_value_map();
    auto matched_scatter = ov::as_type_ptr<v3::ScatterUpdate>(p_map.at(scatter).get_node_shared_ptr());
    auto matched_state_out = ov::as_type_ptr<v1::Add>(p_map.at(state_out).get_node_shared_ptr());
    auto matched_core_attn_update = ov::as_type_ptr<v1::ReduceSum>(p_map.at(core_attn_update).get_node_shared_ptr());
    auto matched_kv_reduce = ov::as_type_ptr<v1::ReduceSum>(p_map.at(kv_reduce).get_node_shared_ptr());

    if (!matched_scatter || !matched_state_out || !matched_core_attn_update || !matched_kv_reduce) {
        return false;
    }

    if (body_results[1]->input_value(0).get_node_shared_ptr() != matched_state_out ||
        body_results[2]->input_value(0).get_node_shared_ptr() != matched_scatter) {
        return false;
    }

    if (matched_scatter->input_value(2).get_node_shared_ptr() != matched_core_attn_update || !is_reduce_sum(matched_core_attn_update, -2, true) ||
        !is_reduce_sum(matched_kv_reduce, -2, false)) {
        return false;
    }

    if (matched_state_out->output(0) != matched_core_attn_update->get_input_node_shared_ptr(0)->input_value(0)) {
        return false;
    }

    body_match.timestep_param = ov::as_type_ptr<v0::Parameter>(p_map.at(timestep_param).get_node_shared_ptr());
    body_match.query_param = ov::as_type_ptr<v0::Parameter>(p_map.at(query_param).get_node_shared_ptr());
    body_match.key_param = ov::as_type_ptr<v0::Parameter>(p_map.at(key_param).get_node_shared_ptr());
    body_match.value_param = ov::as_type_ptr<v0::Parameter>(p_map.at(value_param).get_node_shared_ptr());
    body_match.gate_param = ov::as_type_ptr<v0::Parameter>(p_map.at(gate_param).get_node_shared_ptr());
    body_match.beta_param = ov::as_type_ptr<v0::Parameter>(p_map.at(beta_param).get_node_shared_ptr());
    body_match.recurrent_state_param = ov::as_type_ptr<v0::Parameter>(p_map.at(recurrent_state_param).get_node_shared_ptr());
    body_match.core_attn_state_param = ov::as_type_ptr<v0::Parameter>(p_map.at(core_attn_state_param).get_node_shared_ptr());
    body_match.state_out = matched_state_out;
    body_match.scatter = matched_scatter;

    return body_match.timestep_param && body_match.query_param && body_match.key_param && body_match.value_param && body_match.gate_param &&
           body_match.beta_param && body_match.recurrent_state_param && body_match.core_attn_state_param;
}

bool match_recurrent_attention_loop(const std::shared_ptr<v5::Loop>& loop, LoopMatch& match) {
    if (!match_loop_signature(loop)) {
        return false;
    }

    const auto& body = loop->get_function();
    const auto& body_parameters = body->get_parameters();
    const auto& input_descs = loop->get_input_descriptions();

    LoopBodyMatch body_match;
    if (!match_loop_body_pattern(loop, body_match)) {
        return false;
    }

    const auto trip_count = loop->get_input_node_shared_ptr(0);
    const auto exec_cond = loop->get_input_node_shared_ptr(1);
    const auto trip_count_convert = ov::as_type_ptr<v0::Convert>(trip_count);
    const auto trip_count_gather = ov::as_type_ptr<v8::Gather>(trip_count_convert ? trip_count_convert->get_input_node_shared_ptr(0) : nullptr);
    const auto trip_count_shape = ov::as_type_ptr<v3::ShapeOf>(trip_count_gather ? trip_count_gather->get_input_node_shared_ptr(0) : nullptr);
    if (!trip_count_convert || !trip_count_gather || !trip_count_shape || !is_true_constant(exec_cond) ||
        !is_constant_vector(trip_count_gather->get_input_node_shared_ptr(1), {2}) ||
        !is_constant_vector(trip_count_gather->get_input_node_shared_ptr(2), {0})) {
        return false;
    }

    bool has_query = false;
    bool has_key = false;
    bool has_value = false;
    bool has_gate = false;
    bool has_beta = false;
    bool has_state = false;
    bool has_core_attn = false;
    Output<Node> core_attn_init;

    for (const auto& desc : input_descs) {
        const auto& body_param = body_parameters[desc->m_body_parameter_index];
        if (body_param == body_match.query_param) {
            const auto sliced = ov::as_type_ptr<SlicedD>(desc);
            if (!is_expected_sliced_input(sliced)) {
                return false;
            }
            match.query = loop->input_value(desc->m_input_index);
            has_query = true;
        } else if (body_param == body_match.key_param) {
            const auto sliced = ov::as_type_ptr<SlicedD>(desc);
            if (!is_expected_sliced_input(sliced)) {
                return false;
            }
            match.key = loop->input_value(desc->m_input_index);
            has_key = true;
        } else if (body_param == body_match.value_param) {
            const auto sliced = ov::as_type_ptr<SlicedD>(desc);
            if (!is_expected_sliced_input(sliced)) {
                return false;
            }
            match.value = loop->input_value(desc->m_input_index);
            has_value = true;
        } else if (body_param == body_match.gate_param) {
            const auto sliced = ov::as_type_ptr<SlicedD>(desc);
            if (!is_expected_sliced_input(sliced)) {
                return false;
            }
            match.gate = loop->input_value(desc->m_input_index);
            has_gate = true;
        } else if (body_param == body_match.beta_param) {
            const auto sliced = ov::as_type_ptr<SlicedD>(desc);
            if (!is_expected_sliced_input(sliced)) {
                return false;
            }
            match.beta = loop->input_value(desc->m_input_index);
            has_beta = true;
        } else if (body_param == body_match.recurrent_state_param) {
            const auto merged = ov::as_type_ptr<MergedD>(desc);
            if (!merged || body->get_results()[merged->m_body_value_index]->input_value(0) != body_match.state_out->output(0)) {
                return false;
            }
            match.recurrent_state = loop->input_value(desc->m_input_index);
            has_state = true;
        } else if (body_param == body_match.core_attn_state_param) {
            const auto merged = ov::as_type_ptr<MergedD>(desc);
            if (!merged || body->get_results()[merged->m_body_value_index]->input_value(0) != body_match.scatter->output(0)) {
                return false;
            }
            core_attn_init = loop->input_value(desc->m_input_index);
            has_core_attn = true;
        } else if (body_param == body_match.timestep_param) {
            if (!ov::as_type_ptr<MergedD>(desc)) {
                return false;
            }
        }
    }

    if (!(has_query && has_key && has_value && has_gate && has_beta && has_state && has_core_attn)) {
        return false;
    }

    return trip_count_shape->input_value(0) == match.value && is_exporter_zero_broadcast(match.value, core_attn_init);
}
}  // namespace

/// Matches the graph emitted by optimum-intel's RecurrentAttentionCell conversion:
///
///   Concat(
///       Reshape(Loop(...).output(0), [-1]),
///       Reshape(Loop(...).output(1), [-1]),
///       axis=0)
///
/// where Loop iterates over sequence axis 2 of head-first inputs [B, H, S, D].
/// The loop body implements one recurrent attention step and accumulates the full
/// attention tensor with ScatterUpdate. The fused replacement inserts GatedDeltaNet
/// in seq-first layout [B, S, H, D], then transposes outputs back and rebuilds the
/// original flat concat so downstream consumers remain unchanged.

GatedDeltaNetFusion::GatedDeltaNetFusion() {
    MATCHER_SCOPE(GatedDeltaNetFusion);

    auto loop = pattern::wrap_type<v5::Loop>();
    auto attn_flat = pattern::wrap_type<v1::Reshape>({loop, pattern::any_input()});
    auto state_flat = pattern::wrap_type<v1::Reshape>({loop, pattern::any_input()});
    auto concat = pattern::wrap_type<v0::Concat>({attn_flat, state_flat});

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        auto concat_node = ov::as_type_ptr<v0::Concat>(m.get_match_root());
        if (!concat_node || concat_node->get_axis() != 0 || concat_node->inputs().size() != 2) {
            return false;
        }

        if (transformation_callback(concat_node)) {
            return false;
        }

        auto attn_reshape = ov::as_type_ptr<v1::Reshape>(concat_node->get_input_node_shared_ptr(0));
        auto state_reshape = ov::as_type_ptr<v1::Reshape>(concat_node->get_input_node_shared_ptr(1));
        if (!is_flatten_reshape(attn_reshape) || !is_flatten_reshape(state_reshape)) {
            return false;
        }

        if (attn_reshape->input_value(0).get_node() != state_reshape->input_value(0).get_node()) {
            return false;
        }

        auto loop_node = ov::as_type_ptr<v5::Loop>(attn_reshape->get_input_node_shared_ptr(0));
        if (!loop_node || loop_node->output(0).get_target_inputs().size() != 1 || loop_node->output(1).get_target_inputs().size() != 1) {
            return false;
        }

        LoopMatch match;
        if (!match_recurrent_attention_loop(loop_node, match)) {
            return false;
        }

        const auto q_perm = v0::Constant::create(element::i64, Shape{4}, {0, 2, 1, 3});
        const auto g_perm = v0::Constant::create(element::i64, Shape{3}, {0, 2, 1});
        const auto q_back_perm = v0::Constant::create(element::i64, Shape{4}, {0, 2, 1, 3});
        const auto flatten_shape = v0::Constant::create(element::i32, Shape{1}, {-1});

        auto q_seq = std::make_shared<v1::Transpose>(match.query, q_perm);
        auto k_seq = std::make_shared<v1::Transpose>(match.key, q_perm);
        auto v_seq = std::make_shared<v1::Transpose>(match.value, q_perm);
        auto g_seq = std::make_shared<v1::Transpose>(match.gate, g_perm);
        auto beta_seq = std::make_shared<v1::Transpose>(match.beta, g_perm);

        auto gdn = std::make_shared<ov::op::internal::GatedDeltaNet>(q_seq,
                                                                     k_seq,
                                                                     v_seq,
                                                                     match.recurrent_state,
                                                                     g_seq,
                                                                     beta_seq);

        auto attn_head_first = std::make_shared<v1::Transpose>(gdn->output(0), q_back_perm);
        auto attn_flat_new = std::make_shared<v1::Reshape>(attn_head_first, flatten_shape, false);
        auto state_flat_new = std::make_shared<v1::Reshape>(gdn->output(1), flatten_shape, false);
        auto concat_new = std::make_shared<v0::Concat>(OutputVector{attn_flat_new, state_flat_new}, 0);
        concat_new->set_friendly_name(concat_node->get_friendly_name());

        ov::copy_runtime_info(NodeVector{concat_node, attn_reshape, state_reshape, loop_node},
                              NodeVector{q_seq, k_seq, v_seq, g_seq, beta_seq, gdn, attn_head_first, attn_flat_new, state_flat_new, concat_new});
        register_new_node(gdn);
        ov::replace_node(concat_node, concat_new);
        return true;
    };

    auto matcher = std::make_shared<pattern::Matcher>(concat, matcher_name);
    register_matcher(matcher, callback);
}

}  // namespace ov::pass
