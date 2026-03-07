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
#include "openvino/op/loop.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/gen_pattern.hpp"

using namespace ov::pass;

namespace {

bool matches_linear_attention_loop(const std::shared_ptr<ov::op::v5::Loop>& loop) {
    if (!loop) {
        return false;
    }

    if (loop->get_input_size() < 9 || loop->get_output_size() != 2) {
        return false;
    }

    const auto& body = loop->get_function();
    if (!body) {
        return false;
    }

    if (body->get_parameters().size() < 8 || body->get_results().size() < 3) {
        return false;
    }

    const auto& output_descs = loop->get_output_descriptions();
    if (output_descs.size() != 2) {
        return false;
    }

    auto get_input_node = [](const std::shared_ptr<ov::Node>& n, size_t idx) -> std::shared_ptr<ov::Node> {
        if (!n || n->get_input_size() <= idx) {
            return nullptr;
        }
        return n->input_value(idx).get_node_shared_ptr();
    };

    std::shared_ptr<ov::op::v3::ScatterUpdate> scatter_result;
    std::shared_ptr<ov::op::v1::Add> add_result;
    bool has_bool_const_result = false;
    for (const auto& result : body->get_results()) {
        auto src = result->input_value(0).get_node_shared_ptr();
        if (!scatter_result) {
            scatter_result = std::dynamic_pointer_cast<ov::op::v3::ScatterUpdate>(src);
        }
        if (!add_result) {
            add_result = std::dynamic_pointer_cast<ov::op::v1::Add>(src);
        }
        auto c = std::dynamic_pointer_cast<ov::op::v0::Constant>(src);
        if (c && c->get_element_type() == ov::element::boolean) {
            has_bool_const_result = true;
        }
    }
    if (!scatter_result || !add_result || !has_bool_const_result) {
        return false;
    }

    // Scatter update value must be ReduceSum(keep_dims=true) over Multiply(Add(state), Unsqueeze(Squeeze(q))).
    auto rs_out = std::dynamic_pointer_cast<ov::op::v1::ReduceSum>(get_input_node(scatter_result, 2));
    if (!rs_out || !rs_out->get_keep_dims()) {
        return false;
    }
    auto mul_out = std::dynamic_pointer_cast<ov::op::v1::Multiply>(get_input_node(rs_out, 0));
    if (!mul_out) {
        return false;
    }
    auto add_state = std::dynamic_pointer_cast<ov::op::v1::Add>(get_input_node(mul_out, 0));
    auto unsq_q = std::dynamic_pointer_cast<ov::op::v0::Unsqueeze>(get_input_node(mul_out, 1));
    if (!add_state || !unsq_q) {
        add_state = std::dynamic_pointer_cast<ov::op::v1::Add>(get_input_node(mul_out, 1));
        unsq_q = std::dynamic_pointer_cast<ov::op::v0::Unsqueeze>(get_input_node(mul_out, 0));
    }
    if (!add_state || !unsq_q) {
        return false;
    }
    if (!std::dynamic_pointer_cast<ov::op::v0::Squeeze>(get_input_node(unsq_q, 0))) {
        return false;
    }
    if (add_state != add_result) {
        return false;
    }

    // Add(state) must combine: state_gated and outer_update
    auto add_in0_mul = std::dynamic_pointer_cast<ov::op::v1::Multiply>(get_input_node(add_state, 0));
    auto add_in1_mul = std::dynamic_pointer_cast<ov::op::v1::Multiply>(get_input_node(add_state, 1));
    if (!add_in0_mul || !add_in1_mul) {
        return false;
    }

    auto is_state_gated_mul = [&](const std::shared_ptr<ov::op::v1::Multiply>& m) {
        auto u0 = std::dynamic_pointer_cast<ov::op::v0::Unsqueeze>(get_input_node(m, 0));
        auto u1 = std::dynamic_pointer_cast<ov::op::v0::Unsqueeze>(get_input_node(m, 1));
        return (u0 && std::dynamic_pointer_cast<ov::op::v0::Exp>(get_input_node(u0, 0))) ||
               (u1 && std::dynamic_pointer_cast<ov::op::v0::Exp>(get_input_node(u1, 0)));
    };
    auto is_outer_update_mul = [&](const std::shared_ptr<ov::op::v1::Multiply>& m) {
        auto u0 = std::dynamic_pointer_cast<ov::op::v0::Unsqueeze>(get_input_node(m, 0));
        auto u1 = std::dynamic_pointer_cast<ov::op::v0::Unsqueeze>(get_input_node(m, 1));
        if (!u0 || !u1) {
            return false;
        }
        return std::dynamic_pointer_cast<ov::op::v0::Squeeze>(get_input_node(u0, 0)) ||
               std::dynamic_pointer_cast<ov::op::v0::Squeeze>(get_input_node(u1, 0));
    };

    std::shared_ptr<ov::op::v1::Multiply> state_gated_mul;
    std::shared_ptr<ov::op::v1::Multiply> outer_update_mul;
    if (is_state_gated_mul(add_in0_mul) && is_outer_update_mul(add_in1_mul)) {
        state_gated_mul = add_in0_mul;
        outer_update_mul = add_in1_mul;
    } else if (is_state_gated_mul(add_in1_mul) && is_outer_update_mul(add_in0_mul)) {
        state_gated_mul = add_in1_mul;
        outer_update_mul = add_in0_mul;
    } else {
        return false;
    }

    // outer_update must consume Unsqueeze(Multiply(Subtract(...), beta))
    auto ou_u0 = std::dynamic_pointer_cast<ov::op::v0::Unsqueeze>(get_input_node(outer_update_mul, 0));
    auto ou_u1 = std::dynamic_pointer_cast<ov::op::v0::Unsqueeze>(get_input_node(outer_update_mul, 1));
    if (!ou_u0 || !ou_u1) {
        return false;
    }
    auto delta_mul = std::dynamic_pointer_cast<ov::op::v1::Multiply>(get_input_node(ou_u0, 0));
    if (!delta_mul) {
        delta_mul = std::dynamic_pointer_cast<ov::op::v1::Multiply>(get_input_node(ou_u1, 0));
    }
    if (!delta_mul) {
        return false;
    }
    auto sub = std::dynamic_pointer_cast<ov::op::v1::Subtract>(get_input_node(delta_mul, 0));
    if (!sub) {
        sub = std::dynamic_pointer_cast<ov::op::v1::Subtract>(get_input_node(delta_mul, 1));
    }
    if (!sub) {
        return false;
    }
    auto rs_mid = std::dynamic_pointer_cast<ov::op::v1::ReduceSum>(get_input_node(sub, 0));
    if (!rs_mid) {
        rs_mid = std::dynamic_pointer_cast<ov::op::v1::ReduceSum>(get_input_node(sub, 1));
    }
    if (!rs_mid || rs_mid->get_keep_dims()) {
        return false;
    }
    auto mul_mid = std::dynamic_pointer_cast<ov::op::v1::Multiply>(get_input_node(rs_mid, 0));
    if (!mul_mid) {
        return false;
    }
    if (get_input_node(mul_mid, 0) != state_gated_mul && get_input_node(mul_mid, 1) != state_gated_mul) {
        return false;
    }
    return true;
}

template <typename T>
std::shared_ptr<T> get_single_consumer_as(const ov::Output<ov::Node>& output) {
    const auto& targets = output.get_target_inputs();
    if (targets.size() != 1) {
        return nullptr;
    }
    auto target_node = targets.begin()->get_node()->shared_from_this();
    return ov::as_type_ptr<T>(target_node);
}

std::shared_ptr<ov::op::v1::ReduceProd> find_reduce_prod_from_slice_like(const std::shared_ptr<ov::Node>& slice) {
    if (!slice) {
        return nullptr;
    }
    auto try_inputs = [&](const std::shared_ptr<ov::Node>& node) {
        for (size_t i = 1; i <= 2; ++i) {
            if (node->get_input_size() <= i) {
                continue;
            }
            auto reduce_prod =
                std::dynamic_pointer_cast<ov::op::v1::ReduceProd>(node->input_value(i).get_node_shared_ptr());
            if (reduce_prod) {
                return reduce_prod;
            }
        }
        return std::shared_ptr<ov::op::v1::ReduceProd>{};
    };

    if (std::dynamic_pointer_cast<ov::op::v8::Slice>(slice) ||
        std::dynamic_pointer_cast<ov::op::v1::StridedSlice>(slice)) {
        return try_inputs(slice);
    }
    return nullptr;
}

bool uses_reduce_prod_at(const std::shared_ptr<ov::Node>& slice,
                         const std::shared_ptr<ov::op::v1::ReduceProd>& reduce_prod,
                         size_t idx) {
    if (!slice || !reduce_prod) {
        return false;
    }
    if (slice->get_input_size() <= idx) {
        return false;
    }
    return slice->input_value(idx).get_node_shared_ptr() == reduce_prod;
}

bool replace_concat_slice_with_linear_attention(const std::shared_ptr<ov::op::v5::Loop>& loop,
                                                const std::shared_ptr<ov::op::GatedDeltaNet>& linear_attn) {
    if (!loop || !linear_attn) {
        return false;
    }

    auto reshape_out0 = get_single_consumer_as<ov::op::v1::Reshape>(loop->output(0));
    auto reshape_out1 = get_single_consumer_as<ov::op::v1::Reshape>(loop->output(1));
    if (!reshape_out0 || !reshape_out1) {
        return false;
    }
    auto concat0 = get_single_consumer_as<ov::op::v0::Concat>(reshape_out0->output(0));
    auto concat1 = get_single_consumer_as<ov::op::v0::Concat>(reshape_out1->output(0));
    if (!concat0 || concat0 != concat1) {
        return false;
    }
    auto concat = concat0;
    const auto& concat_targets = concat->output(0).get_target_inputs();
    if (concat_targets.size() != 2) {
        return false;
    }
    std::vector<std::shared_ptr<ov::Node>> slices;
    slices.reserve(2);
    for (const auto& input : concat_targets) {
        auto slice_node = input.get_node()->shared_from_this();
        if (!std::dynamic_pointer_cast<ov::op::v8::Slice>(slice_node) &&
            !std::dynamic_pointer_cast<ov::op::v1::StridedSlice>(slice_node)) {
            return false;
        }
        slices.push_back(slice_node);
    }

    auto reduce_prod = find_reduce_prod_from_slice_like(slices[0]);
    if (!reduce_prod) {
        reduce_prod = find_reduce_prod_from_slice_like(slices[1]);
    }
    if (!reduce_prod) {
        return false;
    }
    const auto is_first_part = [&](const std::shared_ptr<ov::Node>& slice) {
        return uses_reduce_prod_at(slice, reduce_prod, 2) && !uses_reduce_prod_at(slice, reduce_prod, 1);
    };
    const auto is_second_part = [&](const std::shared_ptr<ov::Node>& slice) {
        return uses_reduce_prod_at(slice, reduce_prod, 1) && !uses_reduce_prod_at(slice, reduce_prod, 2);
    };

    std::shared_ptr<ov::Node> slice_value;
    std::shared_ptr<ov::Node> slice_state;
    for (const auto& slice : slices) {
        if (is_first_part(slice)) {
            slice_value = slice;
        }
        if (is_second_part(slice)) {
            slice_state = slice;
        }
    }
    if (!slice_value || !slice_state) {
        return false;
    }

    auto reshape_value = get_single_consumer_as<ov::op::v1::Reshape>(slice_value->output(0));
    auto reshape_state = get_single_consumer_as<ov::op::v1::Reshape>(slice_state->output(0));
    if (!reshape_value || !reshape_state) {
        return false;
    }

    // output0 path in target graph: reshape -> transpose -> reshape
    // Replace it with a direct reshape from GatedDeltaNet output(0).
    auto transpose_value = get_single_consumer_as<ov::op::v1::Transpose>(reshape_value->output(0));
    if (transpose_value) {
        auto reshape_value_final = get_single_consumer_as<ov::op::v1::Reshape>(transpose_value->output(0));
        if (reshape_value_final) {
            auto fused_reshape = std::make_shared<ov::op::v1::Reshape>(linear_attn->output(0),
                                                                       reshape_value_final->input_value(1),
                                                                       reshape_value_final->get_special_zero());
            ov::copy_runtime_info(std::vector<std::shared_ptr<ov::Node>>{linear_attn, reshape_value_final},
                                  fused_reshape);
            if (!ov::replace_output_update_name(reshape_value_final->output(0), fused_reshape->output(0))) {
                reshape_value_final->output(0).replace(fused_reshape->output(0));
            }
        } else {
            if (!ov::replace_output_update_name(transpose_value->output(0), linear_attn->output(0))) {
                transpose_value->output(0).replace(linear_attn->output(0));
            }
        }
    } else {
        if (!ov::replace_output_update_name(reshape_value->output(0), linear_attn->output(0))) {
            reshape_value->output(0).replace(linear_attn->output(0));
        }
    }

    // output1 path in target graph: concat -> slice -> reshape
    // Replace final reshape output directly with GatedDeltaNet output(1).
    if (!ov::replace_output_update_name(reshape_state->output(0), linear_attn->output(1))) {
        reshape_state->output(0).replace(linear_attn->output(1));
    }
    return true;
}

}  // namespace
using namespace ov::gen_pattern;
using namespace ov::pass::pattern;
ov::pass::GatedDeltaNetFusion::GatedDeltaNetFusion() {
    auto query = ov::pass::pattern::any_input(rank_equals(4));
    auto key = ov::pass::pattern::any_input(rank_equals(4));
    auto value = ov::pass::pattern::any_input(rank_equals(4));
    auto init_state = ov::pass::pattern::any_input(rank_equals(4));
    auto gate = ov::pass::pattern::any_input(rank_equals(3));
    auto beta = ov::pass::pattern::any_input(rank_equals(3));
    auto transpose_value = pattern::wrap_type<ov::op::v1::Transpose>({value, {0, 2, 1, 3}});
    auto transpose_gate = pattern::wrap_type<ov::op::v1::Transpose>({gate, {0, 2, 1}});
    auto transpose_beta = pattern::wrap_type<ov::op::v1::Transpose>({beta, {0, 2, 1}});

    auto axis_q_const = pattern::wrap_type<ov::op::v0::Constant>(value_matches("-1") || value_matches("3"));
    auto axis_q = pattern::optional<ov::op::v0::Convert>({axis_q_const});

    auto eps_q_const = pattern::wrap_type<ov::op::v0::Constant>();
    auto eps_q = pattern::optional<ov::op::v0::Convert>({eps_q_const});

    auto inv_const_q_const = pattern::wrap_type<ov::op::v0::Constant>(value_matches("1"));
    auto inv_const_q = pattern::optional<ov::op::v0::Convert>({inv_const_q_const});

    auto axis_k_const = pattern::wrap_type<ov::op::v0::Constant>(value_matches("-1") || value_matches("3"));
    auto axis_k = pattern::optional<ov::op::v0::Convert>({axis_k_const});

    auto eps_k_const = pattern::wrap_type<ov::op::v0::Constant>();
    auto eps_k = pattern::optional<ov::op::v0::Convert>({eps_k_const});

    auto inv_const_k_const = pattern::wrap_type<ov::op::v0::Constant>(value_matches("1"));
    auto inv_const_k = pattern::optional<ov::op::v0::Convert>({inv_const_k_const});

    auto Multiply_14 = pattern::wrap_type<ov::op::v1::Multiply>({query, query});
    auto ReduceSum_15 =
        pattern::wrap_type<ov::op::v1::ReduceSum>({Multiply_14, axis_q->output(0)}, {{"keep_dims", true}});
    auto Add_18 = pattern::wrap_type<ov::op::v1::Add>({ReduceSum_15, eps_q->output(0)});
    auto Sqrt_19 = pattern::wrap_type<ov::op::v0::Sqrt>({Add_18});
    auto Divide_20 = pattern::wrap_type<ov::op::v1::Divide>({inv_const_q->output(0), Sqrt_19});
    auto Power_20 = pattern::wrap_type<ov::op::v1::Power>({Sqrt_19, {-1}});
    auto inv_sqrt_q = std::make_shared<pattern::op::Or>(OutputVector{Divide_20, Power_20});
    auto Multiply_21 = pattern::wrap_type<ov::op::v1::Multiply>({query, inv_sqrt_q->output(0)});
    auto q_type_convert = pattern::optional<ov::op::v0::Convert>({Multiply_21});
    // q / sqrt(d)
    auto transpose_query = pattern::wrap_type<ov::op::v1::Transpose>({q_type_convert, {0, 2, 1, 3}});
    auto Multiply_32 = pattern::wrap_type<ov::op::v1::Divide>({transpose_query, any_input()});
    auto q_candidate = Multiply_32;

    auto Multiply_22 = pattern::wrap_type<ov::op::v1::Multiply>({key, key});
    auto ReduceSum_23 =
        pattern::wrap_type<ov::op::v1::ReduceSum>({Multiply_22, axis_k->output(0)}, {{"keep_dims", true}});
    auto Add_26 = pattern::wrap_type<ov::op::v1::Add>({ReduceSum_23, eps_k->output(0)});
    auto Sqrt_27 = pattern::wrap_type<ov::op::v0::Sqrt>({Add_26});
    auto Divide_28 = pattern::wrap_type<ov::op::v1::Divide>({inv_const_k->output(0), Sqrt_27});
    auto Power_28 = pattern::wrap_type<ov::op::v1::Power>({Sqrt_27, {-1}});
    auto inv_sqrt_k = std::make_shared<pattern::op::Or>(OutputVector{Divide_28, Power_28});
    auto Multiply_29 = pattern::wrap_type<ov::op::v1::Multiply>({key, inv_sqrt_k->output(0)});
    auto k_type_convert = pattern::optional<ov::op::v0::Convert>({Multiply_29});
    auto transpose_key = pattern::wrap_type<ov::op::v1::Transpose>({k_type_convert, {0, 2, 1, 3}});

    auto q_in = std::make_shared<pattern::op::Or>(OutputVector{q_candidate});
    auto k_in = std::make_shared<pattern::op::Or>(OutputVector{transpose_key});

    auto loop_label = ov::pass::pattern::wrap_type<ov::op::v5::Loop>(OutputVector{any_input(),
                                                                                  any_input(),
                                                                                  q_in->output(0),
                                                                                  k_in->output(0),
                                                                                  transpose_value->output(0),
                                                                                  transpose_gate->output(0),
                                                                                  transpose_beta->output(0),
                                                                                  init_state->output(0),
                                                                                  any_input()});

    matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto loop = std::dynamic_pointer_cast<ov::op::v5::Loop>(m.get_match_root());
        if (!matches_linear_attention_loop(loop)) {
            return false;
        }

        std::vector<std::shared_ptr<ov::Node>> rt_nodes{loop};
        ov::OutputVector inputs;
        inputs.reserve(6);

        inputs.push_back(pattern_map.at(query));       // query
        inputs.push_back(pattern_map.at(key));         // key
        inputs.push_back(pattern_map.at(value));       // value
        inputs.push_back(pattern_map.at(init_state));  // initial_state
        inputs.push_back(pattern_map.at(gate));        // g
        inputs.push_back(pattern_map.at(beta));        // beta

        auto linear_attn = std::make_shared<ov::op::GatedDeltaNet>(inputs);
        linear_attn->set_friendly_name(loop->get_friendly_name());
        ov::op::GatedDeltaNet::Config config;
        config.fuse_qk_l2norm = true;
        config.fuse_q_scale = true;
        linear_attn->set_config(config);
        ov::copy_runtime_info(rt_nodes, linear_attn);
        if (!replace_concat_slice_with_linear_attention(loop, linear_attn)) {
            return false;
        }
        ov::replace_node(loop, linear_attn);
        register_new_node(linear_attn);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(loop_label, "GatedDeltaNetFusion");
    register_matcher(m, callback);
}