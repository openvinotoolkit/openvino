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
#include "transformations/symbolic_transformations/symbolic_optimizations.hpp"
#include "transformations/utils/gen_pattern.hpp"
#include "transformations/utils/utils.hpp"

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
    auto skip_convert = [](std::shared_ptr<ov::Node> n) -> std::shared_ptr<ov::Node> {
        while (n) {
            auto cvt = ov::as_type_ptr<ov::op::v0::Convert>(n);
            if (!cvt || cvt->get_input_size() == 0) {
                break;
            }
            n = cvt->input_value(0).get_node_shared_ptr();
        }
        return n;
    };

    std::shared_ptr<ov::op::v3::ScatterUpdate> scatter_result;
    std::shared_ptr<ov::op::v1::Add> add_result;
    bool has_bool_const_result = false;
    for (const auto& result : body->get_results()) {
        auto src_raw = result->input_value(0).get_node_shared_ptr();
        auto src = skip_convert(src_raw);
        if (!scatter_result) {
            scatter_result = ov::as_type_ptr<ov::op::v3::ScatterUpdate>(src);
        }
        if (!add_result) {
            add_result = ov::as_type_ptr<ov::op::v1::Add>(src);
        }
        auto c = ov::as_type_ptr<ov::op::v0::Constant>(src_raw);
        if (c && c->get_element_type() == ov::element::boolean) {
            has_bool_const_result = true;
        }
    }
    if (!scatter_result || !add_result || !has_bool_const_result) {
        return false;
    }

    // Scatter update value must be ReduceSum(keep_dims=true) over Multiply(Add(state), Unsqueeze(Squeeze(q))).
    auto rs_out = ov::as_type_ptr<ov::op::v1::ReduceSum>(skip_convert(get_input_node(scatter_result, 2)));
    if (!rs_out || !rs_out->get_keep_dims()) {
        return false;
    }
    auto mul_out = ov::as_type_ptr<ov::op::v1::Multiply>(skip_convert(get_input_node(rs_out, 0)));
    if (!mul_out) {
        return false;
    }
    auto add_state = ov::as_type_ptr<ov::op::v1::Add>(skip_convert(get_input_node(mul_out, 0)));
    auto unsq_q = ov::as_type_ptr<ov::op::v0::Unsqueeze>(skip_convert(get_input_node(mul_out, 1)));
    if (!add_state || !unsq_q) {
        add_state = ov::as_type_ptr<ov::op::v1::Add>(skip_convert(get_input_node(mul_out, 1)));
        unsq_q = ov::as_type_ptr<ov::op::v0::Unsqueeze>(skip_convert(get_input_node(mul_out, 0)));
    }
    if (!add_state || !unsq_q) {
        return false;
    }
    if (!ov::as_type_ptr<ov::op::v0::Squeeze>(skip_convert(get_input_node(unsq_q, 0)))) {
        return false;
    }
    if (add_state != add_result) {
        return false;
    }

    // Add(state) must combine: state_gated and outer_update
    auto add_in0_mul = ov::as_type_ptr<ov::op::v1::Multiply>(skip_convert(get_input_node(add_state, 0)));
    auto add_in1_mul = ov::as_type_ptr<ov::op::v1::Multiply>(skip_convert(get_input_node(add_state, 1)));
    if (!add_in0_mul || !add_in1_mul) {
        return false;
    }

    auto is_state_gated_mul = [&](const std::shared_ptr<ov::op::v1::Multiply>& m) {
        auto u0 = ov::as_type_ptr<ov::op::v0::Unsqueeze>(skip_convert(get_input_node(m, 0)));
        auto u1 = ov::as_type_ptr<ov::op::v0::Unsqueeze>(skip_convert(get_input_node(m, 1)));
        return (u0 && ov::as_type_ptr<ov::op::v0::Exp>(skip_convert(get_input_node(u0, 0)))) ||
               (u1 && ov::as_type_ptr<ov::op::v0::Exp>(skip_convert(get_input_node(u1, 0))));
    };
    auto is_outer_update_mul = [&](const std::shared_ptr<ov::op::v1::Multiply>& m) {
        auto u0 = ov::as_type_ptr<ov::op::v0::Unsqueeze>(skip_convert(get_input_node(m, 0)));
        auto u1 = ov::as_type_ptr<ov::op::v0::Unsqueeze>(skip_convert(get_input_node(m, 1)));
        if (!u0 || !u1) {
            return false;
        }
        return ov::as_type_ptr<ov::op::v0::Squeeze>(skip_convert(get_input_node(u0, 0))) ||
               ov::as_type_ptr<ov::op::v0::Squeeze>(skip_convert(get_input_node(u1, 0)));
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
    auto ou_u0 = ov::as_type_ptr<ov::op::v0::Unsqueeze>(skip_convert(get_input_node(outer_update_mul, 0)));
    auto ou_u1 = ov::as_type_ptr<ov::op::v0::Unsqueeze>(skip_convert(get_input_node(outer_update_mul, 1)));
    if (!ou_u0 || !ou_u1) {
        return false;
    }
    auto delta_mul = ov::as_type_ptr<ov::op::v1::Multiply>(skip_convert(get_input_node(ou_u0, 0)));
    if (!delta_mul) {
        delta_mul = ov::as_type_ptr<ov::op::v1::Multiply>(skip_convert(get_input_node(ou_u1, 0)));
    }
    if (!delta_mul) {
        return false;
    }
    auto sub = ov::as_type_ptr<ov::op::v1::Subtract>(skip_convert(get_input_node(delta_mul, 0)));
    if (!sub) {
        sub = ov::as_type_ptr<ov::op::v1::Subtract>(skip_convert(get_input_node(delta_mul, 1)));
    }
    if (!sub) {
        return false;
    }
    auto rs_mid = ov::as_type_ptr<ov::op::v1::ReduceSum>(skip_convert(get_input_node(sub, 0)));
    if (!rs_mid) {
        rs_mid = ov::as_type_ptr<ov::op::v1::ReduceSum>(skip_convert(get_input_node(sub, 1)));
    }
    if (!rs_mid || rs_mid->get_keep_dims()) {
        return false;
    }
    auto mul_mid = ov::as_type_ptr<ov::op::v1::Multiply>(skip_convert(get_input_node(rs_mid, 0)));
    if (!mul_mid) {
        return false;
    }
    if (skip_convert(get_input_node(mul_mid, 0)) != state_gated_mul &&
        skip_convert(get_input_node(mul_mid, 1)) != state_gated_mul) {
        return false;
    }
    return true;
}

}  // namespace
using namespace ov::gen_pattern;
using namespace ov::pass::pattern;

static std::shared_ptr<ov::Node> get_scale(std::shared_ptr<ov::Node> scale_pattern,
                                           ov::element::Type default_scale_type,
                                           Matcher& matcher) {
    auto& pm = matcher.get_pattern_value_map();
    if (pm.count(scale_pattern)) {
        auto scale_node = pm.at(scale_pattern);

        // According to the spec, scale should be a scalar or 1D with 1 element
        const auto& pshape = scale_node.get_partial_shape();
        auto rank = pshape.rank();
        if (rank.is_dynamic()) {
            return nullptr;
        }

        if (pshape.is_static() && ov::shape_size(pshape.get_shape()) != 1) {
            return nullptr;
        } else {
            if (rank.get_length() > 1) {
                scale_node = ov::op::util::make_try_fold<ov::op::v1::Reshape>(
                    scale_node,
                    ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                    false);
            }
            return scale_node.get_node_shared_ptr();
        }
    } else {
        return ov::op::v0::Constant::create(default_scale_type, ov::Shape{}, {1.0});
    }
}

ov::pass::RemoveConcatSliceAfterLoop::RemoveConcatSliceAfterLoop() {
    auto value = ov::pass::pattern::any_input(rank_equals(4) && shape_matches("[?, head_num, ?, v_head_size]"));
    auto init_state =
        ov::pass::pattern::any_input(rank_equals(4) && shape_matches("[?, head_num, k_head_size, v_head_size]"));
    auto loop_output0 = ov::pass::pattern::wrap_type<ov::op::v5::Loop>(OutputVector{any_input(),
                                                                                    any_input(),
                                                                                    any_input(),
                                                                                    any_input(),
                                                                                    value->output(0),
                                                                                    any_input(),
                                                                                    any_input(),
                                                                                    init_state->output(0),
                                                                                    any_input()},
                                                                       pattern::output_index_matches(0));

    auto loop_output1 = ov::pass::pattern::wrap_type<ov::op::v5::Loop>(OutputVector{any_input(),
                                                                                    any_input(),
                                                                                    any_input(),
                                                                                    any_input(),
                                                                                    value->output(0),
                                                                                    any_input(),
                                                                                    any_input(),
                                                                                    init_state->output(0),
                                                                                    any_input()},
                                                                       pattern::output_index_matches(1));

    auto reshape_core_attn = pattern::wrap_type<ov::op::v1::Reshape>({loop_output0, {-1}});
    auto reshape_core_state = pattern::wrap_type<ov::op::v1::Reshape>({loop_output1, {-1}});
    auto concat_loop = makeOP<ov::op::v0::Concat>({reshape_core_attn, reshape_core_state}, {{"axis", 0}});
    auto slice_attn = pattern::wrap_type<ov::op::v8::Slice>({concat_loop, {0}, any_input(), {1}, {0}});
    auto reshape_attn = pattern::wrap_type<opset1::Reshape>({slice_attn, any_input()},
                                                            pattern::shape_matches("[?, head_num, ?, v_head_size]"));
    auto slice_state = pattern::wrap_type<ov::op::v8::Slice>({concat_loop, any_input(), {LLONG_MAX}, {1}, {0}});
    auto reshape_state = pattern::wrap_type<opset1::Reshape>({slice_state, any_input()});
    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        bool changed = false;
        auto loop_node = pattern_map.at(loop_output0).get_node_shared_ptr();
        if (pattern_map.count(reshape_attn) != 0) {
            auto reshape_attn_out = pattern_map.at(reshape_attn);
            if (!ov::replace_output_update_name(pattern_map.at(reshape_attn), loop_node->output(0))) {
                reshape_attn_out.replace(loop_node->output(0));
            }
            changed = true;
        }

        if (pattern_map.count(reshape_state) != 0) {
            auto reshape_state_out = pattern_map.at(reshape_state);
            if (!ov::replace_output_update_name(reshape_state_out, loop_node->output(1))) {
                reshape_state_out.replace(loop_node->output(1));
            }
            changed = true;
        }
        return changed;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(reshape_state | reshape_attn, "RemoveConcatSliceAfterLoop");
    register_matcher(m, callback);
}

ov::pass::FuseGDNLoop::FuseGDNLoop() {
    auto query = ov::pass::pattern::any_input(rank_equals(4) && shape_matches("[?, ?, head_num, qk_head_size]"));
    auto key = ov::pass::pattern::any_input(rank_equals(4) && shape_matches("[?, ?, head_num, qk_head_size]"));
    auto value = ov::pass::pattern::any_input(rank_equals(4) && shape_matches("[?, ?, head_num, value_head_size]"));
    auto init_state = ov::pass::pattern::any_input(rank_equals(4));
    auto gate = ov::pass::pattern::any_input(rank_equals(3) && shape_matches("[?, ?, head_num]"));
    auto beta = ov::pass::pattern::any_input(rank_equals(3) && shape_matches("[?, ?, head_num]"));

    auto transpose_query = pattern::wrap_type<ov::op::v1::Transpose>({query, {0, 2, 1, 3}});
    auto transpose_key = pattern::wrap_type<ov::op::v1::Transpose>({key, {0, 2, 1, 3}});
    auto transpose_value = pattern::wrap_type<ov::op::v1::Transpose>({value, {0, 2, 1, 3}});
    auto transpose_gate = pattern::wrap_type<ov::op::v1::Transpose>({gate, {0, 2, 1}});
    auto transpose_beta = pattern::wrap_type<ov::op::v1::Transpose>({beta, {0, 2, 1}});
    auto attn_scale = any_input();
    auto q_scale = pattern::wrap_type<ov::op::v1::Divide>({transpose_query, attn_scale});

    auto loop_output0 = ov::pass::pattern::wrap_type<ov::op::v5::Loop>(OutputVector{any_input(),
                                                                                    any_input(),
                                                                                    q_scale->output(0),
                                                                                    transpose_key->output(0),
                                                                                    transpose_value->output(0),
                                                                                    transpose_gate->output(0),
                                                                                    transpose_beta->output(0),
                                                                                    init_state->output(0),
                                                                                    any_input()},
                                                                       pattern::output_index_matches(0));

    auto transpose_loop_out = pattern::wrap_type<ov::op::v1::Transpose>({loop_output0, {0, 2, 1, 3}});

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto loop = ov::as_type_ptr<ov::op::v5::Loop>(pattern_map.at(loop_output0).get_node_shared_ptr());
        if (!matches_linear_attention_loop(loop)) {
            return false;
        }
        std::vector<std::shared_ptr<ov::Node>> rt_nodes{loop};
        ov::OutputVector inputs;
        inputs.reserve(6);
        std::shared_ptr<ov::Node> scale_node;
        auto q_type = pattern_map.at(transpose_query).get_element_type();
        if (!(scale_node = get_scale(attn_scale, q_type, m)))
            return false;
        inputs.push_back(pattern_map.at(query));       // query
        inputs.push_back(pattern_map.at(key));         // key
        inputs.push_back(pattern_map.at(value));       // value
        inputs.push_back(pattern_map.at(init_state));  // initial_state
        inputs.push_back(pattern_map.at(gate));        // g
        inputs.push_back(pattern_map.at(beta));        // beta

        auto linear_attn = std::make_shared<ov::op::GatedDeltaNet>(inputs);

        linear_attn->set_friendly_name(loop->get_friendly_name());

        ov::op::GatedDeltaNet::Config config;
        config.fuse_qk_l2norm = false;
        config.fuse_q_scale = true;
        linear_attn->set_config(config);
        ov::copy_runtime_info(rt_nodes, linear_attn);
        ov::replace_node(loop, linear_attn);
        register_new_node(linear_attn);
        auto transpose_attn = pattern_map.at(transpose_loop_out);
        if (!ov::replace_output_update_name(transpose_attn, linear_attn->output(0))) {
            transpose_attn.replace(linear_attn->output(0));
        }
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(transpose_loop_out, "FuseGDNLoop");
    register_matcher(m, callback);
}

ov::pass::FuseL2NormIntoGDN::FuseL2NormIntoGDN() {
    auto query = ov::pass::pattern::any_input(rank_equals(4) && shape_matches("[?, ?, head_num, qk_head_size]"));
    auto key = ov::pass::pattern::any_input(rank_equals(4) && shape_matches("[?, ?, head_num, qk_head_size]"));
    auto value = ov::pass::pattern::any_input(rank_equals(4) && shape_matches("[?, ?, head_num, value_head_size]"));
    auto init_state = ov::pass::pattern::any_input(rank_equals(4));
    auto gate = ov::pass::pattern::any_input(rank_equals(3) && shape_matches("[?, ?, head_num]"));
    auto beta = ov::pass::pattern::any_input(rank_equals(3) && shape_matches("[?, ?, head_num]"));
    auto eps_q_const = pattern::wrap_type<ov::op::v0::Constant>();
    auto eps_q = pattern::optional<ov::op::v0::Convert>({eps_q_const});
    auto eps_k_const = pattern::wrap_type<ov::op::v0::Constant>();
    auto eps_k = pattern::optional<ov::op::v0::Convert>({eps_k_const});

    auto l2_norm = [](const ov::Output<ov::Node>& data, const ov::Output<ov::Node>& eps) {
        auto input_convert = pattern::optional<ov::op::v0::Convert>({data});
        auto mul = pattern::wrap_type<ov::op::v1::Multiply>({input_convert, input_convert});
        auto axis_const = pattern::wrap_type<ov::op::v0::Constant>(value_matches("-1") || value_matches("3"));
        auto axis = pattern::optional<ov::op::v0::Convert>({axis_const});
        auto reduce_sum = pattern::wrap_type<ov::op::v1::ReduceSum>({mul, axis}, {{"keep_dims", true}});
        auto add = pattern::wrap_type<ov::op::v1::Add>({reduce_sum, eps});
        auto sqrt = pattern::wrap_type<ov::op::v0::Sqrt>({add});
        auto const_one = pattern::wrap_type<ov::op::v0::Constant>(value_matches("1"));
        auto convert_one = pattern::optional<ov::op::v0::Convert>({const_one});
        auto div = pattern::wrap_type<ov::op::v1::Divide>({any_input(), sqrt});
        auto multiply = pattern::wrap_type<ov::op::v1::Multiply>({input_convert, div});
        return multiply;
    };

    auto normalized_query = l2_norm(query, eps_q);
    auto normalized_key = l2_norm(key, eps_k);
    auto input_query = pattern::optional<ov::op::v0::Convert>({normalized_query});
    auto input_key = pattern::optional<ov::op::v0::Convert>({normalized_key});

    auto gdn = pattern::wrap_type<ov::op::GatedDeltaNet>({input_query, input_key, value, init_state, gate, beta});
    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto gdn_node = ov::as_type_ptr<ov::op::GatedDeltaNet>(pattern_map.at(gdn).get_node_shared_ptr());

        auto config = gdn_node->get_config();
        config.fuse_qk_l2norm = true;
        auto l2_norm_eps_node = pattern_map.at(eps_q_const).get_node_shared_ptr();
        config.l2_norm_eps = ov::as_type_ptr<ov::op::v0::Constant>(l2_norm_eps_node)->cast_vector<float>()[0];
        gdn_node->set_config(config);
        auto new_gdn = gdn_node->clone_with_new_inputs({pattern_map.at(query),
                                                        pattern_map.at(key),
                                                        pattern_map.at(value),
                                                        pattern_map.at(init_state),
                                                        pattern_map.at(gate),
                                                        pattern_map.at(beta)});
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
    symbolic_ctx_manager->register_pass<ov::pass::FuseL2NormIntoGDN>();
    return symbolic_optimizations.run_on_model(model);
}