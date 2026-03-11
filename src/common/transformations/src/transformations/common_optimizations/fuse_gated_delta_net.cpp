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
#include "transformations/symbolic_transformations/symbolic_optimizations.hpp"
#include "transformations/utils/gen_pattern.hpp"
#include "transformations/utils/print_model.hpp"
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
    if (!loop_output_matcher.match(body_results[2]->output(0)))
        return false;
    // match state
    if (!loop_state_matcher.match(body_results[1]->output(0)))
        return false;
    return true;
}

}  // namespace

static std::shared_ptr<ov::Node> get_scale(std::shared_ptr<ov::Node> scale_pattern,
                                           ov::element::Type default_scale_type,
                                           Matcher& matcher) {
    auto& pm = matcher.get_pattern_value_map();
    if (pm.count(scale_pattern)) {
        auto scale_node = pm.at(scale_pattern);

        // According to the spec, scale should be a scalar or 1D with 1 element
        const auto& pshape = scale_node.get_partial_shape();
        auto rank = pshape.rank();
        if (pshape.is_static() && ov::shape_size(pshape.get_shape()) != 1) {
            return nullptr;
        } else {
            if (rank.get_length() > 1) {
                scale_node = ov::op::util::make_try_fold<v1::Reshape>(scale_node,
                                                                      v0::Constant::create(ov::element::i64, {1}, {1}),
                                                                      false);
            }
            return scale_node.get_node_shared_ptr();
        }
    } else {
        return nullptr;
    }
}

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
    auto slice_attn = pattern::wrap_type<ov::op::v8::Slice>({concat_loop, {0}, any_input(), {1}, {0}});
    auto reshape_attn = pattern::wrap_type<v1::Reshape>({slice_attn, any_input()},
                                                        pattern::shape_matches("[?, head_num, ?, v_head_size]"));
    auto state_end = pattern::wrap_type<v0::Constant>(value_matches("LLONG_MAX") || value_matches("-1"));
    auto slice_state = pattern::wrap_type<ov::op::v8::Slice>({concat_loop, any_input(), state_end, {1}, {0}});
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

    auto m = std::make_shared<ov::pass::pattern::Matcher>(reshape_state | reshape_attn, "RemoveConcatSliceAfterLoop");
    register_matcher(m, callback);
}

ov::pass::FuseGDNLoop::FuseGDNLoop() {
    auto query = ov::pass::pattern::any_input(shape_matches("[?, ?, head_num, qk_head_size]"));
    auto key = ov::pass::pattern::any_input(shape_matches("[?, ?, head_num, qk_head_size]"));
    auto value = ov::pass::pattern::any_input(shape_matches("[?, ?, head_num, value_head_size]"));
    auto init_state = ov::pass::pattern::any_input(shape_matches("[?, head_num, qk_head_size, value_head_size]"));
    auto gate = ov::pass::pattern::any_input(shape_matches("[?, ?, head_num]"));
    auto beta = ov::pass::pattern::any_input(shape_matches("[?, ?, head_num]"));

    auto transpose_query = pattern::wrap_type<v1::Transpose>({query, {0, 2, 1, 3}});
    auto transpose_key = pattern::wrap_type<v1::Transpose>({key, {0, 2, 1, 3}});
    auto transpose_value = pattern::wrap_type<v1::Transpose>({value, {0, 2, 1, 3}});
    auto transpose_gate = pattern::wrap_type<v1::Transpose>({gate, {0, 2, 1}});
    auto transpose_beta = pattern::wrap_type<v1::Transpose>({beta, {0, 2, 1}});
    auto attn_scale = any_input(has_static_rank());
    auto q_scale = pattern::wrap_type<v1::Divide>({transpose_query, attn_scale});

    auto loop_output0 = ov::pass::pattern::wrap_type<ov::op::v5::Loop>(
        OutputVector{any_input(),
                     any_input(),
                     q_scale,
                     transpose_key,
                     transpose_value,
                     transpose_gate,
                     transpose_beta,
                     init_state,
                     any_input()},
        pattern::output_index_matches(0) && [](std::shared_ptr<ov::Node> node) -> bool {
            return matches_linear_attention_loop(node);
        });

    auto transpose_loop_out = pattern::wrap_type<v1::Transpose>({loop_output0, {0, 2, 1, 3}});

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto loop_node = pattern_map.at(loop_output0).get_node_shared_ptr();

        std::shared_ptr<ov::Node> scale_node;
        auto q_type = pattern_map.at(transpose_query).get_element_type();
        if (!(scale_node = get_scale(attn_scale, q_type, m)))
            return false;

        ov::OutputVector inputs = {
            pattern_map.at(query),       // query
            pattern_map.at(key),         // key
            pattern_map.at(value),       // value
            pattern_map.at(init_state),  // initial_state
            pattern_map.at(gate),        // g
            pattern_map.at(beta)         // beta
        };

        auto linear_attn = std::make_shared<ov::op::GatedDeltaNet>(inputs);

        linear_attn->set_friendly_name(loop_node->get_friendly_name());

        ov::op::GatedDeltaNet::Config config;
        config.fuse_qk_l2norm = false;
        config.fuse_q_scale = true;
        linear_attn->set_config(config);
        ov::copy_runtime_info(loop_node, linear_attn);
        ov::replace_node(loop_node, linear_attn);
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
    auto query = ov::pass::pattern::any_input(shape_matches("[?, ?, head_num, qk_head_size]"));
    auto key = ov::pass::pattern::any_input(shape_matches("[?, ?, head_num, qk_head_size]"));
    auto value = ov::pass::pattern::any_input(shape_matches("[?, ?, head_num, value_head_size]"));
    auto init_state = ov::pass::pattern::any_input(rank_equals(4));
    auto gate = ov::pass::pattern::any_input(shape_matches("[?, ?, head_num]"));
    auto beta = ov::pass::pattern::any_input(shape_matches("[?, ?, head_num]"));
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
        auto div = pattern::wrap_type<v1::Divide>({any_input(), sqrt});
        auto multiply = pattern::wrap_type<v1::Multiply>({input_convert, div});
        return multiply;
    };

    auto normalized_query = l2_norm(query, eps_q);
    auto normalized_key = l2_norm(key, eps_k);
    auto input_query = pattern::optional<v0::Convert>({normalized_query});
    auto input_key = pattern::optional<v0::Convert>({normalized_key});

    auto gdn = pattern::wrap_type<ov::op::GatedDeltaNet>({input_query, input_key, value, init_state, gate, beta});
    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto gdn_node = ov::as_type_ptr<ov::op::GatedDeltaNet>(pattern_map.at(gdn).get_node_shared_ptr());

        auto config = gdn_node->get_config();
        config.fuse_qk_l2norm = true;
        auto l2_norm_q_eps_node = pattern_map.at(eps_q_const).get_node_shared_ptr();
        auto l2_norm_k_eps_node = pattern_map.at(eps_k_const).get_node_shared_ptr();

        config.q_l2_norm_eps = ov::as_type_ptr<v0::Constant>(l2_norm_q_eps_node)->cast_vector<float>()[0];
        config.k_l2_norm_eps = ov::as_type_ptr<v0::Constant>(l2_norm_k_eps_node)->cast_vector<float>()[0];

        gdn_node->set_config(config);
        auto new_gdn = gdn_node->clone_with_new_inputs({pattern_map.at(query),
                                                        pattern_map.at(key),
                                                        pattern_map.at(value),
                                                        pattern_map.at(init_state),
                                                        pattern_map.at(gate),
                                                        pattern_map.at(beta)});
        ov::copy_runtime_info(gdn_node, new_gdn);
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