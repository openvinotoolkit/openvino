// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_matmuls_fusion.hpp"

#include <initializer_list>
#include <memory>
#include <string>

#include "itt.hpp"
#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/cpu_opset/common/op/batch_gather_matmul.hpp"

using namespace ov::pass;
namespace {
// Note: intermediate nodes remain unchanged,
// but we need to explicitly call shape inference for them to keep shapes consistency
void validate_nodes(const pattern::PatternValueMap& map, const std::initializer_list<std::shared_ptr<ov::Node>> nodes) {
    for (const auto& node : nodes) {
        map.at(node).get_node_shared_ptr()->validate_and_infer_types();
    }
};

std::shared_ptr<ov::op::v0::Unsqueeze> introduce_n_experts_dim(const ov::Output<ov::Node>& data) {
    auto zero_const = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, 0);
    auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(data, zero_const);
    ov::copy_runtime_info(data.get_node_shared_ptr(), {unsqueeze, zero_const});
    return unsqueeze;
}
}  // namespace

ov::intel_cpu::MoE2GeMMFusion::MoE2GeMMFusion() {
    MATCHER_SCOPE(MoE2GeMMFusion);

    auto data_input = pattern::any_input(pattern::rank_equals(3) && pattern::has_static_rank());
    auto experts_input = pattern::wrap_type<ov::op::v1::Reshape>({data_input, pattern::any_input()});
    auto tile =
        pattern::wrap_type<ov::op::v0::Tile>({experts_input, pattern::any_input()}, pattern::consumers_count(1));
    auto after_tile_reshape =
        pattern::wrap_type<ov::op::v1::Reshape>({tile, pattern::any_input()}, pattern::consumers_count(1));
    auto gate_up_matmul = pattern::wrap_type<ov::op::v0::MatMul>({after_tile_reshape, pattern::any_input()},
                                                                 pattern::consumers_count(1),
                                                                 {{"transpose_a", false}, {"transpose_b", true}});
    auto gate_up_bias = pattern::wrap_const();
    auto gate_up_add = pattern::wrap_type<ov::op::v1::Add>({gate_up_matmul, gate_up_bias}, pattern::consumers_count(2));

    // Branch 1: Slice_1 -> Clamp -> Add_1
    auto slice1 = pattern::wrap_type<ov::op::v8::Slice>(
        {gate_up_add, pattern::any_input(), pattern::any_input(), pattern::any_input(), pattern::any_input()},
        pattern::consumers_count(1));
    auto clamp = pattern::wrap_type<ov::op::v0::Clamp>({slice1}, pattern::consumers_count(1));
    auto add1 = pattern::wrap_type<ov::op::v1::Add>({clamp, pattern::wrap_const()}, pattern::consumers_count(1));

    // Branch 2: Slice_2 -> Minimum_1 -> Swish
    auto slice2 = pattern::wrap_type<ov::op::v8::Slice>(
        {gate_up_add, pattern::any_input(), pattern::any_input(), pattern::any_input(), pattern::any_input()},
        pattern::consumers_count(1));
    auto minimum1 =
        pattern::wrap_type<ov::op::v1::Minimum>({slice2, pattern::wrap_const()}, pattern::consumers_count(1));
    auto swish_beta = pattern::wrap_const();
    auto swish = pattern::wrap_type<ov::op::v4::Swish>({minimum1, swish_beta}, pattern::consumers_count(1));

    // Join: Multiply_2
    auto multiply2 = pattern::wrap_type<ov::op::v1::Multiply>({add1, swish}, pattern::consumers_count(1));

    // Down projection
    auto down_proj_matmul = pattern::wrap_type<ov::op::v0::MatMul>({multiply2, pattern::any_input()},
                                                                   pattern::consumers_count(1),
                                                                   {{"transpose_a", false}, {"transpose_b", true}});
    auto down_proj_bias = pattern::wrap_const();
    auto down_proj_add =
        pattern::wrap_type<ov::op::v1::Add>({down_proj_matmul, down_proj_bias}, pattern::consumers_count(1));
    auto end_reshape_target_shape = pattern::any_input();
    auto end_reshape =
        pattern::wrap_type<ov::op::v1::Reshape>({down_proj_add, end_reshape_target_shape}, pattern::consumers_count(1));

    // Routing weights/mask
    auto zero_constant = pattern::wrap_type<ov::op::v0::Constant>(pattern::value_matches("0"));
    auto broadcasted_const = pattern::wrap_type<ov::op::v3::Broadcast>({zero_constant, pattern::any_input()}) |
                             pattern::wrap_type<ov::op::v1::Broadcast, ov::op::v3::Broadcast>(
                                 {zero_constant, pattern::any_input(), pattern::any_input()});
    // Routing weights/mask
    auto router_topk_indices = pattern::any_input();
    auto chosen_experts = pattern::any_input();
    auto slice_begin = pattern::wrap_type<ov::op::v0::Constant>(pattern::value_matches("0, 0"));
    auto slice_end = pattern::wrap_type<ov::op::v3::ShapeOf>({pattern::any_input()});
    auto slice_strides = pattern::wrap_type<ov::op::v0::Constant>(pattern::value_matches("1, 1"));
    auto slice_axes = pattern::wrap_type<ov::op::v0::Constant>(pattern::value_matches("0, 1"));
    auto slice =
        pattern::optional<ov::op::v8::Slice>({chosen_experts, slice_begin, slice_end, slice_strides, slice_axes});
    auto one_constant = pattern::wrap_type<ov::op::v0::Constant>(pattern::value_matches("1"));
    auto scatter_elements_update = pattern::wrap_type<ov::op::v3::ScatterElementsUpdate>(
                                       {broadcasted_const, router_topk_indices, chosen_experts, one_constant}) |
                                   pattern::wrap_type<ov::op::v12::ScatterElementsUpdate>(
                                       {broadcasted_const, router_topk_indices, chosen_experts, one_constant},
                                       {{"reduction", "none"}});

    auto router_transpose = pattern::wrap_type<ov::op::v1::Transpose>({scatter_elements_update, pattern::any_input()});
    auto router_reshape =
        pattern::wrap_type<ov::op::v1::Reshape>({router_transpose, pattern::any_input()},
                                                ov::pass::pattern::rank_equals(3) || ov::pass::pattern::rank_equals(4));
    auto unsqueeze_axis =
        pattern::wrap_type<ov::op::v0::Constant>(pattern::value_matches("3") || pattern::value_matches("-1"));
    auto unsqueeze_routing_weights =
        pattern::optional<ov::op::v0::Unsqueeze>({router_reshape, unsqueeze_axis}, ov::pass::pattern::rank_equals(4));

    auto mul3 =
        pattern::wrap_type<ov::op::v1::Multiply>({end_reshape, unsqueeze_routing_weights}, pattern::consumers_count(1));
    auto reduce_sum = pattern::wrap_type<ov::op::v1::ReduceSum>({mul3, pattern::any_input()},
                                                                pattern::consumers_count(1),
                                                                {{"keep_dims", false}});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto& experts_subgraph_input = pattern_map.at(experts_input);
        const auto& active_indices = pattern_map.at(router_topk_indices);

        const auto gate_up_mm_node = pattern_map.at(gate_up_matmul).get_node_shared_ptr();
        const auto gate_up_add_node = pattern_map.at(gate_up_add).get_node_shared_ptr();
        const auto gate_up_bias_node = pattern_map.at(gate_up_bias).get_node_shared_ptr();

        // BatchGatherMatmul A shape: [n_activated_experts, batch_size * seq_length, hidden_size]
        // Number of activated experts is always 1 for the first BatchGatherMatmul
        const auto unsqueeze = introduce_n_experts_dim(experts_subgraph_input);
        const auto gate_up_gathered_mm = std::make_shared<BatchGatherMatmul>(unsqueeze,
                                                                             gate_up_mm_node->input_value(1),
                                                                             active_indices,
                                                                             gate_up_bias_node);
        ov::replace_node_update_name(gate_up_add_node, gate_up_gathered_mm);

        validate_nodes(pattern_map, {slice1, clamp, add1, slice2, minimum1, swish, multiply2});

        const auto down_proj_mm_node = pattern_map.at(down_proj_matmul).get_node_shared_ptr();
        const auto down_proj_bias_node = pattern_map.at(down_proj_bias).get_node_shared_ptr();

        const auto down_gathered_mm = std::make_shared<BatchGatherMatmul>(pattern_map.at(multiply2),
                                                                          down_proj_mm_node->input_value(1),
                                                                          active_indices,
                                                                          down_proj_bias_node);
        ov::copy_runtime_info(down_proj_mm_node, down_gathered_mm);
        down_gathered_mm->set_friendly_name(down_proj_mm_node->get_friendly_name());

        const auto& chosen_experts_input = pattern_map.at(chosen_experts);
        const auto router_transpose_node = pattern_map.at(router_transpose).get_node_shared_ptr();
        const auto new_router_transpose =
            router_transpose_node->clone_with_new_inputs({chosen_experts_input, router_transpose_node->input_value(1)});
        ov::copy_runtime_info(router_transpose_node, new_router_transpose);
        new_router_transpose->set_friendly_name(router_transpose_node->get_friendly_name());

        const auto router_unsqueeze_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {-1});
        const auto router_unsqueeze =
            std::make_shared<ov::op::v0::Unsqueeze>(new_router_transpose, router_unsqueeze_const);
        ov::copy_runtime_info(router_transpose_node, {router_unsqueeze_const, router_unsqueeze});

        const auto final_mul_node = pattern_map.at(mul3).get_node_shared_ptr();
        const auto new_final_mul =
            final_mul_node->clone_with_new_inputs({down_gathered_mm->output(0), router_unsqueeze->output(0)});
        ov::copy_runtime_info(final_mul_node, new_final_mul);
        new_final_mul->set_friendly_name(final_mul_node->get_friendly_name());

        const auto reduce_sum_node = pattern_map.at(reduce_sum).get_node_shared_ptr();
        const auto new_reduce_sum =
            reduce_sum_node->clone_with_new_inputs({new_final_mul->output(0), reduce_sum_node->input_value(1)});
        ov::copy_runtime_info(reduce_sum_node, new_reduce_sum);
        new_reduce_sum->set_friendly_name(reduce_sum_node->get_friendly_name());

        const auto& end_reshape_out = pattern_map.at(end_reshape);
        const auto end_reshape_rank = end_reshape_out.get_partial_shape().rank();
        const auto& end_reshape_shape = pattern_map.at(end_reshape_target_shape);
        // n_all_experts dimension is cut off after ReduceSum
        const auto slice = std::make_shared<ov::op::v8::Slice>(
            end_reshape_shape,
            ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1}),
            ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {end_reshape_rank.get_length()}),
            ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1}),
            ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {0}));
        ov::copy_runtime_info(end_reshape_out.get_node_shared_ptr(),
                              {slice,
                               slice->get_input_node_shared_ptr(1),
                               slice->get_input_node_shared_ptr(2),
                               slice->get_input_node_shared_ptr(3),
                               slice->get_input_node_shared_ptr(4)});

        const auto reshape = std::make_shared<ov::op::v1::Reshape>(new_reduce_sum, slice, true);
        ov::replace_output_update_name(pattern_map.at(reduce_sum), reshape->output(0));
        // To avoid friendly name duplication
        reshape->set_friendly_name(reshape->get_friendly_name() + "_Reshape");
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(reduce_sum, matcher_name);
    this->register_matcher(m, callback);
}

ov::intel_cpu::MoE3GeMMFusion::MoE3GeMMFusion() {
    MATCHER_SCOPE(MoE3GeMMFusion);

    auto data_input = pattern::any_input(pattern::rank_equals(3) && pattern::has_static_rank());
    auto experts_input =
        pattern::wrap_type<ov::op::v1::Reshape>({data_input, pattern::any_input()}, pattern::rank_equals(2));
    auto tile =
        pattern::wrap_type<ov::op::v0::Tile>({experts_input, pattern::any_input()}, pattern::consumers_count(1));
    auto after_tile_reshape =
        pattern::wrap_type<ov::op::v1::Reshape>({tile, pattern::any_input()}, pattern::consumers_count(2));

    // First GEMM (activation gate)
    auto gate_matmul = pattern::wrap_type<ov::op::v0::MatMul>({after_tile_reshape, pattern::any_input()},
                                                              pattern::consumers_count(1),
                                                              {{"transpose_a", false}, {"transpose_b", true}});
    auto swish = pattern::wrap_type<ov::op::v4::Swish>({gate_matmul}, pattern::consumers_count(1));
    // Second GEMM (up_projection)
    auto up_matmul = pattern::wrap_type<ov::op::v0::MatMul>({after_tile_reshape, pattern::any_input()},
                                                            pattern::consumers_count(1),
                                                            {{"transpose_a", false}, {"transpose_b", true}});
    // Join: Multiply (SwiGLU)
    auto swiglu = pattern::wrap_type<ov::op::v1::Multiply>({swish, up_matmul}, pattern::consumers_count(1));

    // Third GEMM (down_projection)
    auto down_matmul = pattern::wrap_type<ov::op::v0::MatMul>({swiglu, pattern::any_input()},
                                                              pattern::consumers_count(1),
                                                              {{"transpose_a", false}, {"transpose_b", true}});
    auto end_reshape_target_shape = pattern::any_input();
    auto end_reshape =
        pattern::wrap_type<ov::op::v1::Reshape>({down_matmul, end_reshape_target_shape}, pattern::consumers_count(1));

    auto zero_constant = pattern::wrap_type<ov::op::v0::Constant>(pattern::value_matches("0"));
    auto broadcasted_const = pattern::wrap_type<ov::op::v3::Broadcast>({zero_constant, pattern::any_input()}) |
                             pattern::wrap_type<ov::op::v1::Broadcast, ov::op::v3::Broadcast>(
                                 {zero_constant, pattern::any_input(), pattern::any_input()});
    // Routing weights/mask
    auto router_topk_indices = pattern::any_input();
    auto chosen_experts = pattern::any_input();
    auto one_constant = pattern::wrap_type<ov::op::v0::Constant>(pattern::value_matches("1"));
    auto scatter_elements_update = pattern::wrap_type<ov::op::v3::ScatterElementsUpdate>(
                                       {broadcasted_const, router_topk_indices, chosen_experts, one_constant}) |
                                   pattern::wrap_type<ov::op::v12::ScatterElementsUpdate>(
                                       {broadcasted_const, router_topk_indices, chosen_experts, one_constant},
                                       {{"reduction", "none"}});

    auto router_transpose = pattern::wrap_type<ov::op::v1::Transpose>({scatter_elements_update, pattern::any_input()});
    auto router_reshape =
        pattern::wrap_type<ov::op::v1::Reshape>({router_transpose, pattern::any_input()},
                                                ov::pass::pattern::rank_equals(3) || ov::pass::pattern::rank_equals(4));
    auto unsqueeze_axis =
        pattern::wrap_type<ov::op::v0::Constant>(pattern::value_matches("3") || pattern::value_matches("-1"));
    auto unsqueeze_routing_weights =
        pattern::optional<ov::op::v0::Unsqueeze>({router_reshape, unsqueeze_axis}, ov::pass::pattern::rank_equals(4));

    auto mul3 =
        pattern::wrap_type<ov::op::v1::Multiply>({end_reshape, unsqueeze_routing_weights}, pattern::consumers_count(1));
    auto reduce_sum = pattern::wrap_type<ov::op::v1::ReduceSum>({mul3, pattern::any_input()},
                                                                pattern::consumers_count(1),
                                                                {{"keep_dims", false}});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto& experts_subgraph_input = pattern_map.at(experts_input);
        const auto& active_indices = pattern_map.at(router_topk_indices);

        const auto gate_mm_node = pattern_map.at(gate_matmul).get_node_shared_ptr();
        const auto up_mm_node = pattern_map.at(up_matmul).get_node_shared_ptr();
        const auto down_mm_node = pattern_map.at(down_matmul).get_node_shared_ptr();

        // BatchGatherMatmul A shape: [n_activated_experts, batch_size * seq_length, hidden_size]
        // Number of activated experts is always 1 for the first BatchGatherMatmul
        const auto unsqueeze = introduce_n_experts_dim(experts_subgraph_input);
        const auto gate_gathered_mm =
            std::make_shared<BatchGatherMatmul>(unsqueeze, gate_mm_node->input_value(1), active_indices);
        const auto up_gathered_mm =
            std::make_shared<BatchGatherMatmul>(unsqueeze, up_mm_node->input_value(1), active_indices);
        ov::replace_node_update_name(gate_mm_node, gate_gathered_mm);
        ov::replace_node_update_name(up_mm_node, up_gathered_mm);

        validate_nodes(pattern_map, {swish, swiglu});

        const auto down_gathered_mm =
            std::make_shared<BatchGatherMatmul>(pattern_map.at(swiglu), down_mm_node->input_value(1), active_indices);
        ov::copy_runtime_info(down_mm_node, down_gathered_mm);
        down_gathered_mm->set_friendly_name(down_mm_node->get_friendly_name());

        const auto& chosen_experts_input = pattern_map.at(chosen_experts);
        const auto router_transpose_node = pattern_map.at(router_transpose).get_node_shared_ptr();
        const auto new_router_transpose =
            router_transpose_node->clone_with_new_inputs({chosen_experts_input, router_transpose_node->input_value(1)});
        ov::copy_runtime_info(router_transpose_node, new_router_transpose);
        new_router_transpose->set_friendly_name(router_transpose_node->get_friendly_name());

        const auto router_unsqueeze_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {-1});
        const auto router_unsqueeze =
            std::make_shared<ov::op::v0::Unsqueeze>(new_router_transpose, router_unsqueeze_const);
        ov::copy_runtime_info(router_transpose_node, {router_unsqueeze_const, router_unsqueeze});

        const auto final_mul_node = pattern_map.at(mul3).get_node_shared_ptr();
        const auto new_final_mul =
            final_mul_node->clone_with_new_inputs({down_gathered_mm->output(0), router_unsqueeze->output(0)});
        ov::copy_runtime_info(final_mul_node, new_final_mul);
        new_final_mul->set_friendly_name(final_mul_node->get_friendly_name());
        const auto reduce_sum_node = pattern_map.at(reduce_sum).get_node_shared_ptr();
        const auto new_reduce_sum =
            reduce_sum_node->clone_with_new_inputs({new_final_mul->output(0), reduce_sum_node->input_value(1)});
        ov::copy_runtime_info(reduce_sum_node, new_reduce_sum);
        new_reduce_sum->set_friendly_name(reduce_sum_node->get_friendly_name());

        const auto& end_reshape_out = pattern_map.at(end_reshape);
        const auto end_reshape_rank = end_reshape_out.get_partial_shape().rank();
        const auto& end_reshape_shape = pattern_map.at(end_reshape_target_shape);
        // n_all_experts dimension is cut off after ReduceSum
        const auto slice = std::make_shared<ov::op::v8::Slice>(
            end_reshape_shape,
            ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1}),
            ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {end_reshape_rank.get_length()}),
            ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1}),
            ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {0}));
        ov::copy_runtime_info(end_reshape_out.get_node_shared_ptr(),
                              {slice,
                               slice->get_input_node_shared_ptr(1),
                               slice->get_input_node_shared_ptr(2),
                               slice->get_input_node_shared_ptr(3),
                               slice->get_input_node_shared_ptr(4)});

        const auto reshape = std::make_shared<ov::op::v1::Reshape>(new_reduce_sum, slice, true);
        ov::replace_output_update_name(pattern_map.at(reduce_sum), reshape->output(0));
        reshape->set_friendly_name(new_reduce_sum->get_friendly_name() + "_Reshape");
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(reduce_sum, matcher_name);
    this->register_matcher(m, callback);
}
