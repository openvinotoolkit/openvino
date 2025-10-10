// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fuse_moe_experts.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/op/one_hot.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/shape_of_base.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/block.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/decompression.hpp"
#include "transformations/utils/utils.hpp"
namespace ov {
namespace pass {

using namespace ov::pass;
using namespace ov::pass::pattern;

namespace {

bool is_slice_to_end(const std::shared_ptr<Node>& node) {
    auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
    if (!constant || !constant->get_element_type().is_integral_number()) {
        return false;
    }

    const auto values = constant->cast_vector<int64_t>();
    if (values.size() != 2 || values[0] != 1) {
        return false;
    }

    const auto end_value = values[1];
    if (end_value == -1) {
        return true;
    }

    if (constant->get_element_type() == element::i32) {
        return end_value == std::numeric_limits<int32_t>::max();
    }

    if (constant->get_element_type() == element::i64) {
        return end_value == std::numeric_limits<int64_t>::max();
    }

    return false;
}

struct expert_data {
    std::shared_ptr<ov::pass::pattern::op::Block> expert_block;
    std::shared_ptr<Node> gate_proj_weight;
    std::shared_ptr<Node> up_proj_weight;
    std::shared_ptr<Node> down_proj_weight;
    size_t expert_id;
    std::shared_ptr<Node> permute_node;
};

std::shared_ptr<pattern::op::Block> mlp3_no_bias_swiglu_block(
    const Output<Node>& permute_Transpose,    // Transpose -> OneHot -> TopK -> Softmax -> MatMul -> Hidden States
    const Output<Node>& unsqueeze_Unsqueeze,  // Unsqueeze -> Reshape -> Hidden States
    const Output<Node>&
        index_Split_out_1,             // Split -> Unsqueeze -> Divide -> TopK -> Softmax -> MatMul -> Hidden States
    const Output<Node>& index_Reshape  // Reshape -> Divide -> TopK -> Softmax -> MatMul -> Hidden States
) {
    auto index_add__ScatterElementsUpdate_2 = pattern::any_input();
    auto shape_const = wrap_type<ov::op::v0::Constant>();
    auto expert_id = wrap_type<ov::op::v0::Constant>();
    auto slice_end_const = wrap_type<ov::op::v0::Constant>();
    // Gather input 1 is expert IDx
    auto select_Gather_1 = wrap_type<ov::op::v8::Gather>(
        {permute_Transpose, expert_id, wrap_type<ov::op::v0::Constant>(pattern::value_matches("0"))},
        {{"batch_dims", 0}});
    auto squeeze_Squeeze_1 =
        wrap_type<ov::op::v0::Squeeze>({select_Gather_1, wrap_type<ov::op::v0::Constant>(pattern::value_matches("0"))});
    auto ListUnpack_NonZero_1 = wrap_type<ov::op::v3::NonZero>({squeeze_Squeeze_1}, {{"output_type", "i64"}});
    auto ListUnpack_Split_1 = wrap_type<ov::op::v1::Split>(
        {ListUnpack_NonZero_1, wrap_type<ov::op::v0::Constant>(pattern::value_matches("0"))},
        {{"num_splits", 2}});
    ListUnpack_Split_1->set_output_size(2);
    auto ListUnpack_Squeeze_0_1 = wrap_type<ov::op::v0::Squeeze>(
        {ListUnpack_Split_1->output(1), wrap_type<ov::op::v0::Constant>(pattern::value_matches("0"))});
    auto index_add__Convert_1 = wrap_type<ov::op::v0::Convert>({ListUnpack_Squeeze_0_1}, {{"destination_type", "i32"}});
    auto index_add__Reshape_1 = wrap_type<ov::op::v1::Reshape>(
        {index_add__Convert_1, wrap_type<ov::op::v0::Constant>(pattern::value_matches("-1, 1"))},
        {{"special_zero", false}});
    // Input 0 - any scatter or broadcast?
    auto index_add__Slice_1 =
        wrap_type<ov::op::v8::Slice>({index_add__ScatterElementsUpdate_2,
                                      wrap_type<ov::op::v0::Constant>(pattern::value_matches("0, 0")),
                                      slice_end_const,
                                      wrap_type<ov::op::v0::Constant>(pattern::value_matches("1, 1")),
                                      wrap_type<ov::op::v0::Constant>(pattern::value_matches("0, 1"))});
    auto index_add__ShapeOf_14 = wrap_type<ov::op::v3::ShapeOf>({index_add__Slice_1}, {{"output_type", "i32"}});
    auto index_add__Broadcast_16 =
        wrap_type<ov::op::v3::Broadcast>({index_add__Reshape_1, index_add__ShapeOf_14}, {{"mode", "bidirectional"}});
    auto index_Gather_2 = wrap_type<ov::op::v8::Gather>(
        {unsqueeze_Unsqueeze, index_add__Convert_1, wrap_type<ov::op::v0::Constant>(pattern::value_matches("1"))},
        {{"batch_dims", 0}});
    auto reshape_Reshape_1 = wrap_type<ov::op::v1::Reshape>({index_Gather_2, shape_const}, {{"special_zero", true}});
    auto gate_proj_weight = pattern::any_input(pattern::rank_equals(2));
    auto linear_MatMul_gate = wrap_type<ov::op::v0::MatMul>({reshape_Reshape_1, gate_proj_weight},
                                                            {{"transpose_a", false}, {"transpose_b", true}});
    auto silu_Swish = wrap_type<ov::op::v4::Swish>({linear_MatMul_gate});
    auto up_proj_weight = pattern::any_input(pattern::rank_equals(2));
    auto linear_MatMul_up = wrap_type<ov::op::v0::MatMul>({reshape_Reshape_1, up_proj_weight},
                                                          {{"transpose_a", false}, {"transpose_b", true}});
    auto mul_Multiply = wrap_type<ov::op::v1::Multiply>({silu_Swish, linear_MatMul_up}, {{"auto_broadcast", "numpy"}});
    auto down_proj_weight = pattern::any_input(pattern::rank_equals(2));
    auto linear_MatMul_down = wrap_type<ov::op::v0::MatMul>({mul_Multiply, down_proj_weight},
                                                            {{"transpose_a", false}, {"transpose_b", true}});
    auto ListUnpack_Squeeze_1 = wrap_type<ov::op::v0::Squeeze>(
        {ListUnpack_Split_1->output(0), wrap_type<ov::op::v0::Constant>(pattern::value_matches("0"))});
    auto index_Convert_4 = wrap_type<ov::op::v0::Convert>({ListUnpack_Squeeze_1}, {{"destination_type", "i32"}});
    auto index_Multiply_1 =
        wrap_type<ov::op::v1::Multiply>({index_add__Convert_1, index_Split_out_1}, {{"auto_broadcast", "numpy"}});
    auto index_Add_1 = wrap_type<ov::op::v1::Add>({index_Convert_4, index_Multiply_1}, {{"auto_broadcast", "numpy"}});
    auto index_Gather_3 = wrap_type<ov::op::v8::Gather>(
        {index_Reshape, index_Add_1, wrap_type<ov::op::v0::Constant>(pattern::value_matches("0"))},
        {{"batch_dims", 0}});
    auto index_Reshape_8_1 = wrap_type<ov::op::v1::Reshape>(
        {index_Gather_3, wrap_type<ov::op::v0::Constant>(pattern::value_matches("0, 1"))},
        {{"special_zero", true}});
    auto mul_Multiply_2 =
        wrap_type<ov::op::v1::Multiply>({linear_MatMul_down, index_Reshape_8_1}, {{"auto_broadcast", "numpy"}});
    auto index_add__Broadcast_17 =
        wrap_type<ov::op::v3::Broadcast>({mul_Multiply_2, index_add__ShapeOf_14}, {{"mode", "bidirectional"}});
    // Input 0 - any scatter or broadcast?
    auto index_add__ScatterElementsUpdate_5 =
        wrap_type<ov::op::v12::ScatterElementsUpdate>({index_add__ScatterElementsUpdate_2,
                                                       index_add__Broadcast_16,
                                                       index_add__Broadcast_17,
                                                       wrap_type<ov::op::v0::Constant>(pattern::value_matches("0"))},
                                                      {{"reduction", "sum"}, {"use_init_val", true}});
    auto block = std::make_shared<pattern::op::Block>(
        ov::OutputVector{permute_Transpose, unsqueeze_Unsqueeze, index_Split_out_1, index_Reshape},
        ov::OutputVector{index_add__ScatterElementsUpdate_5},
        "expert_block");
    REGISTER_ANCHORS(block,
                     expert_id,
                     gate_proj_weight,
                     up_proj_weight,
                     down_proj_weight,
                     permute_Transpose,
                     slice_end_const);
    return block;
}

}  // namespace

ov::pass::FuseMOEExperts::FuseMOEExperts() : MultiMatcher("FuseMOEExperts") {
    auto axis0 = wrap_type<ov::op::v0::Constant>(pattern::value_matches("0"));
    auto axis1 = wrap_type<ov::op::v0::Constant>(pattern::value_matches("1"));
    auto axis2 = wrap_type<ov::op::v0::Constant>(pattern::value_matches("2"));
    auto slice_start = wrap_type<ov::op::v0::Constant>(pattern::value_matches("0, 0"));
    auto slice_end = wrap_type<ov::op::v0::Constant>();
    auto slice_step = wrap_type<ov::op::v0::Constant>(pattern::value_matches("1, 1"));
    auto slice_axes = wrap_type<ov::op::v0::Constant>(pattern::value_matches("0, 1"));
    auto reshape_shape = wrap_type<ov::op::v0::Constant>(pattern::value_matches("-1, 1"));
    auto reduce_neg1 = wrap_type<ov::op::v0::Constant>(pattern::value_matches("-1"));
    auto transpose_perm = wrap_type<ov::op::v0::Constant>(pattern::value_matches("2, 1, 0"));
    auto one_hot_on = wrap_type<ov::op::v0::Constant>(pattern::value_matches("1"));
    auto one_hot_off = wrap_type<ov::op::v0::Constant>(pattern::value_matches("0"));

    auto view_Reshape = pattern::any_input();
    auto self_model_layers_1_mlp_gate_weight = pattern::any_input();
    auto shape_const = wrap_type<ov::op::v0::Constant>();
    auto expert_num = wrap_type<ov::op::v0::Constant>();
    auto num_topk = wrap_type<ov::op::v0::Constant>();
    // auto linear_MatMul = wrap_type<ov::op::v0::MatMul>({view_Reshape, self_model_layers_1_mlp_gate_weight},
    // {{"transpose_a", false}, {"transpose_b", true}});
    auto linear_MatMul = pattern::any_input();
    auto softmax_Softmax = wrap_type<ov::op::v8::Softmax>({linear_MatMul}, {{"axis", 1}});
    auto topk_TopK = wrap_type<ov::op::v11::TopK>(
        {softmax_Softmax, num_topk},
        {{"axis", -1}, {"mode", "max"}, {"sort", "value"}, {"index_element_type", "i64"}, {"stable", false}});
    topk_TopK->set_output_size(2);
    auto one_hot_OneHot =
        wrap_type<ov::op::v1::OneHot>({topk_TopK->output(1), expert_num, one_hot_on, one_hot_off}, {{"axis", 2}});
    auto permute_Transpose = wrap_type<ov::op::v1::Transpose>({one_hot_OneHot, transpose_perm});
    auto select_Gather = wrap_type<ov::op::v8::Gather>({permute_Transpose, axis0, axis0}, {{"batch_dims", 0}});
    auto squeeze_Squeeze = wrap_type<ov::op::v0::Squeeze>({select_Gather, axis0});
    auto ListUnpack_NonZero = wrap_type<ov::op::v3::NonZero>({squeeze_Squeeze}, {{"output_type", "i64"}});
    auto ListUnpack_Split = wrap_type<ov::op::v1::Split>({ListUnpack_NonZero, axis0}, {{"num_splits", 2}});
    ListUnpack_Split->set_output_size(2);
    auto ListUnpack_Squeeze_0 = wrap_type<ov::op::v0::Squeeze>({ListUnpack_Split->output(1), axis0});
    auto index_add__Convert = wrap_type<ov::op::v0::Convert>({ListUnpack_Squeeze_0}, {{"destination_type", "i32"}});
    auto index_add__Reshape =
        wrap_type<ov::op::v1::Reshape>({index_add__Convert, reshape_shape}, {{"special_zero", false}});
    auto index_add__Slice =
        wrap_type<ov::op::v8::Slice>({pattern::any_input(), slice_start, slice_end, slice_step, slice_axes});
    auto index_add__ShapeOf_6 = wrap_type<ov::op::v3::ShapeOf>({index_add__Slice}, {{"output_type", "i32"}});
    auto index_add__Broadcast_7 =
        wrap_type<ov::op::v3::Broadcast>({index_add__Reshape, index_add__ShapeOf_6}, {{"mode", "bidirectional"}});
    auto unsqueeze_Unsqueeze = wrap_type<ov::op::v0::Unsqueeze>({view_Reshape, axis0});
    auto index_Gather =
        wrap_type<ov::op::v8::Gather>({unsqueeze_Unsqueeze, index_add__Convert, axis1}, {{"batch_dims", 0}});
    auto reshape_Reshape = wrap_type<ov::op::v1::Reshape>({index_Gather, shape_const}, {{"special_zero", true}});
    auto sum_ReduceSum = wrap_type<ov::op::v1::ReduceSum>({topk_TopK->output(0), reduce_neg1}, {{"keep_dims", true}});
    auto div__Divide = wrap_type<ov::op::v1::Divide>({topk_TopK->output(0), sum_ReduceSum},
                                                     {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});
    auto unsqueeze_Unsqueeze_1 = wrap_type<ov::op::v0::Unsqueeze>({div__Divide, axis2});
    auto index_ShapeOf_1 = wrap_type<ov::op::v3::ShapeOf>({unsqueeze_Unsqueeze_1}, {{"output_type", "i32"}});
    auto index_Split = wrap_type<ov::op::v1::Split>({index_ShapeOf_1, axis0}, {{"num_splits", 3}});
    index_Split->set_output_size(3);
    auto index_Reshape =
        wrap_type<ov::op::v1::Reshape>({unsqueeze_Unsqueeze_1, pattern::any_input()}, {{"special_zero", true}});

    auto expert_scatter =
        mlp3_no_bias_swiglu_block(permute_Transpose, unsqueeze_Unsqueeze, index_Split->output(1), index_Reshape);

    // Pattern to match the operations after all experts - this captures the final scatter operation
    // that accumulates results from all experts, followed by reshape and residual add
    // auto final_scatter = wrap_type<ov::op::v12::ScatterElementsUpdate>();
    auto original_shape = pattern::any_input();
    auto last_reshape = wrap_type<ov::op::v1::Reshape>({expert_scatter, original_shape}, {{"special_zero", false}});
    auto residual_input = pattern::any_input();
    auto last_add = wrap_type<ov::op::v1::Add>({residual_input, last_reshape}, {{"auto_broadcast", "numpy"}});

    auto callback = [=](const std::unordered_map<std::shared_ptr<Node>, std::vector<PatternValueMap>>& matches) {
        auto num_expert_scatter = matches.at(expert_scatter).size();
        auto num_last_add = matches.at(last_add).size();

        std::cout << "Found " << num_last_add << " MoE layers with " << num_expert_scatter << " total experts"
                  << std::endl;

        // Collect all experts first
        std::vector<expert_data> all_experts;
        all_experts.reserve(matches.at(expert_scatter).size());
        for (const auto& pm : matches.at(expert_scatter)) {
            auto block_node =
                ov::as_type_ptr<ov::pass::pattern::op::Block>(pm.at(expert_scatter).get_node_shared_ptr());
            auto slice_end_anchor = expert_scatter->get_anchor("slice_end_const", pm);
            if (!slice_end_anchor.has_value() || !is_slice_to_end(slice_end_anchor.value().get_node_shared_ptr())) {
                return false;
            }
            auto gate_proj_node = expert_scatter->get_anchor("gate_proj_weight", pm).value().get_node_shared_ptr();
            auto up_proj_node = expert_scatter->get_anchor("up_proj_weight", pm).value().get_node_shared_ptr();
            auto down_proj_node = expert_scatter->get_anchor("down_proj_weight", pm).value().get_node_shared_ptr();
            auto expert_id_node = expert_scatter->get_anchor("expert_id", pm).value().get_node_shared_ptr();
            auto expert_id_const = ov::as_type_ptr<ov::op::v0::Constant>(expert_id_node);
            auto permute_node = expert_scatter->get_anchor("permute_Transpose", pm).value().get_node_shared_ptr();
            auto expert_id_values = expert_id_const->cast_vector<int64_t>();
            all_experts.push_back({block_node,
                                   gate_proj_node,
                                   up_proj_node,
                                   down_proj_node,
                                   static_cast<std::size_t>(expert_id_values.at(0)),
                                   permute_node});
        }

        std::unordered_map<const ov::Node*, std::vector<expert_data>> experts_by_permute;
        for (const auto& expert : all_experts) {
            experts_by_permute[expert.permute_node.get()].push_back(expert);
        }

        // Process each MoE layer separately
        for (size_t moe_idx = 0; moe_idx < num_last_add; moe_idx++) {
            // Get the pattern match for this specific MoE layer
            const auto& last_add_match = matches.at(last_add)[moe_idx];
            std::shared_ptr<Node> slice_end_node = nullptr;
            auto slice_end_it = last_add_match.find(slice_end);
            if (slice_end_it != last_add_match.end()) {
                slice_end_node = slice_end_it->second.get_node_shared_ptr();
            } else if (auto slice_end_anchor = expert_scatter->get_anchor("slice_end_const", last_add_match)) {
                slice_end_node = slice_end_anchor.value().get_node_shared_ptr();
            }
            if (slice_end_node && !is_slice_to_end(slice_end_node)) {
                continue;
            }
            auto last_add_node = last_add_match.at(last_add).get_node_shared_ptr();
            auto last_reshape_node = last_add_match.at(last_reshape).get_node_shared_ptr();

            auto permute_node =
                expert_scatter->get_anchor("permute_Transpose", last_add_match).value().get_node_shared_ptr();
            auto experts_it = experts_by_permute.find(permute_node.get());
            if (experts_it == experts_by_permute.end() || experts_it->second.empty()) {
                continue;
            }

            auto& experts = experts_it->second;
            std::sort(experts.begin(), experts.end(), [](const expert_data& a, const expert_data& b) {
                return a.expert_id < b.expert_id;
            });

            std::cout << "Processing MoE layer " << moe_idx + 1 << "/" << num_last_add << " with " << experts.size()
                      << " experts" << std::endl;

            // Concatenate weights incrementally to reduce memory usage
            auto const_0 = ov::op::v0::Constant::create(element::i64, Shape{1}, {0});

            // Helper function to detect decompression Convert pattern and get original weight
            auto get_original_weight_and_decompress_info =
                [](const std::shared_ptr<Node>& weight) -> std::pair<std::shared_ptr<Node>, bool> {
                // Check if the current weight is a Convert op with decompression attribute
                if (auto convert_op = ov::as_type_ptr<ov::op::v0::Convert>(weight)) {
                    if (ov::is_decompression(convert_op)) {
                        // Get the original Constant input to this Convert
                        auto convert_input = convert_op->get_input_node_shared_ptr(0);
                        if (ov::as_type_ptr<ov::op::v0::Constant>(convert_input)) {
                            return {convert_input, true};  // Original weight, needs decompression
                        }
                    }
                }
                return {weight, false};  // No decompression needed
            };

            auto build_fused_weight =
                [&](const std::function<std::shared_ptr<Node>(const expert_data&)>& getter) -> Output<Node> {
                OutputVector inputs;
                bool needs_decompress = false;
                auto target_type = getter(experts.front())->get_output_element_type(0);

                for (const auto& expert : experts) {
                    auto weight = getter(expert);
                    auto [original_weight, decompress] = get_original_weight_and_decompress_info(weight);

                    if (decompress) {
                        needs_decompress = true;
                        target_type = weight->get_output_element_type(0);
                    }

                    inputs.emplace_back(ov::op::util::make_try_fold<ov::op::v0::Unsqueeze>(original_weight, const_0));
                }

                auto fused = ov::op::util::make_try_fold<ov::op::v0::Concat>(inputs, 0);

                if (needs_decompress) {
                    auto convert = std::make_shared<ov::op::v0::Convert>(fused, target_type);
                    ov::mark_as_decompression(convert);
                    return convert->output(0);
                }

                return fused;
            };

            auto fused_gate_weights = build_fused_weight([](const expert_data& expert) {
                return expert.gate_proj_weight;
            });
            auto fused_up_weights = build_fused_weight([](const expert_data& expert) {
                return expert.up_proj_weight;
            });
            auto fused_down_weights = build_fused_weight([](const expert_data& expert) {
                return expert.down_proj_weight;
            });

            // Debug output for the collected operations
            // Get input from the pattern - this should be the post-attention layernorm output
            auto view_reshape_node = last_add_match.at(view_Reshape).get_node_shared_ptr();
            auto residual_input_node = last_add_match.at(residual_input).get_node_shared_ptr();

            // Extract dynamic values from the collected experts
            const size_t num_experts = experts.size();

            auto axis0_scalar = ov::op::v0::Constant::create(element::i64, Shape{}, {0});
            auto axis1_scalar = ov::op::v0::Constant::create(element::i64, Shape{}, {1});
            auto axis0_vector = ov::op::v0::Constant::create(element::i64, Shape{1}, {0});
            auto axis1_vector = ov::op::v0::Constant::create(element::i64, Shape{1}, {1});
            auto axis_minus_one_vector = ov::op::v0::Constant::create(element::i64, Shape{1}, {-1});
            auto transpose_perm = ov::op::v0::Constant::create(element::i64, Shape{2}, {1, 0});

            // Hidden dimension from gate weight
            auto gate_weight_shape_op =
                std::make_shared<ov::op::v3::ShapeOf>(experts[0].gate_proj_weight, element::i64);
            auto hidden_dim_scalar =
                std::make_shared<ov::op::v8::Gather>(gate_weight_shape_op, axis1_scalar, axis0_scalar);
            auto hidden_dim = std::make_shared<ov::op::v0::Unsqueeze>(hidden_dim_scalar, axis0_vector);

            // Batch dimension derived from router TopK indices shape
            auto topk = last_add_match.at(topk_TopK).get_node_shared_ptr();
            auto topk_indices_output = topk->output(1);
            auto topk_indices_shape = std::make_shared<ov::op::v3::ShapeOf>(topk_indices_output, element::i64);
            auto batch_dim_scalar =
                std::make_shared<ov::op::v8::Gather>(topk_indices_shape, axis0_scalar, axis0_scalar);
            auto batch_dim = std::make_shared<ov::op::v0::Unsqueeze>(batch_dim_scalar, axis0_vector);

            auto num_experts_const =
                ov::op::v0::Constant::create(element::i64, Shape{}, {static_cast<int64_t>(num_experts)});
            auto num_experts_dim = std::make_shared<ov::op::v0::Unsqueeze>(num_experts_const, axis0_vector);

            // Create the fused MoE computation following the 3-GEMM pattern
            auto tile_shape =
                ov::op::v0::Constant::create(element::i64,
                                             Shape{2},
                                             {static_cast<int64_t>(num_experts), static_cast<int64_t>(1)});

            auto view_reshape_shape =
                std::make_shared<ov::op::v0::Concat>(OutputVector{axis_minus_one_vector, hidden_dim}, 0);
            auto view_reshape = std::make_shared<ov::op::v1::Reshape>(view_reshape_node, view_reshape_shape, false);
            auto repeated_input = std::make_shared<ov::op::v0::Tile>(view_reshape, tile_shape);

            auto batched_shape =
                std::make_shared<ov::op::v0::Concat>(OutputVector{num_experts_dim, batch_dim, hidden_dim}, 0);
            auto batched_input = std::make_shared<ov::op::v1::Reshape>(repeated_input, batched_shape, false);

            auto gate_bmm = std::make_shared<ov::op::v0::MatMul>(batched_input, fused_gate_weights, false, true);
            auto gate_swish = std::make_shared<ov::op::v4::Swish>(gate_bmm);

            auto up_bmm = std::make_shared<ov::op::v0::MatMul>(batched_input, fused_up_weights, false, true);
            auto swiglu_mul = std::make_shared<ov::op::v1::Multiply>(gate_swish, up_bmm);

            auto down_bmm = std::make_shared<ov::op::v0::MatMul>(swiglu_mul, fused_down_weights, false, true);

            auto minus_one_vec = ov::op::v0::Constant::create(element::i64, Shape{1}, {-1});
            auto expert_output_shape = std::make_shared<ov::op::v0::Concat>(
                OutputVector{num_experts_dim, batch_dim, minus_one_vec, hidden_dim},
                0);
            auto expert_outputs = std::make_shared<ov::op::v1::Reshape>(down_bmm, expert_output_shape, false);

            // Routing tensor built via ScatterElementsUpdate to match MOE fusion pattern
            auto topk_values = topk->output(0);
            auto sum_reduce =
                std::make_shared<ov::op::v1::ReduceSum>(topk_values,
                                                        ov::op::v0::Constant::create(element::i64, Shape{1}, {-1}),
                                                        true);
            auto normalized_topk = std::make_shared<ov::op::v1::Divide>(topk_values, sum_reduce);

            auto scatter_shape = std::make_shared<ov::op::v0::Concat>(OutputVector{batch_dim, num_experts_dim}, 0);
            auto zeros_scalar = ov::op::v0::Constant::create(normalized_topk->get_element_type(), Shape{}, {0});
            auto zeros_tensor = std::make_shared<ov::op::v3::Broadcast>(zeros_scalar, scatter_shape);

            auto scatter = std::make_shared<ov::op::v12::ScatterElementsUpdate>(zeros_tensor,
                                                                                topk_indices_output,
                                                                                normalized_topk,
                                                                                axis1_vector);
            scatter->set_reduction(ov::op::v12::ScatterElementsUpdate::Reduction::SUM);
            scatter->set_use_init_val(true);
            auto router_transpose = std::make_shared<ov::op::v1::Transpose>(scatter, transpose_perm);
            auto router_shape =
                std::make_shared<ov::op::v0::Concat>(OutputVector{num_experts_dim, batch_dim, minus_one_vec}, 0);
            auto router_reshape = std::make_shared<ov::op::v1::Reshape>(router_transpose, router_shape, true);
            auto routing_weights = std::make_shared<ov::op::v0::Unsqueeze>(router_reshape, axis_minus_one_vector);

            auto weighted_outputs = std::make_shared<ov::op::v1::Multiply>(expert_outputs, routing_weights);
            auto final_output = std::make_shared<ov::op::v1::ReduceSum>(weighted_outputs, axis0_vector, false);

            // Reshape back to original shape and add residual connection
            auto target_shape = std::make_shared<ov::op::v3::ShapeOf>(view_reshape_node, element::i64);
            auto final_reshape = std::make_shared<ov::op::v1::Reshape>(final_output, target_shape, false);
            auto final_add = std::make_shared<ov::op::v1::Add>(residual_input_node, final_reshape);

            // Replace the last add node with our new computation
            ov::replace_node(last_add_node, final_add);
            std::cout << "MoE layer " << moe_idx + 1 << "/" << num_last_add << " fused successfully." << std::endl;
        }

        return true;
    };

    register_patterns(std::vector<std::shared_ptr<Node>>{expert_scatter, last_add}, callback, true);
}

bool ov::pass::FuseMOE::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(FuseMOE);
    ov::pass::Manager manager(get_pass_config(), "FuseMOE");

    // Use the unified FuseMOE transformation
    manager.register_pass<ov::pass::FuseMOEExperts>();
    manager.run_passes(model);
    return false;
}

}  // namespace pass
}  // namespace ov
