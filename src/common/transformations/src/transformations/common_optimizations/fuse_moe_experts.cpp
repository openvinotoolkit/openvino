// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fuse_moe_experts.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
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
#include "openvino/op/subtract.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/op/util/shape_of_base.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/block.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/runtime/tensor.hpp"
#include "transformations/rt_info/decompression.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::pass {

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v3 = ov::op::v3;
namespace v4 = ov::op::v4;
namespace v8 = ov::op::v8;
namespace v12 = ov::op::v12;
namespace op_util = ov::op::util;

namespace {

// Check if a Slice operation's end parameter matches one found in model.
// The pattern expects a 2-element constant [1, MAX] where.
// This pattern matches Slice operations like: Slice(input, [0,0], [1,INT_MAX], [1,1], [0,1])
// index_add__Slice_2 = makeOP<opset8::Slice>({index_add__ScatterElementsUpdate_5, {0,0}, {1,INT_MAX}, {1,1}, {0,1}});
bool is_slice_to_end(const std::shared_ptr<Node>& node) {
    auto constant = ov::as_type_ptr<v0::Constant>(node);
    if (!constant || !constant->get_element_type().is_integral_number()) {
        return false;
    }

    const auto values = constant->cast_vector<int64_t>();
    // Expect exactly 2 values: [1, MAX] where first dimension ends at 1
    if (values.size() != 2 || values[0] != 1) {
        return false;
    }

    // Second dimension should use maximum value to indicate slicing to the end
    const auto end_value = values[1];
    if (constant->get_element_type() == element::i32) {
        return end_value == std::numeric_limits<int32_t>::max();
    }

    if (constant->get_element_type() == element::i64) {
        return end_value == std::numeric_limits<int64_t>::max();
    }

    return false;
}

// Pattern matching for individual expert computation block with SwiGLU activation
std::shared_ptr<ov::pass::pattern::op::Block> mlp3_no_bias_swiglu_block(
    const Output<Node>& permute_Transpose,    // Transpose -> OneHot -> TopK -> Softmax -> MatMul -> Hidden States
    const Output<Node>& unsqueeze_Unsqueeze,  // Unsqueeze -> Reshape -> Hidden States
    const Output<Node>&
        index_Split_out_1,             // Split -> Unsqueeze -> Divide -> TopK -> Softmax -> MatMul -> Hidden States
    const Output<Node>& index_Reshape  // Reshape -> Divide -> TopK -> Softmax -> MatMul -> Hidden States
) {
    auto index_add__ScatterElementsUpdate_2 = pattern::any_input();
    auto shape_const = pattern::wrap_type<v0::Constant>();
    auto expert_id = pattern::wrap_type<v0::Constant>();
    auto slice_end_const = pattern::wrap_type<v0::Constant>();
    auto select_Gather_1 = pattern::wrap_type<v8::Gather>(
        {permute_Transpose, expert_id, pattern::wrap_type<v0::Constant>(pattern::value_matches("0"))},
        {{"batch_dims", 0}});
    auto squeeze_Squeeze_1 = pattern::wrap_type<v0::Squeeze>(
        {select_Gather_1, pattern::wrap_type<v0::Constant>(pattern::value_matches("0"))});
    // NonZero output_type relaxed to accept both i32 and i64
    auto ListUnpack_NonZero_1 = pattern::wrap_type<v3::NonZero>({squeeze_Squeeze_1});
    auto ListUnpack_Split_1 = pattern::wrap_type_strict_index<v1::Split>(
        {ListUnpack_NonZero_1, pattern::wrap_type<v0::Constant>(pattern::value_matches("0"))},
        {{"num_splits", 2}});
    auto ListUnpack_Squeeze_0_1 = pattern::wrap_type<v0::Squeeze>(
        {ListUnpack_Split_1->output(1), pattern::wrap_type<v0::Constant>(pattern::value_matches("0"))});
    // Convert is optional - pattern matches both with and without type conversion
    auto index_add__Convert_1 = pattern::wrap_type<v0::Convert>({ListUnpack_Squeeze_0_1}) | ListUnpack_Squeeze_0_1;
    auto index_add__Reshape_1 = pattern::wrap_type<v1::Reshape>(
        {index_add__Convert_1, pattern::wrap_type<v0::Constant>(pattern::value_matches("-1, 1"))});
    auto index_add__Slice_1 =
        pattern::wrap_type<v8::Slice>({index_add__ScatterElementsUpdate_2,
                                       pattern::wrap_type<v0::Constant>(pattern::value_matches("0, 0")),
                                       slice_end_const,
                                       pattern::wrap_type<v0::Constant>(pattern::value_matches("1, 1")),
                                       pattern::wrap_type<v0::Constant>(pattern::value_matches("0, 1"))});
    auto index_add__ShapeOf_14 = pattern::wrap_type<v3::ShapeOf>({index_add__Slice_1}, {{"output_type", "i32"}});
    auto index_add__Broadcast_16 =
        pattern::wrap_type<v3::Broadcast>({index_add__Reshape_1, index_add__ShapeOf_14}, {{"mode", "bidirectional"}});
    auto unsqueeze_Unsqueeze_reshape = pattern::wrap_type<v1::Reshape>({unsqueeze_Unsqueeze, pattern::any_input()});
    auto index_Gather_2 =
        pattern::wrap_type<v8::Gather>({unsqueeze_Unsqueeze_reshape,
                                        index_add__Convert_1,
                                        pattern::wrap_type<v0::Constant>(pattern::value_matches("0"))},
                                       {{"batch_dims", 0}});
    auto reshape_Reshape_1_0 = pattern::wrap_type<v1::Reshape>({index_Gather_2, pattern::any_input()});
    auto reshape_Reshape_1_1 = pattern::wrap_type<v1::Reshape>({reshape_Reshape_1_0, pattern::any_input()});
    auto reshape_Reshape_1_2 = pattern::wrap_type<v1::Reshape>({reshape_Reshape_1_1, pattern::any_input()});
    auto reshape_Reshape_1 = pattern::wrap_type<v1::Reshape>({reshape_Reshape_1_2, shape_const});
    auto gate_proj_weight = pattern::any_input(pattern::rank_equals(2));
    auto linear_MatMul_gate = pattern::wrap_type<v0::MatMul>({reshape_Reshape_1, gate_proj_weight},
                                                             {{"transpose_a", false}, {"transpose_b", true}});
    auto silu_Swish = pattern::wrap_type<v4::Swish>({linear_MatMul_gate});
    auto up_proj_weight = pattern::any_input(pattern::rank_equals(2));
    auto linear_MatMul_up = pattern::wrap_type<v0::MatMul>({reshape_Reshape_1, up_proj_weight},
                                                           {{"transpose_a", false}, {"transpose_b", true}});
    auto mul_Multiply = pattern::wrap_type<v1::Multiply>({silu_Swish, linear_MatMul_up}, {{"auto_broadcast", "numpy"}});
    auto down_proj_weight = pattern::any_input(pattern::rank_equals(2));
    auto linear_MatMul_down = pattern::wrap_type<v0::MatMul>({mul_Multiply, down_proj_weight},
                                                             {{"transpose_a", false}, {"transpose_b", true}});
    auto ListUnpack_Squeeze_1 = pattern::wrap_type<v0::Squeeze>(
        {ListUnpack_Split_1->output(0), pattern::wrap_type<v0::Constant>(pattern::value_matches("0"))});
    // Convert is optional - pattern matches both with and without type conversion
    auto index_Convert_4 = pattern::wrap_type<v0::Convert>({ListUnpack_Squeeze_1}) | ListUnpack_Squeeze_1;
    auto index_Multiply_1 =
        pattern::wrap_type<v1::Multiply>({index_add__Convert_1, index_Split_out_1}, {{"auto_broadcast", "numpy"}});
    auto index_Add_1 = pattern::wrap_type<v1::Add>({index_Convert_4, index_Multiply_1}, {{"auto_broadcast", "numpy"}});
    auto index_Gather_3 = pattern::wrap_type<v8::Gather>(
        {index_Reshape, index_Add_1, pattern::wrap_type<v0::Constant>(pattern::value_matches("0"))},
        {{"batch_dims", 0}});
    auto index_Reshape_8_1 = pattern::wrap_type<v1::Reshape>(
        {index_Gather_3, pattern::wrap_type<v0::Constant>(pattern::value_matches("0, 1"))});
    auto mul_Multiply_2 =
        pattern::wrap_type<v1::Multiply>({linear_MatMul_down, index_Reshape_8_1}, {{"auto_broadcast", "numpy"}});
    auto index_add__Broadcast_17 =
        pattern::wrap_type<v3::Broadcast>({mul_Multiply_2, index_add__ShapeOf_14}, {{"mode", "bidirectional"}});
    auto index_add__ScatterElementsUpdate_5 =
        pattern::wrap_type<v12::ScatterElementsUpdate>({index_add__ScatterElementsUpdate_2,
                                                        index_add__Broadcast_16,
                                                        index_add__Broadcast_17,
                                                        pattern::wrap_type<v0::Constant>(pattern::value_matches("0"))},
                                                       {{"reduction", "sum"}, {"use_init_val", true}});
    auto block = std::make_shared<ov::pass::pattern::op::Block>(
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

// Helper function to create commonly used axis constants
struct AxisPatterns {
    std::shared_ptr<Node> axis0 = pattern::wrap_type<v0::Constant>(pattern::value_matches("0"));
    std::shared_ptr<Node> axis1 = pattern::wrap_type<v0::Constant>(pattern::value_matches("1"));
    std::shared_ptr<Node> axis2 = pattern::wrap_type<v0::Constant>(pattern::value_matches("2"));
};

// Helper function to extract a dimension from a tensor shape
// Returns ShapeOf -> Gather(axis_index) -> Unsqueeze as a dimension vector
std::shared_ptr<Node> extract_shape_dim(const Output<Node>& input, size_t axis_index) {
    auto shape_of = std::make_shared<v3::ShapeOf>(input, element::i64);
    auto axis_scalar = v0::Constant::create(element::i64, Shape{}, {static_cast<int64_t>(axis_index)});
    auto axis0_scalar = v0::Constant::create(element::i64, Shape{}, {0});
    auto dim_scalar = std::make_shared<v8::Gather>(shape_of, axis_scalar, axis0_scalar);
    auto axis0_vector = v0::Constant::create(element::i64, Shape{1}, {0});
    return std::make_shared<v0::Unsqueeze>(dim_scalar, axis0_vector);
}

// Helper function to create router pattern (Softmax -> TopK -> OneHot -> Transpose)
// This pattern identifies the expert selection logic from router logits
std::pair<std::shared_ptr<Node>, std::shared_ptr<Node>> create_router_pattern() {
    auto linear_MatMul = pattern::any_input();
    auto expert_num = pattern::wrap_type<v0::Constant>();
    auto num_topk = pattern::wrap_type<v0::Constant>();
    auto one_hot_on = pattern::wrap_type<v0::Constant>(pattern::value_matches("1"));
    auto one_hot_off = pattern::wrap_type<v0::Constant>(pattern::value_matches("0"));
    auto transpose_perm = pattern::wrap_type<v0::Constant>(pattern::value_matches("2, 1, 0"));

    auto softmax = pattern::wrap_type<v8::Softmax>({linear_MatMul}, {{"axis", 1}});
    auto topk = pattern::wrap_type_strict_index<ov::op::v11::TopK>(
        {softmax, num_topk},
        {{"axis", -1}, {"mode", "max"}, {"sort", "value"}, {"index_element_type", "i64"}, {"stable", false}});
    auto one_hot =
        pattern::wrap_type<v1::OneHot>({topk->output(1), expert_num, one_hot_on, one_hot_off}, {{"axis", 2}});
    auto permute = pattern::wrap_type<v1::Transpose>({one_hot, transpose_perm});

    return {topk, permute};
}

// Helper function to create expert indexing pattern (NonZero -> Split -> Convert)
// This extracts indices of tokens assigned to each expert
std::shared_ptr<Node> create_expert_indexing_pattern(const std::shared_ptr<Node>& permute_transpose,
                                                     const AxisPatterns& axes) {
    auto select_gather =
        pattern::wrap_type<v8::Gather>({permute_transpose, axes.axis0, axes.axis0}, {{"batch_dims", 0}});
    auto squeeze = pattern::wrap_type<v0::Squeeze>({select_gather, axes.axis0});
    // NonZero output_type relaxed to accept both i32 and i64
    auto non_zero = pattern::wrap_type<v3::NonZero>({squeeze});
    auto split = pattern::wrap_type_strict_index<v1::Split>({non_zero, axes.axis0}, {{"num_splits", 2}});
    auto squeeze_indices = pattern::wrap_type<v0::Squeeze>({split->output(1), axes.axis0});
    // Convert is optional - pattern matches both with and without type conversion
    auto convert = pattern::wrap_type<v0::Convert>({squeeze_indices}) | squeeze_indices;
    return convert;
}

// Helper function to create routing weights pattern
// This normalizes TopK values and prepares them for weighting expert outputs
std::tuple<std::shared_ptr<Node>, std::shared_ptr<Node>> create_routing_weights_pattern(
    const std::shared_ptr<Node>& topk,
    const AxisPatterns& axes) {
    auto reduce_neg1 = pattern::wrap_type<v0::Constant>(pattern::value_matches("-1"));
    auto sum_reduce = pattern::wrap_type<v1::ReduceSum>({topk->output(0), reduce_neg1}, {{"keep_dims", true}});
    auto normalized = pattern::wrap_type<v1::Divide>({topk->output(0), sum_reduce},
                                                     {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});
    auto unsqueeze = pattern::wrap_type<v0::Unsqueeze>({normalized, axes.axis2});
    auto shape_of = pattern::wrap_type<v3::ShapeOf>({unsqueeze}, {{"output_type", "i32"}});
    auto split = pattern::wrap_type_strict_index<v1::Split>({shape_of, axes.axis0}, {{"num_splits", 3}});
    auto reshape = pattern::wrap_type<v1::Reshape>({unsqueeze, pattern::any_input()});

    return {split, reshape};
}

}  // namespace

ov::pass::FuseMOEExperts::FuseMOEExperts() : MultiMatcher("FuseMOEExperts") {
    // Create axis patterns
    AxisPatterns axes;

    // Create router pattern: Softmax -> TopK -> OneHot -> Transpose
    auto router_result = create_router_pattern();
    auto topk_TopK = router_result.first;
    auto permute_Transpose = router_result.second;

    // Create expert indexing pattern for the first expert (used in outer loop)
    auto index_add__Convert = create_expert_indexing_pattern(permute_Transpose, axes);

    // Prepare additional patterns for expert computation
    auto view_Reshape = pattern::any_input();
    auto slice_end = pattern::wrap_type<v0::Constant>();  // Used in callback for slice validation

    // Pattern for hidden states preparation
    auto unsqueeze_Unsqueeze = pattern::wrap_type<v0::Unsqueeze>({view_Reshape, axes.axis0});

    // Create routing weights pattern
    auto routing_result = create_routing_weights_pattern(topk_TopK, axes);
    auto index_Split = std::get<0>(routing_result);
    auto index_Reshape = std::get<1>(routing_result);

    // Create expert computation pattern (3-GEMM SwiGLU block)
    auto expert_scatter =
        mlp3_no_bias_swiglu_block(permute_Transpose, unsqueeze_Unsqueeze, index_Split->output(1), index_Reshape);

    // Match the final reshape and residual add after all expert computations
    auto original_shape = pattern::any_input();
    auto last_reshape = pattern::wrap_type<v1::Reshape>({expert_scatter, original_shape});
    auto residual_input = pattern::any_input();
    auto last_add = pattern::wrap_type<v1::Add>({residual_input, last_reshape}, {{"auto_broadcast", "numpy"}});

    auto callback = [=](const std::unordered_map<std::shared_ptr<Node>, std::vector<pattern::PatternValueMap>>&
                            matches) {
        auto num_last_add = matches.at(last_add).size();

        // Collect expert data from all matched patterns
        std::vector<expert_data> all_experts;
        all_experts.reserve(matches.at(expert_scatter).size());
        for (const auto& pm : matches.at(expert_scatter)) {
            auto slice_end_anchor = expert_scatter->get_anchor("slice_end_const", pm);
            if (!slice_end_anchor.has_value() || !is_slice_to_end(slice_end_anchor.value().get_node_shared_ptr())) {
                return false;
            }
            auto gate_proj_node = expert_scatter->get_anchor("gate_proj_weight", pm).value().get_node_shared_ptr();
            auto up_proj_node = expert_scatter->get_anchor("up_proj_weight", pm).value().get_node_shared_ptr();
            auto down_proj_node = expert_scatter->get_anchor("down_proj_weight", pm).value().get_node_shared_ptr();
            auto expert_id_node = expert_scatter->get_anchor("expert_id", pm).value().get_node_shared_ptr();
            auto expert_id_const = ov::as_type_ptr<v0::Constant>(expert_id_node);
            auto permute_node = expert_scatter->get_anchor("permute_Transpose", pm).value().get_node_shared_ptr();
            auto expert_id_values = expert_id_const->cast_vector<int64_t>();
            all_experts.push_back({gate_proj_node,
                                   up_proj_node,
                                   down_proj_node,
                                   static_cast<std::size_t>(expert_id_values.at(0)),
                                   permute_node});
        }

        // Group experts by their permute node to identify which MoE layer they belong to
        std::unordered_map<const ov::Node*, std::vector<expert_data>> experts_by_permute;
        for (const auto& expert : all_experts) {
            experts_by_permute[expert.permute_node.get()].push_back(expert);
        }

        // Create shared constants (used across all MoE layers)
        auto const_0 = v0::Constant::create(element::i64, Shape{1}, {0});
        auto const_1 = v0::Constant::create(element::i64, Shape{1}, {1});
        auto const_minus_1 = v0::Constant::create(element::i64, Shape{1}, {-1});
        auto transpose_perm = v0::Constant::create(element::i64, Shape{2}, {1, 0});

        // Process each MoE layer separately
        for (size_t moe_idx = 0; moe_idx < num_last_add; moe_idx++) {
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

            // Build fused weight tensors by concatenating individual expert weights along expert dimension

            // Detect if weights have decompression pattern and extract original compressed weights
            auto get_original_weight_and_decompress_info =
                [](const std::shared_ptr<Node>& weight) -> std::pair<std::shared_ptr<Node>, bool> {
                if (auto convert_op = ov::as_type_ptr<v0::Convert>(weight)) {
                    if (ov::is_decompression(convert_op)) {
                        auto convert_input = convert_op->get_input_node_shared_ptr(0);
                        if (ov::as_type_ptr<v0::Constant>(convert_input)) {
                            return {convert_input, true};
                        }
                    }
                }
                return {weight, false};
            };
            // Build concatenated weight tensor for all experts, preserving decompression if present
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

                    inputs.emplace_back(op_util::make_try_fold<v0::Unsqueeze>(original_weight, const_0));
                }

                auto fused = std::make_shared<v0::Concat>(inputs, 0);
                // Tag the fused expert weight so it can be identified directly, e.g. by tests,
                // instead of being guessed from its rank and shape.
                fused->set_friendly_name(fused->get_friendly_name() + "/FusedMOEWeights");
                if (std::all_of(inputs.begin(), inputs.end(), [](const auto& input) {
                        return op_util::is_constant(input.get_node());
                    })) {
                    // postponed_constant attribute is needed to perform constant folding on serialization step
                    fused->get_rt_info()["postponed_constant"] = true;
                    // disable constant folding here to postpone it to serialization step
                    ov::pass::disable_constant_folding(fused);
                }
                if (needs_decompress) {
                    auto convert = std::make_shared<v0::Convert>(fused, target_type);
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

            // Extract input and residual nodes from the pattern match
            auto view_reshape_node = last_add_match.at(view_Reshape).get_node_shared_ptr();
            auto residual_input_node = last_add_match.at(residual_input).get_node_shared_ptr();
            auto original_shape_node = last_add_match.at(original_shape).get_node_shared_ptr();

            // Build the fused MoE computation
            const size_t num_experts = experts.size();

            // Extract dimensions from shapes
            auto hidden_dim = extract_shape_dim(experts[0].gate_proj_weight, 1);

            auto topk = last_add_match.at(topk_TopK).get_node_shared_ptr();
            auto topk_indices_output = topk->output(1);
            auto batch_dim = extract_shape_dim(topk_indices_output, 0);

            auto num_experts_const = v0::Constant::create(element::i64, Shape{}, {static_cast<int64_t>(num_experts)});
            auto num_experts_dim = std::make_shared<v0::Unsqueeze>(num_experts_const, const_0);

            // Tile and reshape input to prepare for batched expert computation
            auto tile_shape = v0::Constant::create(element::i64,
                                                   Shape{2},
                                                   {static_cast<int64_t>(num_experts), static_cast<int64_t>(1)});

            auto view_reshape_shape = std::make_shared<v0::Concat>(OutputVector{const_minus_1, hidden_dim}, 0);
            auto view_reshape = std::make_shared<v1::Reshape>(view_reshape_node, view_reshape_shape, false);
            auto repeated_input = std::make_shared<v0::Tile>(view_reshape, tile_shape);

            auto batched_shape = std::make_shared<v0::Concat>(OutputVector{num_experts_dim, batch_dim, hidden_dim}, 0);
            auto batched_input = std::make_shared<v1::Reshape>(repeated_input, batched_shape, false);

            // Execute three-GEMM SwiGLU pattern: gate_proj, up_proj, and down_proj
            auto gate_bmm = std::make_shared<v0::MatMul>(batched_input, fused_gate_weights, false, true);
            auto gate_swish = std::make_shared<v4::Swish>(gate_bmm);

            auto up_bmm = std::make_shared<v0::MatMul>(batched_input, fused_up_weights, false, true);
            auto swiglu_mul = std::make_shared<v1::Multiply>(gate_swish, up_bmm);

            auto down_bmm = std::make_shared<v0::MatMul>(swiglu_mul, fused_down_weights, false, true);

            auto minus_one_vec = v0::Constant::create(element::i64, Shape{1}, {-1});
            auto expert_output_shape =
                std::make_shared<v0::Concat>(OutputVector{num_experts_dim, batch_dim, minus_one_vec, hidden_dim}, 0);
            auto expert_outputs = std::make_shared<v1::Reshape>(down_bmm, expert_output_shape, false);

            // Build routing weights tensor from TopK outputs
            auto topk_values = topk->output(0);
            auto sum_reduce =
                std::make_shared<v1::ReduceSum>(topk_values, v0::Constant::create(element::i64, Shape{1}, {-1}), true);
            auto normalized_topk = std::make_shared<v1::Divide>(topk_values, sum_reduce);

            auto scatter_shape = std::make_shared<v0::Concat>(OutputVector{batch_dim, num_experts_dim}, 0);
            auto zeros_scalar = v0::Constant::create(normalized_topk->get_element_type(), Shape{}, {0});
            auto zeros_tensor = std::make_shared<v3::Broadcast>(zeros_scalar, scatter_shape);

            auto scatter = std::make_shared<v12::ScatterElementsUpdate>(zeros_tensor,
                                                                        topk_indices_output,
                                                                        normalized_topk,
                                                                        const_1);
            auto router_transpose = std::make_shared<v1::Transpose>(scatter, transpose_perm);
            auto router_shape =
                std::make_shared<v0::Concat>(OutputVector{num_experts_dim, batch_dim, minus_one_vec}, 0);
            auto router_reshape = std::make_shared<v1::Reshape>(router_transpose, router_shape, true);
            auto routing_weights = std::make_shared<v0::Unsqueeze>(router_reshape, const_minus_1);

            // Apply routing weights and reduce across expert dimension
            auto weighted_outputs = std::make_shared<v1::Multiply>(expert_outputs, routing_weights);
            auto final_output = std::make_shared<v1::ReduceSum>(weighted_outputs, const_0, false);

            // Reshape back to original shape and add residual connection
            auto final_reshape = std::make_shared<v1::Reshape>(final_output, original_shape_node, false);
            auto final_add = std::make_shared<v1::Add>(residual_input_node, final_reshape);

            if (last_reshape_node && !last_reshape_node->get_friendly_name().empty()) {
                final_reshape->set_friendly_name(last_reshape_node->get_friendly_name());
            }
            final_add->set_friendly_name(last_add_node->get_friendly_name());

            if (last_reshape_node) {
                ov::copy_runtime_info(ov::NodeVector{last_reshape_node}, ov::NodeVector{final_reshape});
            }
            ov::copy_runtime_info(ov::NodeVector{last_add_node}, ov::NodeVector{final_add});

            ov::replace_node(last_add_node, final_add);
        }

        return true;
    };

    register_patterns(std::vector<std::shared_ptr<Node>>{expert_scatter, last_add}, callback, true);
}

namespace {

// Weight subgraph: Constant -> [Convert] -> [Subtract(zp)] -> [Multiply(scale)] -> [Convert]
struct DenseWeightChain {
    std::shared_ptr<v0::Constant> weights;
    std::shared_ptr<v0::Convert> convert;
    std::shared_ptr<v1::Subtract> subtract;
    std::shared_ptr<v0::Constant> zp;
    std::shared_ptr<v0::Convert> zp_convert;
    std::shared_ptr<v1::Multiply> multiply;
    std::shared_ptr<v0::Constant> scale;
    std::shared_ptr<v0::Convert> scale_convert;
    std::shared_ptr<v0::Convert> out_convert;
};

// Accepts Constant or Convert(Constant)
bool parse_const_operand(const Output<Node>& output,
                         std::shared_ptr<v0::Constant>& constant,
                         std::shared_ptr<v0::Convert>& convert) {
    auto node = output.get_node_shared_ptr();
    convert = ov::as_type_ptr<v0::Convert>(node);
    if (convert) {
        node = convert->get_input_node_shared_ptr(0);
    }
    constant = ov::as_type_ptr<v0::Constant>(node);
    return constant != nullptr;
}

bool parse_dense_weight_chain(const Output<Node>& output, DenseWeightChain& chain) {
    auto node = output.get_node_shared_ptr();
    if (auto convert = ov::as_type_ptr<v0::Convert>(node)) {
        auto input = convert->get_input_node_shared_ptr(0);
        if (ov::is_type<v1::Multiply>(input) || ov::is_type<v1::Subtract>(input)) {
            chain.out_convert = convert;
            node = input;
        }
    }
    if (auto multiply = ov::as_type_ptr<v1::Multiply>(node)) {
        chain.multiply = multiply;
        // The data input keeps the full weight shape, the scale input is broadcast to it
        const auto& out_shape = multiply->get_output_partial_shape(0);
        const size_t data_idx = multiply->get_input_partial_shape(0) == out_shape ? 0 : 1;
        if (!parse_const_operand(multiply->input_value(1 - data_idx), chain.scale, chain.scale_convert)) {
            return false;
        }
        node = multiply->get_input_node_shared_ptr(data_idx);
    }
    if (auto subtract = ov::as_type_ptr<v1::Subtract>(node)) {
        chain.subtract = subtract;
        if (!parse_const_operand(subtract->input_value(1), chain.zp, chain.zp_convert)) {
            return false;
        }
        node = subtract->get_input_node_shared_ptr(0);
    }
    if (auto convert = ov::as_type_ptr<v0::Convert>(node)) {
        chain.convert = convert;
        node = convert->get_input_node_shared_ptr(0);
    }
    chain.weights = ov::as_type_ptr<v0::Constant>(node);
    return chain.weights && chain.weights->get_shape().size() == 2;
}

bool has_same_structure(const DenseWeightChain& a, const DenseWeightChain& b) {
    auto same = [](const std::shared_ptr<Node>& x, const std::shared_ptr<Node>& y) {
        if (!x || !y) {
            return !x && !y;
        }
        return x->get_output_element_type(0) == y->get_output_element_type(0);
    };
    return same(a.weights, b.weights) && a.weights->get_shape() == b.weights->get_shape() &&
           same(a.convert, b.convert) && same(a.subtract, b.subtract) && same(a.zp, b.zp) &&
           same(a.zp_convert, b.zp_convert) && same(a.multiply, b.multiply) && same(a.scale, b.scale) &&
           same(a.scale_convert, b.scale_convert) && same(a.out_convert, b.out_convert);
}

// Stacks 2D constants into a [num_experts, rows, cols] constant. Rows [row_begin, row_end) are taken
// from constants whose first dimension equals full_rows; single-row (broadcast) constants are kept as is.
std::shared_ptr<v0::Constant> stack_constants(const std::vector<std::shared_ptr<v0::Constant>>& constants,
                                              size_t full_rows,
                                              size_t row_begin,
                                              size_t row_end) {
    const auto& type = constants.front()->get_element_type();
    auto shape_2d = [](const Shape& shape) {
        return shape.empty() ? Shape{1, 1} : shape;
    };
    const auto src_shape = shape_2d(constants.front()->get_shape());
    if (src_shape.size() != 2) {
        return nullptr;
    }
    size_t begin = 0;
    size_t rows = src_shape[0];
    if (src_shape[0] == full_rows) {
        begin = row_begin;
        rows = row_end - row_begin;
    } else if (src_shape[0] != 1) {
        return nullptr;
    }
    const size_t cols = src_shape[1];
    if ((cols * type.bitwidth()) % 8 != 0) {
        return nullptr;
    }
    const size_t row_bytes = cols * type.bitwidth() / 8;

    ov::Tensor stacked(type, Shape{constants.size(), rows, cols});
    auto* dst = static_cast<uint8_t*>(stacked.data());
    for (const auto& constant : constants) {
        if (constant->get_element_type() != type || shape_2d(constant->get_shape()) != src_shape) {
            return nullptr;
        }
        const auto* src = static_cast<const uint8_t*>(constant->get_data_ptr()) + begin * row_bytes;
        std::memcpy(dst, src, rows * row_bytes);
        dst += rows * row_bytes;
    }
    return std::make_shared<v0::Constant>(stacked);
}

// Rebuilds the (decompression) weight subgraph over stacked per-expert constants
std::shared_ptr<Node> build_stacked_weight(const std::vector<DenseWeightChain>& chains,
                                           size_t row_begin,
                                           size_t row_end,
                                           const std::string& suffix) {
    const auto& proto = chains.front();
    const size_t full_rows = proto.weights->get_shape()[0];

    auto stack = [&](std::shared_ptr<v0::Constant> DenseWeightChain::*member) -> std::shared_ptr<Node> {
        std::vector<std::shared_ptr<v0::Constant>> constants;
        constants.reserve(chains.size());
        for (const auto& chain : chains) {
            constants.push_back(chain.*member);
        }
        auto stacked = stack_constants(constants, full_rows, row_begin, row_end);
        if (stacked) {
            stacked->set_friendly_name((proto.*member)->get_friendly_name() + suffix);
            stacked->get_rt_info() = (proto.*member)->get_rt_info();
        }
        return stacked;
    };
    auto clone_convert = [&](const Output<Node>& input, const std::shared_ptr<v0::Convert>& origin) {
        auto convert = std::make_shared<v0::Convert>(input, origin->get_destination_type());
        convert->set_friendly_name(origin->get_friendly_name() + suffix);
        convert->get_rt_info() = origin->get_rt_info();
        return convert;
    };

    std::shared_ptr<Node> current = stack(&DenseWeightChain::weights);
    if (!current) {
        return nullptr;
    }
    if (proto.convert) {
        current = clone_convert(current, proto.convert);
    }
    if (proto.subtract) {
        std::shared_ptr<Node> zp = stack(&DenseWeightChain::zp);
        if (!zp) {
            return nullptr;
        }
        if (proto.zp_convert) {
            zp = clone_convert(zp, proto.zp_convert);
        }
        auto subtract = std::make_shared<v1::Subtract>(current, zp);
        subtract->set_friendly_name(proto.subtract->get_friendly_name() + suffix);
        subtract->get_rt_info() = proto.subtract->get_rt_info();
        current = subtract;
    }
    if (proto.multiply) {
        std::shared_ptr<Node> scale = stack(&DenseWeightChain::scale);
        if (!scale) {
            return nullptr;
        }
        if (proto.scale_convert) {
            scale = clone_convert(scale, proto.scale_convert);
        }
        auto multiply = std::make_shared<v1::Multiply>(current, scale);
        multiply->set_friendly_name(proto.multiply->get_friendly_name() + suffix);
        multiply->get_rt_info() = proto.multiply->get_rt_info();
        current = multiply;
    }
    if (proto.out_convert) {
        current = clone_convert(current, proto.out_convert);
    }
    return current;
}

}  // namespace

ov::pass::FuseDenseMOEExperts::FuseDenseMOEExperts() {
    MATCHER_SCOPE(FuseDenseMOEExperts);

    // Single expert: SwiGLU MLP over all tokens weighted by the dense routing weight of the expert
    auto x = pattern::any_input(pattern::rank_equals(2));
    auto gate_up_weights = pattern::any_input();
    auto single_data_consumer = [](const Output<Node>& output) {
        const auto consumers = output.get_target_inputs();
        // Exported split lengths can depend on ShapeOf of the first expert's projection.
        return std::count_if(consumers.begin(), consumers.end(), [](const Input<Node>& input) {
                   return !ov::is_type<op_util::ShapeOfBase>(input.get_node());
               }) == 1;
    };
    auto gate_up = pattern::wrap_type<v0::MatMul>(
        {x, gate_up_weights},
        single_data_consumer && pattern::attrs_match({{"transpose_a", false}, {"transpose_b", true}}));
    auto split_axis = pattern::wrap_type<v0::Constant>();
    auto split = pattern::wrap_type_strict_index<v1::VariadicSplit>({gate_up, split_axis, pattern::any_input()});
    auto act = pattern::wrap_type<v4::Swish>({split->output(0)}, pattern::consumers_count(1));
    auto swiglu = pattern::wrap_type<v1::Multiply>({act, split->output(1)}, pattern::consumers_count(1));
    auto down_weights = pattern::any_input();
    auto down = pattern::wrap_type<v0::MatMul>(
        {swiglu, down_weights},
        pattern::consumers_count(1) && pattern::attrs_match({{"transpose_a", false}, {"transpose_b", true}}));

    // Dense routing weight: sum_k(one_hot[t, k, e] * routing[t, k])
    auto one_hot = pattern::wrap_type<v1::OneHot>();
    auto permute = pattern::wrap_type<v1::Transpose>(
        {one_hot, pattern::wrap_type<v0::Constant>(pattern::value_matches("2, 1, 0"))});
    auto expert_id = pattern::wrap_type<v0::Constant>(pattern::rank_equals(0));
    auto mask = pattern::wrap_type<v8::Gather>(
        {permute, expert_id, pattern::wrap_type<v0::Constant>(pattern::value_matches("0"))},
        {{"batch_dims", 0}});
    auto mask_convert = pattern::optional<v0::Convert>({mask});
    auto routing = pattern::any_input(pattern::rank_equals(2));
    auto routing_t =
        pattern::wrap_type<v1::Transpose>({routing, pattern::wrap_type<v0::Constant>(pattern::value_matches("1, 0"))});
    auto weighted_mask = pattern::wrap_type<v1::Multiply>({mask_convert, routing_t}, pattern::consumers_count(1));
    auto token_weights = pattern::wrap_type<v1::ReduceSum>(
        {weighted_mask, pattern::wrap_type<v0::Constant>(pattern::value_matches("0"))},
        pattern::consumers_count(1),
        {{"keep_dims", false}});
    auto unsqueeze_axis = pattern::wrap_type<v0::Constant>();
    auto token_weights_2d =
        pattern::wrap_type<v0::Unsqueeze>({token_weights, unsqueeze_axis}, pattern::consumers_count(1));
    auto expert_out = pattern::wrap_type<v1::Multiply>({down, token_weights_2d});

    // The root is the last Add of the chain accumulating the expert outputs
    auto chain_tail = pattern::wrap_type<v1::Add>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto tail = m.get_match_root();
        if (transformation_callback(tail)) {
            return false;
        }
        const auto consumers = tail->output(0).get_target_inputs();
        if (consumers.empty() || (consumers.size() == 1 && ov::is_type<v1::Add>(consumers.begin()->get_node()))) {
            return false;
        }

        OutputVector leaves;
        std::vector<std::shared_ptr<Node>> stack{tail};
        while (!stack.empty()) {
            const auto node = stack.back();
            stack.pop_back();
            for (const auto& input : node->input_values()) {
                const auto input_node = input.get_node_shared_ptr();
                if (ov::is_type<v1::Add>(input_node) && input.get_target_inputs().size() == 1) {
                    stack.push_back(input_node);
                } else {
                    leaves.push_back(input);
                }
            }
        }

        struct ExpertInfo {
            size_t id;
            DenseWeightChain gate_up;
            DenseWeightChain down;
        };
        std::vector<ExpertInfo> experts;
        OutputVector other_addends;
        Output<Node> x_value, permute_value, routing_value;
        std::shared_ptr<v1::OneHot> one_hot_node;
        int64_t gate_rows = -1;

        for (const auto& leaf : leaves) {
            pattern::Matcher expert_matcher(expert_out);
            if (leaf.get_target_inputs().size() != 1 || !expert_matcher.match(leaf)) {
                other_addends.push_back(leaf);
                continue;
            }
            const auto& pm = expert_matcher.get_pattern_value_map();
            if (experts.empty()) {
                x_value = pm.at(x);
                permute_value = pm.at(permute);
                routing_value = pm.at(routing);
                one_hot_node = ov::as_type_ptr<v1::OneHot>(pm.at(one_hot).get_node_shared_ptr());
            } else if (pm.at(x) != x_value || pm.at(permute) != permute_value || pm.at(routing) != routing_value) {
                return false;
            }

            const auto axis = ov::as_type_ptr<v0::Constant>(pm.at(unsqueeze_axis).get_node_shared_ptr());
            const auto axis_values = axis->cast_vector<int64_t>();
            if (axis_values.size() != 1 || (axis_values[0] != 1 && axis_values[0] != -1)) {
                return false;
            }

            const auto split_node = pm.at(split).get_node_shared_ptr();
            const auto split_axis_values =
                ov::as_type_ptr<v0::Constant>(pm.at(split_axis).get_node_shared_ptr())->cast_vector<int64_t>();
            if (split_axis_values.size() != 1 || (split_axis_values[0] != 1 && split_axis_values[0] != -1) ||
                split_node->get_output_size() != 2 || split_node->get_output_target_inputs(0).size() != 1 ||
                split_node->get_output_target_inputs(1).size() != 1) {
                return false;
            }
            const auto& gate_shape = split_node->get_output_partial_shape(0);
            const auto& up_shape = split_node->get_output_partial_shape(1);
            if (gate_shape.rank() != 2 || up_shape.rank() != 2 || gate_shape[1].is_dynamic() ||
                up_shape[1].is_dynamic()) {
                return false;
            }

            ExpertInfo info;
            info.id = static_cast<size_t>(
                ov::as_type_ptr<v0::Constant>(pm.at(expert_id).get_node_shared_ptr())->cast_vector<int64_t>()[0]);
            if (!parse_dense_weight_chain(pm.at(gate_up_weights), info.gate_up) ||
                !parse_dense_weight_chain(pm.at(down_weights), info.down)) {
                return false;
            }
            const auto expert_gate_rows = gate_shape[1].get_length();
            if (static_cast<size_t>(expert_gate_rows + up_shape[1].get_length()) !=
                info.gate_up.weights->get_shape()[0]) {
                return false;
            }
            if (experts.empty()) {
                gate_rows = expert_gate_rows;
            } else if (gate_rows != expert_gate_rows || !has_same_structure(experts.front().gate_up, info.gate_up) ||
                       !has_same_structure(experts.front().down, info.down)) {
                return false;
            }
            experts.push_back(std::move(info));
        }
        if (experts.size() < 2) {
            return false;
        }

        // Routing mask must be a one-hot of the top-k indices covering exactly all experts
        const auto depth = ov::as_type_ptr<v0::Constant>(one_hot_node->get_input_node_shared_ptr(1));
        const auto on_value = ov::as_type_ptr<v0::Constant>(one_hot_node->get_input_node_shared_ptr(2));
        const auto off_value = ov::as_type_ptr<v0::Constant>(one_hot_node->get_input_node_shared_ptr(3));
        const auto& topk_indices = one_hot_node->input_value(0);
        if (!depth || !on_value || !off_value || on_value->cast_vector<float>() != std::vector<float>{1.0f} ||
            off_value->cast_vector<float>() != std::vector<float>{0.0f} ||
            topk_indices.get_partial_shape().rank() != 2 ||
            (one_hot_node->get_axis() != 2 && one_hot_node->get_axis() != -1)) {
            return false;
        }
        const auto num_experts = static_cast<size_t>(depth->cast_vector<int64_t>()[0]);
        std::sort(experts.begin(), experts.end(), [](const ExpertInfo& a, const ExpertInfo& b) {
            return a.id < b.id;
        });
        if (experts.size() != num_experts) {
            return false;
        }
        for (size_t i = 0; i < num_experts; ++i) {
            if (experts[i].id != i) {
                return false;
            }
        }

        const auto& x_shape = x_value.get_partial_shape();
        if (x_shape[1].is_dynamic() || routing_value.get_element_type() != tail->get_output_element_type(0)) {
            return false;
        }
        const auto hidden_size = x_shape[1].get_length();
        const auto output_size = static_cast<int64_t>(experts.front().down.weights->get_shape()[0]);
        const auto gate_up_rows = experts.front().gate_up.weights->get_shape()[0];

        std::vector<DenseWeightChain> gate_up_chains, down_chains;
        for (const auto& expert : experts) {
            gate_up_chains.push_back(expert.gate_up);
            down_chains.push_back(expert.down);
        }
        const auto gate_weights = build_stacked_weight(gate_up_chains, 0, gate_rows, "/fused_dense_moe_gate");
        const auto up_weights = build_stacked_weight(gate_up_chains, gate_rows, gate_up_rows, "/fused_dense_moe_up");
        const auto fused_down_weights =
            build_stacked_weight(down_chains, 0, static_cast<size_t>(output_size), "/fused_dense_moe_down");
        if (!gate_weights || !up_weights || !fused_down_weights) {
            return false;
        }

        const auto num_experts_i64 = static_cast<int64_t>(num_experts);
        auto i64_const = [](const Shape& shape, const std::vector<int64_t>& values) {
            return v0::Constant::create(element::i64, shape, values);
        };

        // Batched expert computation over all tokens: [E, T, H]
        auto tile = std::make_shared<v0::Tile>(x_value, i64_const(Shape{2}, {num_experts_i64, 1}));
        auto tiled_input =
            std::make_shared<v1::Reshape>(tile, i64_const(Shape{3}, {num_experts_i64, -1, hidden_size}), false);
        auto gate_mm = std::make_shared<v0::MatMul>(tiled_input, gate_weights, false, true);
        auto gate_act = std::make_shared<v4::Swish>(gate_mm);
        auto up_mm = std::make_shared<v0::MatMul>(tiled_input, up_weights, false, true);
        auto swiglu_mul = std::make_shared<v1::Multiply>(gate_act, up_mm);
        auto down_mm = std::make_shared<v0::MatMul>(swiglu_mul, fused_down_weights, false, true);
        auto expert_outputs =
            std::make_shared<v1::Reshape>(down_mm, i64_const(Shape{3}, {num_experts_i64, -1, output_size}), false);

        // Dense routing weights [E, T, 1]; sum reduction keeps the one-hot accumulation semantics
        auto routing_shape = std::make_shared<v3::ShapeOf>(routing_value, element::i64);
        auto num_tokens =
            std::make_shared<v8::Gather>(routing_shape, i64_const(Shape{1}, {0}), i64_const(Shape{}, {0}));
        auto scatter_shape =
            std::make_shared<v0::Concat>(OutputVector{num_tokens, i64_const(Shape{1}, {num_experts_i64})}, 0);
        auto zeros =
            std::make_shared<v3::Broadcast>(v0::Constant::create(routing_value.get_element_type(), Shape{}, {0}),
                                            scatter_shape);
        auto scatter = std::make_shared<v12::ScatterElementsUpdate>(zeros,
                                                                    topk_indices,
                                                                    routing_value,
                                                                    i64_const(Shape{}, {1}),
                                                                    v12::ScatterElementsUpdate::Reduction::SUM,
                                                                    true);
        auto router_transpose = std::make_shared<v1::Transpose>(scatter, i64_const(Shape{2}, {1, 0}));
        auto router_reshape =
            std::make_shared<v1::Reshape>(router_transpose, i64_const(Shape{2}, {num_experts_i64, -1}), false);
        auto router_unsqueeze = std::make_shared<v0::Unsqueeze>(router_reshape, i64_const(Shape{1}, {-1}));

        auto weighted_outputs = std::make_shared<v1::Multiply>(expert_outputs, router_unsqueeze);
        std::shared_ptr<Node> result =
            std::make_shared<v1::ReduceSum>(weighted_outputs, i64_const(Shape{1}, {0}), false);
        for (const auto& addend : other_addends) {
            result = std::make_shared<v1::Add>(result, addend);
        }

        result->set_friendly_name(tail->get_friendly_name());
        ov::copy_runtime_info(tail, result);
        ov::replace_node(tail, result);
        return true;
    };

    auto matcher = std::make_shared<pattern::Matcher>(chain_tail, matcher_name);
    register_matcher(matcher, callback);
}

bool ov::pass::FuseMOE::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(FuseMOE);
    ov::pass::Manager manager(get_pass_config(), "FuseMOE");

    manager.register_pass<ov::pass::FuseMOEExperts>();
    manager.run_passes(model);
    return false;
}

}  // namespace ov::pass
