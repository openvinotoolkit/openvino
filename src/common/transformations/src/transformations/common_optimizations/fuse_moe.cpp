// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fuse_moe.hpp"

#include <cstdint>
#include <limits>
#include <tuple>
#include <chrono>
#include <iostream>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rank.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/op/one_hot.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/op/util/shape_of_base.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/utils/gen_pattern.hpp"
#include "transformations/utils/utils.hpp"
#include "transformations/common_optimizations/nop_elimination.hpp"
#include "openvino/core/validation_util.hpp"
#include "transformations/utils/print_model.hpp"

namespace ov {
namespace pass {

using namespace ov::gen_pattern;
using namespace ov::pass;

namespace {

// Local MOE configuration structure to avoid dependency on v16::MOE
struct MOEConfig {
    size_t expert_num = 0;
    size_t hidden_size = 0;
    size_t intermediate_size = 0;
    size_t group_size = 0;
    size_t topk = 0;
    ov::element::Type weight_type = ov::element::f32;
};

auto gen_expert_pattern(std::shared_ptr<ov::Node> final_hidden_states,
                        std::shared_ptr<ov::Node> hidden_states,
                        std::shared_ptr<ov::Node> expert_mask,
                        std::shared_ptr<ov::Node> routing_weights) {
    // shape: [1], aka topk
    auto routing_weights_shapeof_split = makePattern(ov::Rank(1));
    // shape: [2], data = [1, hidden_size]
    auto index_add__ShapeOf_22 = makePattern("[2]");

    auto hidden_size = ov::gen_pattern::Symbol("hidden_size");
    auto expert_no = ov::gen_pattern::Symbol("expert_no");

    // Simplified pattern for FP16/FP32 weights only
#define WEIGHT_PATTERN(idx)                                                                                   \
    auto weight_const##idx = pattern::wrap_type<ov::op::v0::Constant>();                                      \
    auto weight_const_convert##idx = makePattern<ov::op::v0::Convert>({weight_const##idx});                   \
    /* f16+convert | f32 */                                                                                   \
    auto final_weight##idx = weight_const_convert##idx | weight_const##idx;

    // expert_mask[expert_idx]
    auto select_Gather_2 = makePattern<ov::op::v8::Gather>({expert_mask, expert_no, 0}, {{"batch_dims", 0}});
    auto squeeze_Squeeze_7 = makePattern<opset1::Squeeze>({select_Gather_2, 0});   //  tensor_array<i64[2,?]> __module.model.layers.1.mlp/aten::squeeze/Squeeze_7(__module.model.layers.1.mlp/aten::select/Gather_7, 60)

    // x = torch.where(expert_mask[expert_idx]), x shape: [2, nonzero], dim0: topk, dim1: batch
    auto ListUnpack_NonZero_2 = makePattern<ov::op::v3::NonZero>({squeeze_Squeeze_7}, {{"output_type", "i64"}});
    // topk, batch = torch.where(expert_mask[expert_idx])
    auto ListUnpack_Split_2 = makePattern<ov::op::v1::Split>({ListUnpack_NonZero_2, 0}, {{"num_splits", 2}});
    ListUnpack_Split_2->set_output_size(2);
    // batch
    auto ListUnpack_Squeeze_0_2_0 = makePattern<ov::op::v0::Squeeze>({ListUnpack_Split_2->output(1), 0});
    auto ListUnpack_Squeeze_0_2_1 =
        makePattern<ov::op::v1::Reshape>({ListUnpack_Split_2->output(1), {-1}}, {{"special_zero", false}});
    auto ListUnpack_Squeeze_0_2 = ListUnpack_Squeeze_0_2_0 | ListUnpack_Squeeze_0_2_1;
    auto index_add__Convert_2_org =
        makePattern<ov::op::v0::Convert>({ListUnpack_Squeeze_0_2}, {{"destination_type", "i32"}});
    auto index_add__Convert_2 = index_add__Convert_2_org | ListUnpack_Squeeze_0_2;
    auto index_add__Reshape_2 =
        makePattern<ov::op::v1::Reshape>({index_add__Convert_2, {-1, 1}}, {{"special_zero", false}});
    auto index_add__Broadcast_25 =
        makePattern<ov::op::v3::Broadcast>({index_add__Reshape_2, index_add__ShapeOf_22}, {{"mode", "bidirectional"}});
    auto index_Gather_4 =
        makePattern<ov::op::v8::Gather>({hidden_states, index_add__Convert_2, 1}, {{"batch_dims", 0}});
    auto reshape_Reshape_2 =
        makePattern<ov::op::v1::Reshape>({index_Gather_4, {-1, hidden_size}}, {{"special_zero", true}});
    WEIGHT_PATTERN(0)
    auto gate_linear_MatMul = makePattern<ov::op::v0::MatMul>({reshape_Reshape_2, final_weight0},
                                                              {{"transpose_a", false}, {"transpose_b", true}});
    auto silu_Swish = makePattern<ov::op::v4::Swish>({gate_linear_MatMul});
    WEIGHT_PATTERN(1)
    auto up_linear_MatMul = makePattern<ov::op::v0::MatMul>({reshape_Reshape_2, final_weight1},
                                                            {{"transpose_a", false}, {"transpose_b", true}});
    auto mul_Multiply =
        makePattern<ov::op::v1::Multiply>({silu_Swish, up_linear_MatMul}, {{"auto_broadcast", "numpy"}});
    WEIGHT_PATTERN(2)
    auto down_linear_MatMul =
        makePattern<ov::op::v0::MatMul>({mul_Multiply, final_weight2}, {{"transpose_a", false}, {"transpose_b", true}});
    auto ListUnpack_Squeeze_2_0 = makePattern<ov::op::v0::Squeeze>({ListUnpack_Split_2->output(0), 0});
    auto ListUnpack_Squeeze_2_1 =
        makePattern<ov::op::v1::Reshape>({ListUnpack_Split_2->output(0), {-1}}, {{"special_zero", false}});
    auto ListUnpack_Squeeze_2 = ListUnpack_Squeeze_2_0 | ListUnpack_Squeeze_2_1;
    auto index_Convert_6 = makePattern<ov::op::v0::Convert>({ListUnpack_Squeeze_2}, {{"destination_type", "i32"}});
    // self.topk * batch, index_split=shapeof(routing_weights), shape: [batch, self.topk, 1]
    auto index_Multiply_2 = makePattern<ov::op::v1::Multiply>({index_add__Convert_2, routing_weights_shapeof_split},
                                                              {{"auto_broadcast", "numpy"}});
    // self.topk * batch + topk
    auto index_Add_2 = makePattern<ov::op::v1::Add>({index_Convert_6 | ListUnpack_Squeeze_2, index_Multiply_2},
                                                    {{"auto_broadcast", "numpy"}});
    // routing_weights', shape[self.topk * batch, 1]
    auto index_Gather_5 = makePattern<ov::op::v8::Gather>({routing_weights, index_Add_2, 0}, {{"batch_dims", 0}});
    auto index_Reshape_8_2 = makePattern<ov::op::v1::Reshape>({index_Gather_5, {0, 1}}, {{"special_zero", true}});
    auto mul_Multiply_3 = makePattern<ov::op::v1::Multiply>({down_linear_MatMul, index_Gather_5 | index_Reshape_8_2},
                                                            {{"auto_broadcast", "numpy"}});
    auto index_add__Broadcast_26 =
        makePattern<ov::op::v3::Broadcast>({mul_Multiply_3, index_add__ShapeOf_22}, {{"mode", "bidirectional"}});
    auto index_add__ScatterElementsUpdate_8 = makePattern<ov::op::v12::ScatterElementsUpdate>(
        {final_hidden_states, index_add__Broadcast_25, index_add__Broadcast_26, 0},
        {{"reduction", "sum"}, {"use_init_val", true}});

    auto result = index_add__ScatterElementsUpdate_8;
    auto extract_expert = [=](ov::pass::pattern::Matcher& m,
                              MOEConfig& config,
                              std::vector<std::shared_ptr<ov::op::v0::Constant>>& expert_constants) {
        PatternValidator validator(m);
        if (!validator) {
            return -1;
        }

        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();
        auto hidden_size = static_cast<size_t>(validator["hidden_size"]);
        auto expert_no = static_cast<int>(validator["expert_no"]);

        auto expert_mask_node = pattern_map.at(expert_mask);
        auto ps = expert_mask_node.get_partial_shape();
        if (ps.rank().is_dynamic() || ps[0].is_dynamic() || ps[1].is_dynamic()) {
            return -1;
        }

        auto expert_num = ps[0].get_length();
        auto topk = ps[1].get_length();

        auto last_node = pattern_map.at(index_add__ScatterElementsUpdate_8).get_node_shared_ptr();

        // Extract constants for this expert
        std::shared_ptr<ov::op::v0::Constant> gate_weight, up_weight, down_weight;

        gate_weight = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(weight_const0).get_node_shared_ptr());
        up_weight = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(weight_const1).get_node_shared_ptr());
        down_weight = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(weight_const2).get_node_shared_ptr());

        // Store only the weight constants (no scale/zp for simplified FP16/FP32 pattern)
        expert_constants.clear();
        expert_constants.push_back(gate_weight);   // gate_weight
        expert_constants.push_back(up_weight);     // up_weight
        expert_constants.push_back(down_weight);   // down_weight

        auto gate_shape = gate_weight->get_shape();
        auto up_shape = up_weight->get_shape();
        auto down_shape = down_weight->get_shape();
        auto intermediate_size = gate_shape[0];
        size_t group_size = 0;
        // checking weight should be enough, scale/zp should be checked in the pattern
        OPENVINO_ASSERT(gate_shape == up_shape,
                        "up shape must be equal to gate shape, gate shape: ",
                        gate_shape,
                        ", up shape: ",
                        up_shape);
        OPENVINO_ASSERT(hidden_size == down_shape[0],
                        "down weight shape[0] is not expected, expected: ",
                        hidden_size,
                        ", current: ",
                        down_shape[0]);
        if (gate_shape.size() == 3) {
            group_size = gate_shape[2];
            OPENVINO_ASSERT(down_shape.size() == 3 && gate_shape[2] == down_shape[2],
                            "down shape is not compatible gate shape, gate shape: ",
                            gate_shape,
                            ", down shape: ",
                            down_shape);
        }

        config.expert_num = expert_num;
        config.hidden_size = hidden_size;
        config.intermediate_size = intermediate_size;
        config.group_size = group_size;
        config.topk = topk;
        config.weight_type = gate_weight->get_element_type();
        OPENVINO_ASSERT(up_weight->get_element_type() == config.weight_type,
                        "precision of up weight must be same with gate, gate: ",
                        config.weight_type,
                        ", up: ",
                        up_weight->get_element_type());
        OPENVINO_ASSERT(down_weight->get_element_type() == config.weight_type,
                        "precision of down weight must be same with gate, gate: ",
                        config.weight_type,
                        ", down: ",
                        down_weight->get_element_type());
        return expert_no;
    };

    return std::make_tuple(result, extract_expert);
}

}  // namespace

ov::pass::FuseMOEUnified::FuseMOEUnified() {
    MATCHER_SCOPE(FuseMOEUnified);
    // Use the original router/expert pattern but collect all experts in a single pass
    // param1: [batch*seq, 2048]
    auto final_hidden_states = makePattern(ov::Rank(2));
    auto router_logits = makePattern(ov::Rank(2));
    // f32[?,128]
    auto softmax_Softmax = makePattern<ov::op::v8::Softmax>({router_logits}, {{"axis", 1}});
    auto topk_TopK = makePattern<ov::op::v11::TopK>({softmax_Softmax, pattern::any_input()});
    topk_TopK->set_output_size(2);
    auto sum_ReduceSum = makePattern<ov::op::v1::ReduceSum>({topk_TopK->output(0), {-1}}, {{"keep_dims", true}});
    auto div__Divide = makePattern<ov::op::v1::Divide>({topk_TopK->output(0), sum_ReduceSum},
                                                       {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});
    auto one_hot_OneHot = makePattern<ov::op::v1::OneHot>(
        {topk_TopK->output(1), pattern::any_input(), pattern::any_input(), pattern::any_input()},
        {{"axis", 2}});
    // param2: expert_mask: [128, 8, batch]
    auto permute_Transpose = makePattern<ov::op::v1::Transpose>({one_hot_OneHot, {2, 1, 0}});

    // hidden_states_2d: f32[-1, 2048]
    auto view_Reshape = makePattern(ov::Rank(2));
    // param1: hidden_states: f32[1, -1, 2048]
    auto unsqueeze_Unsqueeze = makePattern<ov::op::v0::Unsqueeze>({view_Reshape, 0});
    auto unsqueeze_Unsqueeze_1 = makePattern<ov::op::v0::Unsqueeze>({div__Divide, 2});
    auto index_ShapeOf_1 = makePattern<ov::op::v3::ShapeOf>({unsqueeze_Unsqueeze_1}, {{"output_type", "i32"}});
    auto index_Slice = makePattern<ov::op::v8::Slice>({index_ShapeOf_1, {0}, {2}, {1}, {0}});
    auto index_ReduceProd = makePattern<ov::op::v1::ReduceProd>({index_Slice, 0}, {{"keep_dims", true}});
    auto index_Concat = makePattern<ov::op::v0::Concat>({index_ReduceProd, {-1}}, {{"axis", 0}});
    // param4: routing weights: [self.topk * batch, 1]
    auto index_Reshape =
        makePattern<ov::op::v1::Reshape>({unsqueeze_Unsqueeze_1, index_Concat}, {{"special_zero", true}});

    auto [result, extract_func] =
        gen_expert_pattern(final_hidden_states, unsqueeze_Unsqueeze, permute_Transpose, index_Reshape);

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS, result = result, extract_func = extract_func](ov::pass::pattern::Matcher& m) {
        auto start_time = std::chrono::high_resolution_clock::now();
        std::cout << "[MOE_TIMING] Starting FuseMOE transformation" << std::endl;

        MOEConfig config;
        std::vector<std::shared_ptr<ov::op::v0::Constant>> expert_constants;
        auto expert_no = extract_func(m, config, expert_constants);

        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();

        if (expert_no == 0) {
            auto pattern_extract_time = std::chrono::high_resolution_clock::now();
            std::cout << "[MOE_TIMING] Pattern extraction took: "
                      << std::chrono::duration_cast<std::chrono::microseconds>(pattern_extract_time - start_time).count()
                      << " µs" << std::endl;
            // This is the first expert - collect all experts in this single pass
            // router_logits: i32[batch*seq, expert_num]
            auto router_logits_node = pattern_map.at(router_logits).get_node_shared_ptr();
            // f32[batch*seq, hidden_size]
            auto hidden_states_2d = pattern_map.at(view_Reshape).get_node_shared_ptr();
            // f32[batch*seq, hidden_size]
            auto last_node = pattern_map.at(result).get_node_shared_ptr();

            // Extract hidden_size and intermediate_size from the first expert weights
            size_t hidden_size = config.hidden_size;
            size_t intermediate_size = config.intermediate_size;

            // Find all experts by traversing the scatter chain
            std::vector<std::vector<std::shared_ptr<ov::op::v0::Constant>>> all_expert_weights;
            all_expert_weights.push_back(expert_constants); // Add first expert

            // Traverse forward to find subsequent experts
            std::shared_ptr<ov::Node> current = last_node;
            while (current) {
                // Find the next ScatterElementsUpdate operation
                std::shared_ptr<ov::Node> next_scatter = nullptr;
                for (auto& user : current->get_users()) {
                    if (auto scatter = ov::as_type_ptr<ov::op::v12::ScatterElementsUpdate>(user)) {
                        next_scatter = scatter;
                        break;
                    }
                }

                if (!next_scatter) break;

                // Extract weights from this expert
                std::vector<std::shared_ptr<ov::op::v0::Constant>> current_expert_weights;
                std::set<std::shared_ptr<ov::Node>> visited;

                std::function<void(std::shared_ptr<ov::Node>)> extract_weights;
                extract_weights = [&](std::shared_ptr<ov::Node> node) {
                    if (visited.count(node)) return;
                    visited.insert(node);

                    if (auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(node)) {
                        auto weight_input = matmul->get_input_node_shared_ptr(1);
                        std::shared_ptr<ov::op::v0::Constant> weight_const = nullptr;

                        if (auto constant = ov::as_type_ptr<ov::op::v0::Constant>(weight_input)) {
                            weight_const = constant;
                        } else if (auto convert = ov::as_type_ptr<ov::op::v0::Convert>(weight_input)) {
                            weight_const = ov::as_type_ptr<ov::op::v0::Constant>(convert->get_input_node_shared_ptr(0));
                        }

                        if (weight_const) {
                            auto shape = weight_const->get_shape();
                            if (shape.size() == 2) {
                                // Check if this looks like an expert weight based on dimensions
                                // Gate/up weights: [intermediate_size, packed_hidden_size]
                                // Down weights: [hidden_size, packed_intermediate_size]
                                if ((shape[0] == intermediate_size && shape[1] % hidden_size == 0) ||  // gate/up weights
                                    (shape[0] == hidden_size && shape[1] % intermediate_size == 0)) {   // down weights
                                    current_expert_weights.push_back(weight_const);
                                }
                            }
                        }
                    }

                    for (auto& input : node->input_values()) {
                        extract_weights(input.get_node_shared_ptr());
                    }
                };

                extract_weights(next_scatter);

                // Sort and select weights in the correct order (gate, up, down)
                std::sort(current_expert_weights.begin(), current_expert_weights.end(),
                          [](const std::shared_ptr<ov::op::v0::Constant>& a,
                             const std::shared_ptr<ov::op::v0::Constant>& b) {
                    auto a_shape = a->get_shape();
                    auto b_shape = b->get_shape();
                    // Sort by first dimension: smaller dimensions (intermediate_size) come before larger (hidden_size)
                    if (a_shape[0] != b_shape[0]) {
                        return a_shape[0] < b_shape[0];
                    }
                    return false;
                });

                std::vector<std::shared_ptr<ov::op::v0::Constant>> expert_weights_ordered;
                size_t gate_up_count = 0;
                for (auto& weight : current_expert_weights) {
                    auto shape = weight->get_shape();
                    if (shape[0] == intermediate_size && gate_up_count < 2) {
                        // Gate and up weights (first two with intermediate_size)
                        expert_weights_ordered.push_back(weight);
                        gate_up_count++;
                    } else if (shape[0] == hidden_size && expert_weights_ordered.size() == 2) {
                        // Down weight (hidden_size x packed_intermediate_size)
                        expert_weights_ordered.push_back(weight);
                        break;
                    }
                }

                if (expert_weights_ordered.size() == 3) {
                    all_expert_weights.push_back(expert_weights_ordered);
                }

                current = next_scatter;
            }

            // Update config with actual expert count
            config.expert_num = all_expert_weights.size();

            auto experts_collected_time = std::chrono::high_resolution_clock::now();
            std::cout << "[MOE_TIMING] Expert collection took: "
                      << std::chrono::duration_cast<std::chrono::microseconds>(experts_collected_time - pattern_extract_time).count()
                      << " µs, found " << config.expert_num << " experts" << std::endl;

            // Concatenate corresponding weights across experts to create 3D tensors
            // Separate weights by type: gate, up, down
            std::vector<std::shared_ptr<ov::op::v0::Constant>> gate_weights, up_weights, down_weights;

            for (const auto& expert_weights : all_expert_weights) {
                if (expert_weights.size() == 3) {
                    gate_weights.push_back(expert_weights[0]);  // gate weight
                    up_weights.push_back(expert_weights[1]);    // up weight
                    down_weights.push_back(expert_weights[2]);  // down weight
                }
            }

            // Create concatenated weight tensors using Concat op
            // First unsqueeze each 2D weight to 3D by adding expert dimension at axis 0
            ov::OutputVector gate_weights_3d, up_weights_3d, down_weights_3d;

            for (auto& gate_weight : gate_weights) {
                auto unsqueeze_axes = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int32_t>{0});
                auto gate_3d = std::make_shared<ov::op::v0::Unsqueeze>(gate_weight, unsqueeze_axes);
                gate_weights_3d.push_back(gate_3d);
            }

            for (auto& up_weight : up_weights) {
                auto unsqueeze_axes = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int32_t>{0});
                auto up_3d = std::make_shared<ov::op::v0::Unsqueeze>(up_weight, unsqueeze_axes);
                up_weights_3d.push_back(up_3d);
            }

            for (auto& down_weight : down_weights) {
                auto unsqueeze_axes = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int32_t>{0});
                auto down_3d = std::make_shared<ov::op::v0::Unsqueeze>(down_weight, unsqueeze_axes);
                down_weights_3d.push_back(down_3d);
            }

            // Gate weights: [num_experts, intermediate_size, packed_hidden_size]
            auto gate_concat = std::make_shared<ov::op::v0::Concat>(gate_weights_3d, 0);
            gate_concat->set_friendly_name("moe_gate_weights_concat");

            // Up weights: [num_experts, intermediate_size, packed_hidden_size]
            auto up_concat = std::make_shared<ov::op::v0::Concat>(up_weights_3d, 0);
            up_concat->set_friendly_name("moe_up_weights_concat");

            // Down weights: [num_experts, hidden_size, packed_intermediate_size]
            auto down_concat = std::make_shared<ov::op::v0::Concat>(down_weights_3d, 0);
            down_concat->set_friendly_name("moe_down_weights_concat");

            auto weight_concat_time = std::chrono::high_resolution_clock::now();
            std::cout << "[MOE_TIMING] Weight concatenation took: "
                      << std::chrono::duration_cast<std::chrono::microseconds>(weight_concat_time - experts_collected_time).count()
                      << " µs" << std::endl;

            // Create decomposed MOE implementation following Qwen3VL structure
            // Instead of using MOE operator, implement with standard ops

            // 1. Router processing: TopK and normalization
            auto router_output = std::make_shared<ov::op::v11::TopK>(
                router_logits_node,
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, config.topk),
                -1, "max", "value", ov::element::i32);
            auto routing_weights = router_output->output(0);  // [batch_size, topk]
            auto router_indices = router_output->output(1);   // [batch_size, topk]

            // Normalize routing weights
            auto sum_weights = std::make_shared<ov::op::v1::ReduceSum>(
                routing_weights,
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int32_t>{-1}),
                true);
            routing_weights = std::make_shared<ov::op::v1::Divide>(routing_weights, sum_weights);

            auto router_time = std::chrono::high_resolution_clock::now();
            std::cout << "[MOE_TIMING] Router processing took: "
                      << std::chrono::duration_cast<std::chrono::microseconds>(router_time - weight_concat_time).count()
                      << " µs" << std::endl;

            // 2. Repeat hidden states for all experts (following Qwen3VL else branch)
            // hidden_states: [batch_size, hidden_size] -> [num_experts, batch_size, hidden_size]
            auto repeat_shape = std::make_shared<ov::op::v0::Constant>(
                ov::element::i32, ov::Shape{3},
                std::vector<int32_t>{static_cast<int32_t>(config.expert_num), 1, 1});
            auto broadcasted_hidden_states = std::make_shared<ov::op::v3::Broadcast>(
                hidden_states_2d, repeat_shape, "bidirectional");

            // Reshape to [num_experts, batch_size, hidden_size]
            auto reshape_shape_expanded = std::make_shared<ov::op::v0::Constant>(
                ov::element::i32, ov::Shape{3},
                std::vector<int32_t>{static_cast<int32_t>(config.expert_num), -1, static_cast<int32_t>(config.hidden_size)});
            auto repeated_hidden_states = std::make_shared<ov::op::v1::Reshape>(broadcasted_hidden_states, reshape_shape_expanded, false);

            // 3. Batch matrix multiply: gate_up = bmm(hidden_states, gate_up_proj)
            // gate_concat shape: [num_experts, intermediate_size, hidden_size]
            // up_concat shape: [num_experts, intermediate_size, hidden_size]
            // Concatenate gate and up weights along intermediate_size axis: [num_experts, 2*intermediate_size, hidden_size]
            auto gate_up_weights = std::make_shared<ov::op::v0::Concat>(
                ov::OutputVector{gate_concat, up_concat}, 1);

            // Convert weights to f32 if needed (handling f16 weights)
            std::shared_ptr<ov::Node> gate_up_weights_f32 = gate_up_weights;
            if (gate_concat->get_element_type() != repeated_hidden_states->get_element_type()) {
                gate_up_weights_f32 = std::make_shared<ov::op::v0::Convert>(gate_up_weights, repeated_hidden_states->get_element_type());
            }

            auto gate_up_output = std::make_shared<ov::op::v0::MatMul>(
                repeated_hidden_states, gate_up_weights_f32, false, true);

            auto gate_up_matmul_time = std::chrono::high_resolution_clock::now();
            std::cout << "[MOE_TIMING] Gate+Up MatMul took: "
                      << std::chrono::duration_cast<std::chrono::microseconds>(gate_up_matmul_time - router_time).count()
                      << " µs" << std::endl;

            // 4. Split gate_up into gate and up parts along the intermediate_size axis
            auto split_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, 2);
            auto split_lengths = std::make_shared<ov::op::v0::Constant>(
                ov::element::i32, ov::Shape{2},
                std::vector<int32_t>{static_cast<int32_t>(config.intermediate_size), static_cast<int32_t>(config.intermediate_size)});
            auto gate_up_split = std::make_shared<ov::op::v1::VariadicSplit>(gate_up_output, split_axis, split_lengths);

            auto gate_proj = gate_up_split->output(0);  // [num_experts, batch_size, intermediate_size]
            auto up_proj = gate_up_split->output(1);    // [num_experts, batch_size, intermediate_size]

            // 5. Apply activation (SiLU/Swish): gated_output = up * silu(gate)
            auto gate_activated = std::make_shared<ov::op::v4::Swish>(gate_proj);
            auto gated_output = std::make_shared<ov::op::v1::Multiply>(up_proj, gate_activated);

            // 6. Down projection: output = bmm(gated_output, down_proj)
            // Convert down weights to f32 if needed (handling f16 weights)
            std::shared_ptr<ov::Node> down_concat_f32 = down_concat;
            if (down_concat->get_element_type() != gated_output->get_element_type()) {
                down_concat_f32 = std::make_shared<ov::op::v0::Convert>(down_concat, gated_output->get_element_type());
            }

            auto expert_outputs = std::make_shared<ov::op::v0::MatMul>(
                gated_output, down_concat_f32, false, true);
            // expert_outputs shape: [num_experts, batch_size, hidden_size]

            auto down_matmul_time = std::chrono::high_resolution_clock::now();
            std::cout << "[MOE_TIMING] Down projection MatMul took: "
                      << std::chrono::duration_cast<std::chrono::microseconds>(down_matmul_time - gate_up_matmul_time).count()
                      << " µs" << std::endl;

            // 7. Apply routing and combine expert outputs
            // Reshape expert_outputs to [num_experts, batch_size, hidden_size]
            // routing_weights shape: [batch_size, topk]
            // router_indices shape: [batch_size, topk]

            // Create expert mask using one_hot encoding
            auto expert_mask = std::make_shared<ov::op::v1::OneHot>(
                router_indices,
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, config.expert_num),
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, 1.0f),
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, 0.0f),
                2);  // [batch_size, topk, num_experts]

            // Transpose expert_outputs: [num_experts, batch_size, hidden_size] -> [batch_size, num_experts, hidden_size]
            auto transpose_perm = std::make_shared<ov::op::v0::Constant>(
                ov::element::i32, ov::Shape{3}, std::vector<int32_t>{1, 0, 2});
            auto expert_outputs_transposed = std::make_shared<ov::op::v1::Transpose>(expert_outputs, transpose_perm);

            // Apply expert mask and routing weights
            // expert_mask: [batch_size, topk, num_experts]
            // expert_outputs_transposed: [batch_size, num_experts, hidden_size]
            // routing_weights: [batch_size, topk]

            // Broadcast and multiply
            auto expert_mask_expanded = std::make_shared<ov::op::v0::Unsqueeze>(
                expert_mask, std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, 3));
            // expert_mask_expanded: [batch_size, topk, num_experts, 1]

            auto expert_outputs_expanded = std::make_shared<ov::op::v0::Unsqueeze>(
                expert_outputs_transposed, std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, 1));
            // expert_outputs_expanded: [batch_size, 1, num_experts, hidden_size]

            auto masked_outputs = std::make_shared<ov::op::v1::Multiply>(expert_mask_expanded, expert_outputs_expanded);
            // masked_outputs: [batch_size, topk, num_experts, hidden_size]

            // Sum over experts
            auto summed_outputs = std::make_shared<ov::op::v1::ReduceSum>(
                masked_outputs,
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int32_t>{2}),
                false);
            // summed_outputs: [batch_size, topk, hidden_size]

            // Apply routing weights
            auto routing_weights_expanded = std::make_shared<ov::op::v0::Unsqueeze>(
                routing_weights, std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, 2));
            // routing_weights_expanded: [batch_size, topk, 1]

            auto weighted_outputs = std::make_shared<ov::op::v1::Multiply>(summed_outputs, routing_weights_expanded);

            // Sum over topk to get final output
            auto final_output = std::make_shared<ov::op::v1::ReduceSum>(
                weighted_outputs,
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int32_t>{1}),
                false);
            // final_output: [batch_size, hidden_size]

            // Check whether the plugin accepts this decomposed structure
            if (transformation_callback(final_output)) {
                return false;
            }

            // Replace the final scatter node (last expert's output) with our decomposed MOE
            ov::replace_node(current ? current : last_node, final_output);
            register_new_node(final_output);

            final_output->set_friendly_name("moe_decomposed");

            auto end_time = std::chrono::high_resolution_clock::now();
            std::cout << "[MOE_TIMING] Routing and final operations took: "
                      << std::chrono::duration_cast<std::chrono::microseconds>(end_time - down_matmul_time).count()
                      << " µs" << std::endl;
            std::cout << "[MOE_TIMING] Total transformation time: "
                      << std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count()
                      << " µs" << std::endl;

            return true;
        } else {
            // Skip subsequent experts as they will be handled by the first expert's callback
            return false;
        }
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}

bool ov::pass::FuseMOE::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(FuseMOE);
    ov::pass::Manager manager(get_pass_config(), "FuseMOE");

    // manager.register_pass<ov::pass::EliminateSqueeze>();
    // Use the unified FuseMOE transformation
    manager.register_pass<ov::pass::PrintModel>("before_fuse_moe_pseudocode.cpp");
    manager.register_pass<ov::pass::FuseMOEUnified>();

    manager.run_passes(model);
    return false;
}

} // namespace pass
} // namespace ov
