// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "ov_ops/moe_compressed.hpp"
#include "plugin/ops/moe_offload_constant.hpp"

using namespace ov::intel_gpu;

// Helper: build a minimal MOECompressed node with 22 inputs (indices 0..21).
// Inputs 0-2 are Parameters (hidden_states, routing_weights, topk_indices).
// Inputs 3-11 are routed expert Constants (weight/scale/zp × gate/up/down).
// Inputs 12-21 are shared expert Constants.
// Returns the MOECompressed node and the constants at each input position.
struct MoETestGraph {
    std::shared_ptr<ov::op::internal::MOECompressed> moe_node;
    std::vector<std::shared_ptr<ov::op::v0::Constant>> constants;  // indices 3..21

    static MoETestGraph build(size_t num_experts = 4, size_t hidden_size = 128, size_t inter_size = 256, size_t group_size = 128) {
        MoETestGraph g;

        const size_t top_k = 2;
        ov::op::internal::MOECompressed::Config config{};
        config.hidden_size = hidden_size;
        config.inter_size = inter_size;
        config.num_expert = num_experts;
        config.num_shared_expert = 1;
        config.top_k = top_k;
        config.group_size = group_size;
        config.has_zp = true;
        config.out_type = ov::element::f16;

        size_t group_num = hidden_size / group_size;
        size_t group_num2 = inter_size / group_size;

        // Build input OutputVector (22 inputs)
        ov::OutputVector inputs;

        // 0: hidden_states [1, 1, hidden_size]
        auto hidden = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 1, hidden_size});
        inputs.push_back(hidden->output(0));

        // 1: routing_weights [1, 1, top_k]
        auto routing = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 1, top_k});
        inputs.push_back(routing->output(0));

        // 2: topk_indices [1, 1, top_k]
        auto topk_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1, 1, top_k});
        inputs.push_back(topk_idx->output(0));

        // Helper to create a constant with given shape and type
        auto make_const = [](ov::element::Type type, const ov::Shape& shape) {
            return std::make_shared<ov::op::v0::Constant>(type, shape, 0);
        };

        // Inputs 3-11: routed expert weights (gate/up/down × weight/scale/zp)
        // gate_w [E, inter, group_size, group_num]
        g.constants.push_back(make_const(ov::element::u8, {num_experts, inter_size, group_size, group_num}));   // 3: w0_weight
        g.constants.push_back(make_const(ov::element::f16, {num_experts, inter_size, group_num, 1}));           // 4: w0_scale
        g.constants.push_back(make_const(ov::element::u8, {num_experts, inter_size, group_num, 1}));            // 5: w0_zp
        g.constants.push_back(make_const(ov::element::u8, {num_experts, inter_size, group_size, group_num}));   // 6: w1_weight
        g.constants.push_back(make_const(ov::element::f16, {num_experts, inter_size, group_num, 1}));           // 7: w1_scale
        g.constants.push_back(make_const(ov::element::u8, {num_experts, inter_size, group_num, 1}));            // 8: w1_zp
        g.constants.push_back(make_const(ov::element::u8, {num_experts, hidden_size, group_size, group_num2})); // 9: w2_weight
        g.constants.push_back(make_const(ov::element::f16, {num_experts, hidden_size, group_num2, 1}));         // 10: w2_scale
        g.constants.push_back(make_const(ov::element::u8, {num_experts, hidden_size, group_num2, 1}));          // 11: w2_zp

        // Inputs 12-21: shared expert weights (same structure but num_experts=1 for shared)
        g.constants.push_back(make_const(ov::element::u8, {1, inter_size, group_size, group_num}));   // 12: shared_gate_weight
        g.constants.push_back(make_const(ov::element::f16, {1, inter_size, group_num, 1}));           // 13: shared_gate_scale
        g.constants.push_back(make_const(ov::element::u8, {1, inter_size, group_num, 1}));            // 14: shared_gate_zp
        g.constants.push_back(make_const(ov::element::u8, {1, inter_size, group_size, group_num}));   // 15: shared_up_weight
        g.constants.push_back(make_const(ov::element::f16, {1, inter_size, group_num, 1}));           // 16: shared_up_scale
        g.constants.push_back(make_const(ov::element::u8, {1, inter_size, group_num, 1}));            // 17: shared_up_zp
        g.constants.push_back(make_const(ov::element::u8, {1, hidden_size, group_size, group_num2})); // 18: shared_down_weight
        g.constants.push_back(make_const(ov::element::f16, {1, hidden_size, group_num2, 1}));         // 19: shared_down_scale
        g.constants.push_back(make_const(ov::element::u8, {1, hidden_size, group_num2, 1}));          // 20: shared_down_zp
        // Input 21 is sometimes shared_gate_gate (for gating the shared expert output)
        g.constants.push_back(make_const(ov::element::f16, {1, hidden_size}));                        // 21: shared_gate_gate_weight

        for (auto& c : g.constants) {
            inputs.push_back(c->output(0));
        }

        g.moe_node = std::make_shared<ov::op::internal::MOECompressed>(inputs, config);
        return g;
    }
};

// Test that routed expert constants (inputs 3-11) are classified as RoutedExpert
TEST(moe_offload_constant, routed_expert_classification) {
    auto g = MoETestGraph::build();

    // Inputs 3..11 map to g.constants[0..8] (routed experts)
    for (size_t i = 0; i < 9; i++) {
        auto role = get_moe_constant_role(g.constants[i]);
        EXPECT_EQ(role, MoEConstantRole::RoutedExpert)
            << "Constant at MOECompressed input " << (i + 3) << " should be RoutedExpert";
    }
}

// Test that shared expert constants (inputs 12-21) are classified as SharedExpert
TEST(moe_offload_constant, shared_expert_classification) {
    auto g = MoETestGraph::build();

    // Inputs 12..21 map to g.constants[9..18] (shared experts)
    for (size_t i = 9; i < g.constants.size(); i++) {
        auto role = get_moe_constant_role(g.constants[i]);
        EXPECT_EQ(role, MoEConstantRole::SharedExpert)
            << "Constant at MOECompressed input " << (i + 3) << " should be SharedExpert";
    }
}

// Test that a constant NOT connected to MOECompressed is NotMoE
TEST(moe_offload_constant, unconnected_constant_is_not_moe) {
    auto c = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{4, 128}, 0);
    EXPECT_EQ(get_moe_constant_role(c), MoEConstantRole::NotMoE);
}

// Test is_moe_related_constant for both routed and shared
TEST(moe_offload_constant, is_moe_related_covers_both) {
    auto g = MoETestGraph::build();

    // All constants connected to MOECompressed should be "MoE related"
    for (auto& c : g.constants) {
        EXPECT_TRUE(is_moe_related_constant(c));
    }

    // Unconnected constant is not MoE related
    auto unconnected = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{4, 128}, 0);
    EXPECT_FALSE(is_moe_related_constant(unconnected));
}

// KEY TEST: Shared expert constants must NOT be classified as RoutedExpert
// This is the regression test for the bug where shared constants were wrongly cropped.
TEST(moe_offload_constant, shared_expert_not_cropped_by_partial_upload) {
    auto g = MoETestGraph::build();

    // Shared expert constants (inputs 12-21) must NEVER be RoutedExpert
    for (size_t i = 9; i < g.constants.size(); i++) {
        auto role = get_moe_constant_role(g.constants[i]);
        EXPECT_NE(role, MoEConstantRole::RoutedExpert)
            << "Shared expert constant at input " << (i + 3)
            << " must NOT be classified as RoutedExpert (would be wrongly cropped by partial upload)";
    }
}
