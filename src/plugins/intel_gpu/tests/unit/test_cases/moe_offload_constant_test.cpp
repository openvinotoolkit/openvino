// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <numeric>
#include <optional>
#include "intel_gpu/runtime/internal_properties.hpp"
#include "test_utils.h"
#include "common_test_utils/test_assertions.hpp"
#include "openvino/core/memory_util.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/op/result.hpp"
#include "ov_ops/moe_compressed.hpp"
#include "plugin/ops/moe_offload_constant.hpp"
#include "test_utils.h"

using namespace ov::intel_gpu;

// A constant viewing a weights file, as produced by reading a model with mmap
static std::shared_ptr<ov::op::v0::Constant> make_file_backed_const(ov::element::Type type, const ov::Shape& shape) {
    const auto byte_size = ov::util::get_memory_size(type, ov::shape_size(shape));
    auto weights_file = std::make_shared<ov::AlignedBuffer>(byte_size);
    auto view = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(weights_file->get_ptr<char>(),
                                                                                       byte_size,
                                                                                       weights_file,
                                                                                       ov::create_base_descriptor(1, 0, weights_file));
    return std::make_shared<ov::op::v0::Constant>(type, shape, view);
}

// Helper: build a minimal MOECompressed node with 22 inputs (indices 0..21).
// Inputs 0-2 are Parameters (hidden_states, routing_weights, topk_indices).
// Inputs 3-11 are routed expert Constants (weight/scale/zp × gate/up/down).
// Inputs 12-21 are shared expert Constants.
// Returns the MOECompressed node and the constants at each input position.
struct MoETestGraph {
    std::shared_ptr<ov::op::internal::MOECompressed> moe_node;
    std::vector<std::shared_ptr<ov::op::v0::Constant>> constants;  // indices 3..21
    ov::ParameterVector parameters;

    static MoETestGraph build(size_t num_experts = 4,
                              size_t hidden_size = 128,
                              size_t inter_size = 256,
                              size_t group_size = 128,
                              ov::op::internal::MOE::Expert_type expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU,
                              bool file_backed_weights = true) {
        MoETestGraph g;

        const size_t top_k = 2;
        ov::op::internal::MOECompressed::Config config{};
        config.expert_type = expert_type;
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
        g.parameters.push_back(hidden);
        inputs.push_back(hidden->output(0));

        // 1: routing_weights [1, 1, top_k]
        auto routing = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 1, top_k});
        g.parameters.push_back(routing);
        inputs.push_back(routing->output(0));

        // 2: topk_indices [1, 1, top_k]
        auto topk_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1, 1, top_k});
        g.parameters.push_back(topk_idx);
        inputs.push_back(topk_idx->output(0));

        // Helper to create a constant with given shape and type
        auto make_const = [](ov::element::Type type, const ov::Shape& shape) {
            return std::make_shared<ov::op::v0::Constant>(type, shape, 0);
        };
        auto make_weight = [&](ov::element::Type type, const ov::Shape& shape) {
            return file_backed_weights ? make_file_backed_const(type, shape) : make_const(type, shape);
        };

        // Inputs 3-11: routed expert weights (gate/up/down × weight/scale/zp)
        // gate_w [E, inter, group_size, group_num]
        g.constants.push_back(make_weight(ov::element::u8, {num_experts, inter_size, group_size, group_num}));  // 3: w0_weight
        g.constants.push_back(make_const(ov::element::f16, {num_experts, inter_size, group_num, 1}));           // 4: w0_scale
        g.constants.push_back(make_const(ov::element::u8, {num_experts, inter_size, group_num, 1}));            // 5: w0_zp
        g.constants.push_back(make_weight(ov::element::u8, {num_experts, inter_size, group_size, group_num}));  // 6: w1_weight
        g.constants.push_back(make_const(ov::element::f16, {num_experts, inter_size, group_num, 1}));           // 7: w1_scale
        g.constants.push_back(make_const(ov::element::u8, {num_experts, inter_size, group_num, 1}));            // 8: w1_zp
        g.constants.push_back(make_weight(ov::element::u8, {num_experts, hidden_size, group_size, group_num2}));  // 9: w2_weight
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

    std::shared_ptr<ov::Model> to_model() const {
        return std::make_shared<ov::Model>(ov::OutputVector{moe_node}, parameters);
    }
};

TEST(moe_offload_constant, validates_resident_weight_budget) {
    auto g = MoETestGraph::build();
    auto result = std::make_shared<ov::op::v0::Result>(g.moe_node);
    ov::Model model(ov::ResultVector{result}, g.parameters);
    const uint64_t resident_bytes = get_model_resident_constant_bytes(model, 1);

    EXPECT_GT(resident_bytes, 0U);
    EXPECT_NO_THROW(validate_model_resident_constant_memory(model, 1, resident_bytes));
    EXPECT_THROW(validate_model_resident_constant_memory(model, 1, resident_bytes - 1), ov::Exception);
    EXPECT_NO_THROW(validate_model_resident_constant_memory(model, 0, 1));
    EXPECT_NO_THROW(validate_model_resident_constant_memory(model, 0, 0));
}

TEST(moe_offload_constant, offload_ratio_reduces_resident_weight_estimate) {
    auto g = MoETestGraph::build();
    auto result = std::make_shared<ov::op::v0::Result>(g.moe_node);
    ov::Model model(ov::ResultVector{result}, g.parameters);

    EXPECT_LT(get_model_resident_constant_bytes(model, 75), get_model_resident_constant_bytes(model, 0));
    EXPECT_LE(get_model_resident_constant_bytes(model, 100), get_model_resident_constant_bytes(model, 75));
    EXPECT_LT(get_model_resident_constant_bytes(model, 100), get_model_resident_constant_bytes(model, 0));
}

TEST(moe_offload_constant, non_moe_model_is_not_subject_to_resident_constant_check) {
    auto constant = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{4, 128}, 0);
    auto result = std::make_shared<ov::op::v0::Result>(constant);
    ov::Model model(ov::ResultVector{result}, ov::ParameterVector{});

    EXPECT_EQ(get_model_resident_constant_bytes(model, 0), 0U);
    EXPECT_NO_THROW(validate_model_resident_constant_memory(model, 0, 1));
}

// Test that routed expert constants (inputs 3-11) are classified as OffloadableRoutedExpert
static uint64_t sum_constant_bytes(const MoETestGraph& g, MoEConstantRole role) {
    uint64_t bytes = 0;
    for (const auto& constant : g.constants) {
        if (get_moe_constant_role(constant) == role)
            bytes += constant->get_byte_size();
    }
    return bytes;
}

// Test that routed expert constants (inputs 3-11) are classified as OffloadableRoutedExpert
TEST(moe_offload_constant, routed_expert_classification) {
    auto g = MoETestGraph::build();

    // Inputs 3..11 map to g.constants[0..8] (routed experts)
    for (size_t i = 0; i < 9; i++) {
        auto role = get_moe_constant_role(g.constants[i]);
        EXPECT_EQ(role, MoEConstantRole::OffloadableRoutedExpert)
            << "Constant at MOECompressed input " << (i + 3) << " should be OffloadableRoutedExpert";
    }
}

// Test that shared expert constants (inputs 12-21) are classified as AlwaysResidentSharedExpert
TEST(moe_offload_constant, shared_expert_classification) {
    auto g = MoETestGraph::build();

    // Inputs 12..21 map to g.constants[9..18] (shared experts)
    for (size_t i = 9; i < g.constants.size(); i++) {
        auto role = get_moe_constant_role(g.constants[i]);
        EXPECT_EQ(role, MoEConstantRole::AlwaysResidentSharedExpert)
            << "Constant at MOECompressed input " << (i + 3) << " should be AlwaysResidentSharedExpert";
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

// KEY TEST: Shared expert constants must NOT be classified as OffloadableRoutedExpert
// This is the regression test for the bug where shared constants were wrongly cropped.
TEST(moe_offload_constant, shared_expert_not_cropped_by_partial_upload) {
    auto g = MoETestGraph::build();

    // Shared expert constants (inputs 12-21) must NEVER be OffloadableRoutedExpert
    for (size_t i = 9; i < g.constants.size(); i++) {
        auto role = get_moe_constant_role(g.constants[i]);
        EXPECT_NE(role, MoEConstantRole::OffloadableRoutedExpert)
            << "Shared expert constant at input " << (i + 3)
            << " must NOT be classified as OffloadableRoutedExpert (would be wrongly cropped by partial upload)";
    }
}

TEST(moe_offload_constant, partial_upload_uses_preferred_allocation_type) {
    auto& engine = tests::get_test_engine();
    auto config = tests::get_test_default_config(engine);
    config.set_property(ov::intel_gpu::offload_ratio(20));
    auto graph = MoETestGraph::build();
    auto constant = graph.constants[0];
    const auto shape = constant->get_output_shape(0);
    const auto layout = cldnn::layout(shape,
                                      cldnn::element_type_to_data_type(constant->get_output_element_type(0)),
                                      cldnn::format::get_default_format(shape.size()));

    const auto desc = try_prepare_partial_upload(engine,
                                                 config,
                                                 constant,
                                                 shape,
                                                 layout.data_type,
                                                 layout.format,
                                                 layout);

    ASSERT_TRUE(desc.enabled);
    EXPECT_LT(desc.upload_bytes, layout.bytes_count());
    EXPECT_EQ(desc.memory->get_allocation_type(), engine.get_preferred_memory_allocation_type());
}

TEST(moe_offload_constant, gemm2_moe_is_not_otd_capable_and_not_discounted) {
    auto gemm3_graph = MoETestGraph::build(4, 128, 256, 128, ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU);
    EXPECT_TRUE(is_otd_capable_moe(gemm3_graph.moe_node.get()));

    auto gemm2_graph = MoETestGraph::build(4, 128, 256, 128, ov::op::internal::MOE::Expert_type::GEMM2_BIAS_SWIGLU_CLAMP);
    EXPECT_FALSE(is_otd_capable_moe(gemm2_graph.moe_node.get()));

    // For GEMM2, constants must be classified as NotOffloadableMoE, yet still recognized as MoE related
    for (auto& c : gemm2_graph.constants) {
        EXPECT_EQ(get_moe_constant_role(c), MoEConstantRole::NotOffloadableMoE);
        EXPECT_TRUE(is_moe_related_constant(c));
    }

    // Model with only GEMM2 MOE nodes should not trigger the OTD resident check (returns 0)
    auto result = std::make_shared<ov::op::v0::Result>(gemm2_graph.moe_node);
    ov::Model model(ov::ResultVector{result}, gemm2_graph.parameters);
    EXPECT_EQ(get_model_resident_constant_bytes(model, 50), 0U);

    // Partial upload must be disabled for GEMM2 constants
    auto& engine = tests::get_test_engine();
    auto config = tests::get_test_default_config(engine);
    config.set_property(ov::intel_gpu::offload_ratio(50));
    auto constant = gemm2_graph.constants[0];
    const auto shape = constant->get_output_shape(0);
    const auto layout = cldnn::layout(shape,
                                      cldnn::element_type_to_data_type(constant->get_output_element_type(0)),
                                      cldnn::format::get_default_format(shape.size()));
    const auto desc = try_prepare_partial_upload(engine,
                                                 config,
                                                 constant,
                                                 shape,
                                                 layout.data_type,
                                                 layout.format,
                                                 layout);
    EXPECT_FALSE(desc.enabled);
}

TEST(moe_offload_constant, shared_routed_constant_is_not_offloadable) {
    auto g = MoETestGraph::build();
    auto routed_const = g.constants[0];

    // Simulate sharing: attach another non-MoE consumer to the routed constant
    auto dummy_relu = std::make_shared<ov::op::v0::Relu>(routed_const);
    EXPECT_EQ(get_moe_constant_role(routed_const), MoEConstantRole::NotOffloadableRoutedExpert);
    EXPECT_TRUE(is_moe_related_constant(routed_const));

    // Must not be partially uploaded
    auto& engine = tests::get_test_engine();
    auto config = tests::get_test_default_config(engine);
    config.set_property(ov::intel_gpu::offload_ratio(50));
    const auto shape = routed_const->get_output_shape(0);
    const auto layout = cldnn::layout(shape,
                                      cldnn::element_type_to_data_type(routed_const->get_output_element_type(0)),
                                      cldnn::format::get_default_format(shape.size()));
    const auto desc = try_prepare_partial_upload(engine,
                                                 config,
                                                 routed_const,
                                                 shape,
                                                 layout.data_type,
                                                 layout.format,
                                                 layout);
    EXPECT_FALSE(desc.enabled);
}

TEST(moe_offload_constant, resident_fallback_keeps_sibling_constants_fully_resident) {
    auto& engine = tests::get_test_engine();
    auto config = tests::get_test_default_config(engine);
    config.set_property(ov::intel_gpu::offload_ratio(75));

    for (bool tensor_weight : {false, true}) {
        SCOPED_TRACE(tensor_weight);
        auto graph = MoETestGraph::build();
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(graph.moe_node)};
        if (tensor_weight) {
            auto weight = std::make_shared<ov::op::v0::Parameter>(graph.constants[0]->get_element_type(), graph.constants[0]->get_shape());
            graph.moe_node->input(3).replace_source_output(weight);
            graph.parameters.push_back(weight);
        } else {
            auto consumer = std::make_shared<ov::op::v0::Relu>(graph.constants[0]);
            results.push_back(std::make_shared<ov::op::v0::Result>(consumer));
        }
        ov::Model model(results, graph.parameters);
        EXPECT_FALSE(is_moe_otd_eligible(graph.moe_node.get()));
        EXPECT_EQ(get_model_resident_constant_bytes(model, 75), get_model_resident_constant_bytes(model, 0));
        EXPECT_EQ(get_model_resident_constant_bytes(model, 100), get_model_resident_constant_bytes(model, 0));

        for (size_t index = 1; index < 9; ++index) {
            const auto& constant = graph.constants[index];
            EXPECT_FALSE(is_moe_constant_offloadable(constant));
            const auto shape = constant->get_shape();
            const auto layout =
                cldnn::layout(shape, cldnn::element_type_to_data_type(constant->get_element_type()), cldnn::format::get_default_format(shape.size()));
            const auto desc = try_prepare_partial_upload(engine, config, constant, shape, layout.data_type, layout.format, layout);
            EXPECT_FALSE(desc.enabled);
        }
    }
}

TEST(moe_offload_constant, shared_routed_weights_propagate_resident_fallback) {
    auto first = MoETestGraph::build();
    auto second = MoETestGraph::build();
    second.moe_node->input(3).replace_source_output(first.constants[0]);

    EXPECT_TRUE(is_moe_otd_eligible(first.moe_node.get()));
    EXPECT_TRUE(is_moe_otd_eligible(second.moe_node.get()));
    EXPECT_TRUE(is_moe_constant_offloadable(first.constants[0]));

    auto weight = std::make_shared<ov::op::v0::Parameter>(second.constants[3]->get_element_type(), second.constants[3]->get_shape());
    second.moe_node->input(6).replace_source_output(weight);
    second.parameters.push_back(weight);
    auto parameters = first.parameters;
    parameters.insert(parameters.end(), second.parameters.begin(), second.parameters.end());
    ov::Model model(ov::ResultVector{std::make_shared<ov::op::v0::Result>(first.moe_node), std::make_shared<ov::op::v0::Result>(second.moe_node)}, parameters);

    EXPECT_FALSE(is_moe_otd_eligible(first.moe_node.get()));
    EXPECT_FALSE(is_moe_otd_eligible(second.moe_node.get()));
    EXPECT_FALSE(is_moe_constant_offloadable(first.constants[0]));
    EXPECT_FALSE(is_moe_constant_offloadable(first.constants[1]));
    EXPECT_FALSE(is_moe_constant_offloadable(second.constants[1]));
    EXPECT_EQ(get_model_resident_constant_bytes(model, 75), get_model_resident_constant_bytes(model, 0));
}

namespace {
PartialUploadDesc prepare_partial_upload(const std::shared_ptr<ov::op::v0::Constant>& constant, size_t offload_ratio) {
    auto& engine = tests::get_test_engine();
    auto config = tests::get_test_default_config(engine);
    config.set_property(ov::intel_gpu::offload_ratio(offload_ratio));
    const auto shape = constant->get_output_shape(0);
    const auto layout =
        cldnn::layout(shape, cldnn::element_type_to_data_type(constant->get_output_element_type(0)), cldnn::format::get_default_format(shape.size()));
    return try_prepare_partial_upload(engine, config, constant, shape, layout.data_type, layout.format, layout);
}
}  // namespace

TEST(moe_offload_constant, slot_count_of_file_backed_weights) {
    auto graph = MoETestGraph::build(/*num_experts=*/8);

    EXPECT_EQ(get_moe_offload_slot_count(*graph.moe_node, 0), 0U);
    EXPECT_EQ(get_moe_offload_slot_count(*graph.moe_node, 25), 6U);
    EXPECT_EQ(get_moe_offload_slot_count(*graph.moe_node, 99), 1U);
    EXPECT_EQ(get_moe_offload_slot_count(*graph.moe_node, 100), 1U);
}

// Weights recomputed by the transformations (converted, transposed, ...) have no weights file to be streamed from
TEST(moe_offload_constant, no_offload_of_weights_not_backed_by_file) {
    auto graph = MoETestGraph::build(4, 128, 256, 128, ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU, /*file_backed_weights=*/false);
    ov::Model model(ov::ResultVector{std::make_shared<ov::op::v0::Result>(graph.moe_node)}, graph.parameters);

    EXPECT_EQ(get_moe_offload_slot_count(*graph.moe_node, 50), 0U);
    EXPECT_FALSE(prepare_partial_upload(graph.constants[0], 50).enabled);
    EXPECT_EQ(get_model_resident_constant_bytes(model, 50), get_model_resident_constant_bytes(model, 0));
}

// Only the routed weights are offloaded: scales, zero points and shared experts stay fully resident
TEST(moe_offload_constant, partial_upload_only_for_routed_weights) {
    auto graph = MoETestGraph::build();

    for (size_t i = 0; i < graph.constants.size(); i++) {
        const auto input_index = i + 3;
        const bool is_routed_weight = input_index == 3 || input_index == 6 || input_index == 9;
        EXPECT_EQ(prepare_partial_upload(graph.constants[i], 50).enabled, is_routed_weight) << "MOECompressed input " << input_index;
    }
}

// NNCF int3 models share one scalar zero point among all the layers: only the routed weights are streamed,
// so the resident scales and zero points may have any other consumer
TEST(moe_offload_constant, zero_point_shared_with_other_layers_does_not_block_offload) {
    auto graph = MoETestGraph::build(/*num_experts=*/8);
    auto zp = std::make_shared<ov::op::v0::Constant>(ov::element::u8, ov::Shape{}, 4);
    for (const size_t index : {5, 8, 11}) {
        graph.moe_node->input(index).replace_source_output(zp);
    }
    auto other_layer = std::make_shared<ov::op::v0::Relu>(zp);

    EXPECT_TRUE(is_moe_otd_eligible(graph.moe_node.get()));
    EXPECT_EQ(get_moe_offload_slot_count(*graph.moe_node, 50), 4U);
    EXPECT_TRUE(prepare_partial_upload(graph.constants[0], 50).enabled);
    EXPECT_FALSE(is_moe_constant_offloadable(zp));
}

// The slots copy each expert's slice of a scale / zero point: one shared by the experts but not a scalar cannot be sliced,
// so the layer stays resident at compilation instead of failing at the first inference
TEST(moe_offload_constant, scale_without_expert_slices_keeps_layer_resident) {
    auto graph = MoETestGraph::build();
    graph.moe_node->input(4).replace_source_output(std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 256, 1, 1}, 0));
    ov::Model model(ov::ResultVector{std::make_shared<ov::op::v0::Result>(graph.moe_node)}, graph.parameters);

    EXPECT_EQ(get_moe_offload_slot_count(*graph.moe_node, 50), 0U);
    EXPECT_FALSE(prepare_partial_upload(graph.constants[0], 50).enabled);
    EXPECT_EQ(get_model_resident_constant_bytes(model, 50), get_model_resident_constant_bytes(model, 0));
}

// OFFLOAD_RATIO that cannot offload any MoE layer of the model is an error, not a silent full residency
TEST(moe_offload_constant, no_offloadable_moe_layer_is_rejected) {
    auto in_memory = MoETestGraph::build(4, 128, 256, 128, ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU, /*file_backed_weights=*/false);
    ov::Model model(ov::ResultVector{std::make_shared<ov::op::v0::Result>(in_memory.moe_node)}, in_memory.parameters);
    OV_EXPECT_THROW(validate_model_resident_constant_memory(model, 50, 0),
                    ov::Exception,
                    testing::AllOf(testing::HasSubstr("no MoE layer can be offloaded"), testing::HasSubstr("not unmodified views of the weights file")));
    EXPECT_NO_THROW(validate_model_resident_constant_memory(model, 0, 0));

    // One offloadable layer is enough
    auto file_backed = MoETestGraph::build();
    auto parameters = in_memory.parameters;
    parameters.insert(parameters.end(), file_backed.parameters.begin(), file_backed.parameters.end());
    ov::Model mixed(ov::ResultVector{std::make_shared<ov::op::v0::Result>(in_memory.moe_node), std::make_shared<ov::op::v0::Result>(file_backed.moe_node)},
                    parameters);
    EXPECT_NO_THROW(validate_model_resident_constant_memory(mixed, 50, 0));
}

// The slices are streamed from ov::weights_path: it must hold the data of the constant at its offset
TEST(moe_offload_constant, weights_file_offset_is_checked_against_the_file) {
    const auto path = std::filesystem::temp_directory_path() / "moe_offload_constant_weights_file_offset.bin";
    const auto write_file = [&](const std::vector<char>& data) {
        std::ofstream file(path, std::ios::binary);
        file.write(data.data(), static_cast<std::streamsize>(data.size()));
    };
    std::vector<char> file_data(256);
    std::iota(file_data.begin(), file_data.end(), char{0});
    write_file(file_data);

    // Bytes 64..191 of the file, viewed as read with mmap
    constexpr size_t offset = 64, size = 128;
    auto weights_file = std::make_shared<ov::AlignedBuffer>(file_data.size());
    std::copy(file_data.begin(), file_data.end(), weights_file->get_ptr<char>());
    auto view = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(weights_file->get_ptr<char>() + offset,
                                                                                       size,
                                                                                       weights_file,
                                                                                       ov::create_base_descriptor(1, 0, weights_file));
    const ov::op::v0::Constant constant(ov::element::u8, ov::Shape{size}, view);
    EXPECT_EQ(get_weights_file_offset(constant, path), offset);

    // The same bytes read without mmap into a buffer of their own: the offset is in the IR attribute
    auto region = std::make_shared<ov::AlignedBuffer>(size);
    std::copy_n(file_data.begin() + offset, size, region->get_ptr<char>());
    auto region_view =
        std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(region->get_ptr<char>(), size, region, ov::create_base_descriptor(2, 0, region));
    ov::op::v0::Constant read_constant(ov::element::u8, ov::Shape{size}, region_view);
    OV_EXPECT_THROW(get_weights_file_offset(read_constant, path), ov::Exception, testing::HasSubstr("cannot be located"));
    read_constant.get_rt_info()[ov::WeightlessCacheAttribute::get_type_info_static()] = ov::WeightlessCacheAttribute(size, offset, ov::element::u8);
    EXPECT_EQ(get_weights_file_offset(read_constant, path), offset);

    const ov::op::v0::Constant in_memory(ov::element::u8, ov::Shape{size}, 0);
    OV_EXPECT_THROW(get_weights_file_offset(in_memory, path), ov::Exception, testing::HasSubstr("cannot be located"));

    // Another file of the same size
    write_file(std::vector<char>(file_data.size(), 0));
    OV_EXPECT_THROW(get_weights_file_offset(constant, path), ov::Exception, testing::HasSubstr("does not contain the data of constant"));
    std::filesystem::remove(path);
}

// The estimate keeps the scales and zero points fully resident and adds their slot copies: slots x bytes per expert
TEST(moe_offload_constant, resident_estimate_counts_scale_and_zero_point_slots) {
    for (const bool scalar_zp : {false, true}) {
        SCOPED_TRACE(scalar_zp);
        auto graph = MoETestGraph::build(/*num_experts=*/4);
        if (scalar_zp) {
            auto zp = std::make_shared<ov::op::v0::Constant>(ov::element::u8, ov::Shape{}, 4);
            for (const size_t index : {5, 8, 11}) {
                graph.moe_node->input(index).replace_source_output(zp);
            }
        }
        ov::Model model(ov::ResultVector{std::make_shared<ov::op::v0::Result>(graph.moe_node)}, graph.parameters);
        uint64_t weights = 0;
        uint64_t params = 0;
        for (size_t index = 3; index <= 11; ++index) {
            const auto* constant = ov::as_type<ov::op::v0::Constant>(graph.moe_node->get_input_node_ptr(index));
            ((index % 3 == 0) ? weights : params) += ov::shape_size(constant->get_shape()) > 1 ? constant->get_byte_size() : 0;
        }
        // Ratio 50 keeps 2 of the 4 experts: half of the weights go, half of the per-expert scales and zero points come as slots
        EXPECT_EQ(get_model_resident_constant_bytes(model, 0) - get_model_resident_constant_bytes(model, 50), weights / 2 - params / 2);
    }
}

// Layers sharing a routed weight read the same partially uploaded constant: one that stays resident keeps the other resident too
TEST(moe_offload_constant, layer_sharing_weights_with_a_resident_layer_stays_resident) {
    auto first = MoETestGraph::build();
    auto second = MoETestGraph::build();
    second.moe_node->input(3).replace_source_output(first.constants[0]);
    const auto& up_weight = second.constants[3];
    second.moe_node->input(6).replace_source_output(std::make_shared<ov::op::v0::Constant>(up_weight->get_element_type(), up_weight->get_shape(), 0));

    std::string reason;
    EXPECT_FALSE(is_moe_otd_eligible(first.moe_node.get(), &reason));
    EXPECT_EQ(reason, "the routed weights are not unmodified views of the weights file");
    EXPECT_EQ(get_moe_offload_slot_count(*first.moe_node, 50), 0U);
    EXPECT_EQ(get_moe_offload_slot_count(*second.moe_node, 50), 0U);
    for (const auto& constant : {first.constants[0], first.constants[3], second.constants[6]}) {
        EXPECT_FALSE(is_moe_constant_offloadable(constant));
        EXPECT_FALSE(prepare_partial_upload(constant, 50).enabled);
    }
}

TEST(moe_offload_constant, auto_ratio_no_moe_model_resolves_zero) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 16});
    auto model = std::make_shared<ov::Model>(ov::OutputVector{input}, ov::ParameterVector{input});

    EXPECT_EQ(resolve_auto_offload_ratio_for_budget(*model, 1), 0U);
}

TEST(moe_offload_constant, auto_ratio_zero_budget_resolves_zero) {
    auto g = MoETestGraph::build();
    EXPECT_EQ(resolve_auto_offload_ratio_for_budget(*g.to_model(), 0), 0U);
}

TEST(moe_offload_constant, auto_ratio_non_positive_budget_for_moe_resolves_zero) {
    auto g = MoETestGraph::build();
    const uint64_t fixed_bytes = sum_constant_bytes(g, MoEConstantRole::AlwaysResidentSharedExpert);
    // Set budget such that budget * 0.85 <= fixed_bytes (i.e. budget_for_moe <= 0)
    const uint64_t budget = fixed_bytes;

    EXPECT_EQ(resolve_auto_offload_ratio_for_budget(*g.to_model(), budget), 0U);
}

TEST(moe_offload_constant, auto_ratio_all_moe_weights_fit_resolves_zero) {
    auto g = MoETestGraph::build();

    EXPECT_EQ(resolve_auto_offload_ratio_for_budget(*g.to_model(), uint64_t{1} << 40), 0U);
}

TEST(moe_offload_constant, auto_ratio_uses_routed_expert_bytes_for_resident_fraction) {
    auto g = MoETestGraph::build();
    const uint64_t routed_bytes = sum_constant_bytes(g, MoEConstantRole::OffloadableRoutedExpert);
    const uint64_t fixed_bytes = sum_constant_bytes(g, MoEConstantRole::AlwaysResidentSharedExpert);
    const uint64_t budget = static_cast<uint64_t>((static_cast<double>(fixed_bytes) + 0.5 * static_cast<double>(routed_bytes)) / 0.85);

    EXPECT_EQ(resolve_auto_offload_ratio_for_budget(*g.to_model(), budget), 50U);
}

TEST(moe_offload_constant, auto_ratio_calculates_boundary_ratio_75) {
    auto g = MoETestGraph::build();
    const uint64_t routed_bytes = sum_constant_bytes(g, MoEConstantRole::OffloadableRoutedExpert);
    const uint64_t fixed_bytes = sum_constant_bytes(g, MoEConstantRole::AlwaysResidentSharedExpert);
    // 25% resident -> 75% offload
    const uint64_t budget = static_cast<uint64_t>((static_cast<double>(fixed_bytes) + 0.25 * static_cast<double>(routed_bytes)) / 0.85);

    EXPECT_EQ(resolve_auto_offload_ratio_for_budget(*g.to_model(), budget), 75U);
}

TEST(moe_offload_constant, auto_ratio_calculates_quarter_offload_25) {
    auto g = MoETestGraph::build();
    const uint64_t routed_bytes = sum_constant_bytes(g, MoEConstantRole::OffloadableRoutedExpert);
    const uint64_t fixed_bytes = sum_constant_bytes(g, MoEConstantRole::AlwaysResidentSharedExpert);
    // 75% resident -> 25% offload
    const uint64_t budget = static_cast<uint64_t>((static_cast<double>(fixed_bytes) + 0.75 * static_cast<double>(routed_bytes)) / 0.85);

    EXPECT_EQ(resolve_auto_offload_ratio_for_budget(*g.to_model(), budget), 25U);
}

TEST(moe_offload_constant, auto_ratio_counts_shared_expert_bytes_as_fixed_weights) {
    auto g = MoETestGraph::build();
    const uint64_t routed_bytes = sum_constant_bytes(g, MoEConstantRole::OffloadableRoutedExpert);
    const uint64_t budget = static_cast<uint64_t>(static_cast<double>(routed_bytes) / 0.85);

    EXPECT_GT(resolve_auto_offload_ratio_for_budget(*g.to_model(), budget), 0U);
}

TEST(moe_offload_constant, resolve_auto_offload_ratio_with_engine_no_moe) {
    auto& engine = tests::get_test_engine();
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 16});
    auto model = std::make_shared<ov::Model>(ov::OutputVector{input}, ov::ParameterVector{input});

    EXPECT_EQ(resolve_auto_offload_ratio(*model, engine), 0U);
}

TEST(moe_offload_constant, resolve_auto_offload_ratio_with_engine_small_model_fits) {
    auto& engine = tests::get_test_engine();
    auto g = MoETestGraph::build();

    // The small test graph is << device memory, so all weights fit -> resolves to 0
    EXPECT_EQ(resolve_auto_offload_ratio(*g.to_model(), engine), 0U);
}

TEST(moe_offload_constant, mixed_consumer_routed_and_shared_is_not_routed_eligible) {
    // When a constant is connected to both routed expert and shared expert inputs,
    // it must not be classified as OffloadableRoutedExpert (must not be cropped by partial upload),
    // and must be classified as AlwaysResidentSharedExpert.
    auto g = MoETestGraph::build();
    // g.constants[0] is w0_weight (connected to MOECompressed input 3)
    // Also connect g.constants[0] to input 12 (shared) by creating an additional consumer input
    auto c = g.constants[0];

    // Build another MOECompressed node using c at a shared input index (12)
    ov::OutputVector inputs;
    for (const auto& p : g.parameters) {
        inputs.push_back(p->output(0));
    }
    // inputs 3..11: other constants
    for (size_t i = 0; i < 9; ++i) {
        inputs.push_back(g.constants[i]->output(0));
    }
    // input 12: use c (which was already input 3 in g.moe_node)
    inputs.push_back(c->output(0));
    for (size_t i = 10; i < g.constants.size(); ++i) {
        inputs.push_back(g.constants[i]->output(0));
    }
    auto extra_moe = std::make_shared<ov::op::internal::MOECompressed>(inputs, g.moe_node->get_config());

    EXPECT_NE(get_moe_constant_role(c), MoEConstantRole::OffloadableRoutedExpert);
    EXPECT_EQ(get_moe_constant_role(c), MoEConstantRole::AlwaysResidentSharedExpert);
}

TEST(moe_offload_constant, mixed_consumer_routed_and_non_moe_is_not_routed_eligible) {
    // When a constant is connected to a routed expert input and also to a non-MoE op (Result),
    // it requires the full tensor and must not be classified as RoutedExpert.
    auto g = MoETestGraph::build();
    auto c = g.constants[0];  // g.constants[0] is w0_weight (connected to MOECompressed input 3)

    auto non_moe_consumer = std::make_shared<ov::op::v0::Result>(c);

    EXPECT_NE(get_moe_constant_role(c), MoEConstantRole::OffloadableRoutedExpert);
    EXPECT_EQ(get_moe_constant_role(c), MoEConstantRole::NotOffloadableRoutedExpert);
}

TEST(moe_offload_constant, auto_ratio_mixed_consumer_constant_counted_as_fixed_weight) {
    auto g = MoETestGraph::build();
    auto mixed_c = g.constants[0];  // g.constants[0] is w0_weight (connected to MOECompressed input 3)

    // Connect mixed_c to a non-MoE consumer so it becomes ineligible for routed offload.
    auto extra_res = std::make_shared<ov::op::v0::Result>(mixed_c);
    auto moe_res = std::make_shared<ov::op::v0::Result>(g.moe_node);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{moe_res, extra_res}, g.parameters);

    EXPECT_EQ(get_moe_constant_role(mixed_c), MoEConstantRole::NotOffloadableRoutedExpert);

    const uint64_t routed_bytes = sum_constant_bytes(g, MoEConstantRole::OffloadableRoutedExpert);
    const uint64_t shared_bytes = sum_constant_bytes(g, MoEConstantRole::AlwaysResidentSharedExpert);
    const uint64_t mixed_c_bytes = mixed_c->get_byte_size();
    // mixed_c is now non-routed (NotMoE), so it is counted as fixed weight along with shared experts:
    const uint64_t w_fixed = shared_bytes + mixed_c_bytes;

    // Set budget such that w_fixed is covered and 50% of remaining routed_bytes fits
    const uint64_t budget = static_cast<uint64_t>(
        (static_cast<double>(w_fixed) + 0.5 * static_cast<double>(routed_bytes)) / 0.85);

    EXPECT_EQ(resolve_auto_offload_ratio_for_budget(*model, budget), 50U);
}

TEST(moe_offload_constant, estimate_available_tracked_device_memory_calculates_difference) {
    auto& engine = tests::get_test_engine();
    uint64_t used_bytes = 0;
    for (const auto& stat : engine.get_memory_statistics()) {
        used_bytes += stat.second;
    }

    // When upper_bound exceeds used_bytes, difference is returned
    const uint64_t headroom = 1024 * 1024;
    EXPECT_EQ(estimate_available_tracked_device_memory_bytes(engine, used_bytes + headroom), headroom);

    // When upper_bound equals or is less than used_bytes, 0 is returned
    EXPECT_EQ(estimate_available_tracked_device_memory_bytes(engine, used_bytes), 0U);
    if (used_bytes > 0) {
        EXPECT_EQ(estimate_available_tracked_device_memory_bytes(engine, used_bytes - 1), 0U);
    }
}

#if defined(__linux__)
TEST(moe_offload_constant, query_available_ram_bytes_returns_positive) {
    EXPECT_GT(query_available_ram_bytes(), 0U);
}
#endif
