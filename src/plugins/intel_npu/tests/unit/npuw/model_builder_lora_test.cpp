// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "model_builder.hpp"
#include "npuw_transformations/lora_stateful_to_stateless.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/core.hpp"
#include "ov_ops/lora_subgraph.hpp"
#include "transformations/common_optimizations/lora_subgraph_fusion.hpp"

namespace {

ov::test::npuw::LLMConfig make_lora_config(bool stateful = false) {
    ov::test::npuw::LLMConfig config;
    config.num_layers = 1;
    config.hidden_size = 8;
    config.num_heads = 2;
    config.num_kv_heads = 1;
    config.head_dim = 4;
    config.intermediate_size = 16;
    config.vocab_size = 32;
    config.precision = ov::element::f16;
    config.use_kv_cache = false;
    config.internal_position_ids = true;
    config.lm_head_weight = {};
    config.lora_rank = 2;
    config.lora_targets = {"q_proj", "v_proj"};
    config.lora_stateful = stateful;
    return config;
}

std::shared_ptr<ov::Model> build_lora_model(bool stateful = false) {
    ov::test::npuw::ModelBuilder builder;
    auto model = builder.build_llm(make_lora_config(stateful));
    model->validate_nodes_and_infer_types();
    return model;
}

std::map<std::string, ov::Output<ov::Node>> inputs_by_name(const std::shared_ptr<ov::Model>& model) {
    std::map<std::string, ov::Output<ov::Node>> inputs;
    for (const auto& input : model->inputs()) {
        inputs.emplace(input.get_any_name(), input);
    }
    return inputs;
}

void expect_input_shape(const std::map<std::string, ov::Output<ov::Node>>& inputs,
                        const std::string& name,
                        const ov::PartialShape& shape,
                        ov::element::Type type = ov::element::f16) {
    auto it = inputs.find(name);
    ASSERT_NE(it, inputs.end()) << name;
    EXPECT_EQ(it->second.get_partial_shape(), shape);
    EXPECT_EQ(it->second.get_element_type(), type);
}

ov::Tensor make_i64_tensor(const ov::Shape& shape, const std::vector<int64_t>& values) {
    ov::Tensor tensor(ov::element::i64, shape);
    std::copy(values.begin(), values.end(), tensor.data<int64_t>());
    return tensor;
}

ov::Tensor make_f32_tensor(const ov::Shape& shape, float value) {
    ov::Tensor tensor(ov::element::f32, shape);
    std::fill_n(tensor.data<float>(), ov::shape_size(shape), value);
    return tensor;
}

ov::Tensor make_float_tensor(ov::element::Type type, const ov::Shape& shape, float value) {
    if (type == ov::element::f16) {
        ov::Tensor tensor(type, shape);
        std::fill_n(tensor.data<ov::float16>(), ov::shape_size(shape), ov::float16(value));
        return tensor;
    }
    return make_f32_tensor(shape, value);
}

void set_inputs(ov::InferRequest& request, const ov::CompiledModel& compiled_model, float lora_value) {
    for (const auto& input : compiled_model.inputs()) {
        const auto& name = input.get_any_name();
        if (name == "input_ids") {
            request.set_tensor(input, make_i64_tensor({1, 2}, {1, 2}));
        } else if (name == "attention_mask") {
            request.set_tensor(input, make_i64_tensor({1, 2}, {1, 1}));
        } else if (name.find(".MatMul.alpha") != std::string::npos) {
            request.set_tensor(input,
                               make_f32_tensor(input.get_partial_shape().to_shape(), lora_value == 0.0f ? 0.0f : 1.0f));
        } else if (name.find("lora_state_") != std::string::npos) {
            request.set_tensor(
                input,
                make_float_tensor(input.get_element_type(), input.get_partial_shape().to_shape(), lora_value));
        }
    }
}

std::vector<float> read_f32_output(ov::InferRequest& request, const ov::Output<const ov::Node>& output) {
    const auto tensor = request.get_tensor(output);
    const auto size = ov::shape_size(tensor.get_shape());
    if (tensor.get_element_type() == ov::element::f16) {
        const auto* data = tensor.data<const ov::float16>();
        std::vector<float> result(size);
        std::transform(data, data + size, result.begin(), [](ov::float16 value) {
            return static_cast<float>(value);
        });
        return result;
    }
    const auto* data = tensor.data<const float>();
    return {data, data + size};
}

float max_abs_diff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
    OPENVINO_ASSERT(lhs.size() == rhs.size());
    float diff = 0.0f;
    for (size_t i = 0; i < lhs.size(); ++i) {
        diff = std::max(diff, std::abs(lhs[i] - rhs[i]));
    }
    return diff;
}

}  // namespace

TEST(ModelBuilderLoraTest, BuildsStatelessLoraInputsWithNPUWNamesAndShapes) {
    const auto model = build_lora_model();
    const auto inputs = inputs_by_name(model);

    expect_input_shape(inputs, "lora_state_model.layers.0.self_attn.q_proj.MatMul.A", {2, 8});
    expect_input_shape(inputs, "lora_state_model.layers.0.self_attn.q_proj.MatMul.B", {8, 2});
    expect_input_shape(inputs, "lora_state_model.layers.0.self_attn.q_proj.MatMul.alpha", {1, 2}, ov::element::f32);
    expect_input_shape(inputs, "lora_state_model.layers.0.self_attn.v_proj.MatMul.A", {2, 8});
    expect_input_shape(inputs, "lora_state_model.layers.0.self_attn.v_proj.MatMul.B", {4, 2});
    expect_input_shape(inputs, "lora_state_model.layers.0.self_attn.v_proj.MatMul.alpha", {1, 2}, ov::element::f32);
}

TEST(ModelBuilderLoraTest, StatefulLoraCanBeConvertedToNPUWStatelessInputs) {
    auto model = build_lora_model(/*stateful=*/true);
    ASSERT_EQ(model->get_sinks().size(), 6u);
    ASSERT_TRUE(inputs_by_name(model).count("lora_state_model.layers.0.self_attn.q_proj.MatMul.A") == 0);

    ov::npuw::LoraStatefulToStatelessPass().run_on_model(model);
    model->validate_nodes_and_infer_types();

    EXPECT_TRUE(model->get_sinks().empty());
    const auto inputs = inputs_by_name(model);
    expect_input_shape(inputs, "lora_state_model.layers.0.self_attn.q_proj.MatMul.A", {2, 8});
    expect_input_shape(inputs, "lora_state_model.layers.0.self_attn.v_proj.MatMul.B", {4, 2});
}

TEST(ModelBuilderLoraTest, StatefulLoraMatchesCommonLoraSubgraphFusion) {
    auto model = build_lora_model(/*stateful=*/true);

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::LoraSubgraphFusion>();
    manager.run_passes(model);

    size_t fused_lora_count = 0;
    for (const auto& op : model->get_ordered_ops()) {
        if (ov::is_type<ov::op::internal::LoraSubgraph>(op)) {
            ++fused_lora_count;
        }
    }
    EXPECT_EQ(fused_lora_count, 2u);
}

TEST(ModelBuilderLoraTest, StatelessLoraCompilesAndChangesInferenceOutput) {
    auto model = build_lora_model();

    ov::Core core;
    ov::CompiledModel compiled_model;
    try {
        compiled_model = core.compile_model(model, "CPU");
    } catch (const std::exception& ex) {
        GTEST_SKIP() << "CPU plugin is unavailable: " << ex.what();
    }

    auto request = compiled_model.create_infer_request();

    set_inputs(request, compiled_model, 0.0f);
    request.infer();
    const auto zero_lora_output = read_f32_output(request, compiled_model.output(0));

    set_inputs(request, compiled_model, 0.25f);
    request.infer();
    const auto nonzero_lora_output = read_f32_output(request, compiled_model.output(0));

    EXPECT_GT(max_abs_diff(zero_lora_output, nonzero_lora_output), 1e-6f);
}
