// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include "openvino/openvino.hpp"
#include "openvino/opsets/opset13.hpp"

namespace ov {
namespace test {
namespace {

// Parameters: enable_snippets, constant_branch.
using ConstantResultInPlaceParams = std::tuple<bool, bool>;

class ConstantResultInPlaceTest : public testing::TestWithParam<ConstantResultInPlaceParams> {};

TEST_P(ConstantResultInPlaceTest, PreserveConstantResultAcrossInferences) {
    const auto params = GetParam();
    const bool enable_snippets = std::get<0>(params);
    const bool constant_branch = std::get<1>(params);
    ov::Core core;

    const ov::Shape shape{100, 1, 512};
    const ov::Shape affine_shape{1, 1, 512};
    const size_t size = ov::shape_size(shape);
    std::vector<float> constant_values(size);
    std::vector<float> input_values(size);
    for (size_t i = 0; i < size; ++i) {
        const auto index = static_cast<float>(i);
        constant_values[i] = 0.003f * std::sin(index * 0.17f);
        input_values[i] = std::cos(index * 0.11f) + 0.4f * std::sin(index * 0.07f);
    }

    std::vector<float> scale1(512), bias1(512), scale2(512), bias2(512);
    for (size_t i = 0; i < 512; ++i) {
        const auto index = static_cast<float>(i);
        scale1[i] = 0.95f + 0.03f * std::sin(index);
        bias1[i] = 0.03f * std::cos(index);
        scale2[i] = 1.0f + 0.03f * std::cos(index);
        bias2[i] = 0.02f * std::sin(index);
    }

    const auto axes = ov::opset13::Constant::create(ov::element::i32, ov::Shape{1}, {-1});
    const auto norm = [&](const ov::Output<ov::Node>& value,
                          const std::vector<float>& scale,
                          const std::vector<float>& bias) {
        const auto mvn = std::make_shared<ov::opset13::MVN>(value, axes, true, 1e-5, ov::op::MVNEpsMode::INSIDE_SQRT);
        const auto multiply = std::make_shared<ov::opset13::Multiply>(
            mvn,
            ov::opset13::Constant::create(ov::element::f32, affine_shape, scale));
        return std::make_shared<ov::opset13::Add>(multiply,
                                                  ov::opset13::Constant::create(ov::element::f32, affine_shape, bias));
    };

    const auto parameter = std::make_shared<ov::opset13::Parameter>(ov::element::f32, shape);
    // A computed constant has one non-constant consumer, which must not overwrite its cached buffer.
    ov::ParameterVector parameters{parameter};
    ov::Output<ov::Node> branch_input;
    if (constant_branch) {
        branch_input = ov::opset13::Constant::create(ov::element::f32, shape, constant_values);
    } else {
        const auto branch_parameter = std::make_shared<ov::opset13::Parameter>(ov::element::f32, shape);
        parameters.push_back(branch_parameter);
        branch_input = branch_parameter;
    }
    const auto residual = norm(branch_input, scale1, bias1);
    const auto sum = std::make_shared<ov::opset13::Add>(residual, parameter);
    const auto output = norm(sum, scale2, bias2);
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{output}, parameters, "ConstantResultInPlace");

    const ov::AnyMap config{
        {ov::inference_num_threads.name(), 1},
        {ov::num_streams.name(), 1},
        {"SNIPPETS_MODE", enable_snippets ? "ENABLE" : "DISABLE"},
    };
    auto compiled = core.compile_model(model, "CPU", config);
    std::vector<float> baseline;
    for (size_t call = 0; call < 10; ++call) {
        auto request = compiled.create_infer_request();
        auto input = ov::Tensor(ov::element::f32, shape);
        std::copy(input_values.begin(), input_values.end(), input.data<float>());
        request.set_input_tensor(0, input);
        if (!constant_branch) {
            auto branch_tensor = ov::Tensor(ov::element::f32, shape);
            std::copy(constant_values.begin(), constant_values.end(), branch_tensor.data<float>());
            request.set_input_tensor(1, branch_tensor);
        }
        request.infer();

        const auto result = request.get_output_tensor();
        ASSERT_EQ(result.get_element_type(), ov::element::f32);
        ASSERT_EQ(result.get_size(), size);
        const auto* values = result.data<const float>();
        if (call == 0) {
            baseline.assign(values, values + size);
        } else {
            for (size_t i = 0; i < size; ++i) {
                ASSERT_EQ(values[i], baseline[i]) << "Call " << call + 1 << ", element " << i;
            }
        }
    }
    // Different inputs must also match a fresh compilation with the same configuration.
    for (size_t call = 0; call < 3; ++call) {
        auto fresh = core.compile_model(model, "CPU", config);
        const auto infer = [&](ov::CompiledModel& compiled_model) {
            auto request = compiled_model.create_infer_request();
            auto input = ov::Tensor(ov::element::f32, shape);
            for (size_t i = 0; i < size; ++i) {
                input.data<float>()[i] = input_values[i] * (1.0f + 0.25f * call);
            }
            request.set_input_tensor(0, input);
            if (!constant_branch) {
                auto branch_tensor = ov::Tensor(ov::element::f32, shape);
                std::copy(constant_values.begin(), constant_values.end(), branch_tensor.data<float>());
                request.set_input_tensor(1, branch_tensor);
            }
            request.infer();
            const auto result = request.get_output_tensor();
            const auto* values = result.data<const float>();
            return std::vector<float>(values, values + result.get_size());
        };
        const auto actual = infer(compiled);
        const auto expected = infer(fresh);
        ASSERT_EQ(actual.size(), expected.size());
        for (size_t i = 0; i < actual.size(); ++i) {
            ASSERT_EQ(actual[i], expected[i]) << "Different input " << call << ", element " << i;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(smoke_CPU,
                         ConstantResultInPlaceTest,
                         testing::Combine(testing::Bool(), testing::Bool()),
                         [](const testing::TestParamInfo<ConstantResultInPlaceParams>& info) {
                             const bool enable_snippets = std::get<0>(info.param);
                             return std::string(enable_snippets ? "SnippetsEnabled" : "SnippetsDisabled") +
                                    (std::get<1>(info.param) ? "ConstantBranch" : "ParameterBranch");
                         });

}  // namespace
}  // namespace test
}  // namespace ov
