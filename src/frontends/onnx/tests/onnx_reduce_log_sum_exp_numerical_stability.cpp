// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "onnx_utils.hpp"
#include "openvino/openvino.hpp"
#include "openvino/opsets/opset13.hpp"

using namespace ov;
using namespace ov::frontend::onnx::tests;

TEST(ReduceLogSumExpNumericalStabilityTest, TestNumericalStability) {
    // Test the numerically stable implementation directly using OpenVINO operations
    // This verifies that the stable log-sum-exp formula works correctly for large values

    ov::Core core;

    // Test case 1: Large positive values that would overflow exp() in naive implementation
    {
        auto input = std::make_shared<ov::opset13::Parameter>(ov::element::f32, ov::Shape{2});
        auto axes = std::make_shared<ov::opset13::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});

        // Create the numerically stable ReduceLogSumExp: k + log(sum(exp(x - k))) where k = max(x)
        auto k = std::make_shared<ov::opset13::ReduceMax>(input, axes, false);  // keepdims=false
        auto input_minus_k = std::make_shared<ov::opset13::Subtract>(input, k);
        auto exp_node = std::make_shared<ov::opset13::Exp>(input_minus_k);
        auto sum_node = std::make_shared<ov::opset13::ReduceSum>(exp_node, axes, false);  // keepdims=false
        auto log_node = std::make_shared<ov::opset13::Log>(sum_node);
        auto result = std::make_shared<ov::opset13::Add>(k, log_node);

        auto model = std::make_shared<ov::Model>(ov::OutputVector{result}, ov::ParameterVector{input});

        // Test with large values that would cause overflow in naive implementation
        std::vector<float> input_data = {100.0f, 101.0f};  // These would cause exp() overflow

        auto compiled_model = core.compile_model(model, "CPU");
        auto infer_request = compiled_model.create_infer_request();

        auto input_tensor = ov::Tensor(ov::element::f32, ov::Shape{2}, input_data.data());
        infer_request.set_input_tensor(input_tensor);
        infer_request.infer();

        auto output_tensor = infer_request.get_output_tensor();
        auto output_data = output_tensor.data<float>();

        // Expected result using numerically stable formula: max + log(sum(exp(x - max)))
        // max = 101, so result should be 101 + log(exp(100-101) + exp(101-101)) = 101 + log(exp(-1) + 1)
        float expected = 101.0f + std::log(std::exp(-1.0f) + 1.0f);

        EXPECT_TRUE(std::isfinite(output_data[0])) << "Result should be finite, not inf or nan";
        EXPECT_NEAR(output_data[0], expected, 1e-5f) << "Numerical result should match expected stable computation";
    }

    // Test case 2: Very large values near float32 limits
    {
        auto input = std::make_shared<ov::opset13::Parameter>(ov::element::f32, ov::Shape{3});
        auto axes = std::make_shared<ov::opset13::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});

        // Create the numerically stable ReduceLogSumExp: k + log(sum(exp(x - k))) where k = max(x)
        auto k = std::make_shared<ov::opset13::ReduceMax>(input, axes, false);  // keepdims=false
        auto input_minus_k = std::make_shared<ov::opset13::Subtract>(input, k);
        auto exp_node = std::make_shared<ov::opset13::Exp>(input_minus_k);
        auto sum_node = std::make_shared<ov::opset13::ReduceSum>(exp_node, axes, false);  // keepdims=false
        auto log_node = std::make_shared<ov::opset13::Log>(sum_node);
        auto result = std::make_shared<ov::opset13::Add>(k, log_node);

        auto model = std::make_shared<ov::Model>(ov::OutputVector{result}, ov::ParameterVector{input});

        // Test with values that are close to causing overflow (88.7 is near log(FLT_MAX))
        std::vector<float> input_data = {88.0f, 89.0f, 90.0f};

        auto compiled_model = core.compile_model(model, "CPU");
        auto infer_request = compiled_model.create_infer_request();

        auto input_tensor = ov::Tensor(ov::element::f32, ov::Shape{3}, input_data.data());
        infer_request.set_input_tensor(input_tensor);
        infer_request.infer();

        auto output_tensor = infer_request.get_output_tensor();
        auto output_data = output_tensor.data<float>();

        EXPECT_TRUE(std::isfinite(output_data[0])) << "Result should be finite even for large input values";

        // The result should be close to the maximum value plus a small correction
        EXPECT_GT(output_data[0], 90.0f) << "Result should be greater than the maximum input";
        EXPECT_LT(output_data[0], 92.0f) << "Result should not be much larger than maximum input";
    }

    // Test case 3: Mixed positive and negative values
    {
        auto input = std::make_shared<ov::opset13::Parameter>(ov::element::f32, ov::Shape{4});
        auto axes = std::make_shared<ov::opset13::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});

        // Create the numerically stable ReduceLogSumExp: k + log(sum(exp(x - k))) where k = max(x)
        auto k = std::make_shared<ov::opset13::ReduceMax>(input, axes, false);  // keepdims=false
        auto input_minus_k = std::make_shared<ov::opset13::Subtract>(input, k);
        auto exp_node = std::make_shared<ov::opset13::Exp>(input_minus_k);
        auto sum_node = std::make_shared<ov::opset13::ReduceSum>(exp_node, axes, false);  // keepdims=false
        auto log_node = std::make_shared<ov::opset13::Log>(sum_node);
        auto result = std::make_shared<ov::opset13::Add>(k, log_node);

        auto model = std::make_shared<ov::Model>(ov::OutputVector{result}, ov::ParameterVector{input});

        std::vector<float> input_data = {-10.0f, 0.0f, 10.0f, 50.0f};

        auto compiled_model = core.compile_model(model, "CPU");
        auto infer_request = compiled_model.create_infer_request();

        auto input_tensor = ov::Tensor(ov::element::f32, ov::Shape{4}, input_data.data());
        infer_request.set_input_tensor(input_tensor);
        infer_request.infer();

        auto output_tensor = infer_request.get_output_tensor();
        auto output_data = output_tensor.data<float>();

        EXPECT_TRUE(std::isfinite(output_data[0])) << "Result should be finite for mixed values";

        // Result should be dominated by the largest value (50.0)
        EXPECT_GT(output_data[0], 50.0f) << "Result should be greater than the maximum input";
        EXPECT_LT(output_data[0], 51.0f) << "Result should be close to the maximum input for this case";
    }
}

// Test with keepdims=true to ensure both modes work
TEST(ReduceLogSumExpNumericalStabilityTest, TestWithKeepDims) {
    ov::Core core;

    auto input = std::make_shared<ov::opset13::Parameter>(ov::element::f32, ov::Shape{2});
    auto axes = std::make_shared<ov::opset13::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});

    // Create the numerically stable ReduceLogSumExp: k + log(sum(exp(x - k))) where k = max(x)
    auto k = std::make_shared<ov::opset13::ReduceMax>(input, axes, true);  // keepdims=true
    auto input_minus_k = std::make_shared<ov::opset13::Subtract>(input, k);
    auto exp_node = std::make_shared<ov::opset13::Exp>(input_minus_k);
    auto sum_node = std::make_shared<ov::opset13::ReduceSum>(exp_node, axes, true);  // keepdims=true
    auto log_node = std::make_shared<ov::opset13::Log>(sum_node);
    auto result = std::make_shared<ov::opset13::Add>(k, log_node);

    auto model = std::make_shared<ov::Model>(ov::OutputVector{result}, ov::ParameterVector{input});

    // Test with large values that would cause overflow in naive implementation
    std::vector<float> input_data = {89.0f, 89.0f};  // Values that cause overflow in naive implementation

    auto compiled_model = core.compile_model(model, "CPU");
    auto infer_request = compiled_model.create_infer_request();

    auto input_tensor = ov::Tensor(ov::element::f32, ov::Shape{2}, input_data.data());
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();

    auto output_tensor = infer_request.get_output_tensor();
    auto output_data = output_tensor.data<float>();

    // For these specific values, stable result should be approximately 89 + log(2)
    float expected = 89.0f + std::log(2.0f);

    EXPECT_TRUE(std::isfinite(output_data[0])) << "Stable implementation should remain finite";
    EXPECT_NEAR(output_data[0], expected, 1e-5f) << "Stable implementation should match expected mathematical result";

    // Verify output shape is [1] due to keepdims=true
    EXPECT_EQ(output_tensor.get_shape(), ov::Shape({1}));
}

// Test that demonstrates the difference between naive and stable implementations
TEST(ReduceLogSumExpNumericalStabilityTest, CompareNaiveVsStable) {
    // This test shows why the stable implementation is necessary

    std::vector<float> test_values = {89.0f, 89.0f};  // Values that cause overflow in naive implementation

    // Naive implementation (would overflow): log(sum(exp(x)))
    float naive_result = std::log(std::exp(test_values[0]) + std::exp(test_values[1]));

    // Stable implementation: k + log(sum(exp(x - k))) where k = max(x)
    float k = std::max(test_values[0], test_values[1]);
    float stable_result = k + std::log(std::exp(test_values[0] - k) + std::exp(test_values[1] - k));

    // The naive implementation should produce inf or nan for large values
    // The stable implementation should produce a finite result
    EXPECT_FALSE(std::isfinite(naive_result)) << "Naive implementation should overflow for large values";
    EXPECT_TRUE(std::isfinite(stable_result)) << "Stable implementation should remain finite";

    // For these specific values, stable result should be approximately 89 + log(2)
    float expected = 89.0f + std::log(2.0f);
    EXPECT_NEAR(stable_result, expected, 1e-5f) << "Stable implementation should match expected mathematical result";
}