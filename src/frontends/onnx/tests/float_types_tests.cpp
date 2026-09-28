// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "common_test_utils/test_case.hpp"
#include "onnx_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"

using namespace ov;
using namespace ov::frontend::onnx::tests;

namespace {
std::shared_ptr<ov::op::v0::Constant> get_single_constant(const std::shared_ptr<ov::Model>& model) {
    std::shared_ptr<ov::op::v0::Constant> found{nullptr};
    for (const auto& op : model->get_ordered_ops()) {
        if (const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(op)) {
            EXPECT_EQ(found, nullptr) << "More than one Constant found in the model";
            found = constant;
        }
    }
    EXPECT_NE(found, nullptr) << "No Constant found in the model";
    return found;
}
}  // namespace

TEST(ONNXFeFloatTypes, float4e2m1_input) {
    const auto model = convert_model("float4e2m1_input.onnx");

    ASSERT_EQ(model->get_parameters().size(), 1);
    EXPECT_EQ(model->get_parameters()[0]->get_element_type(), ov::element::f4e2m1);
    EXPECT_EQ(model->get_parameters()[0]->get_partial_shape(), ov::PartialShape({6}));
    ASSERT_EQ(model->get_results().size(), 2);
    EXPECT_EQ(model->get_results()[1]->get_element_type(), ov::element::f4e2m1);
}

TEST(ONNXFeFloatTypes, float4e2m1_constant) {
    const auto model = convert_model("float4e2m1_constant.onnx");

    const auto constant = get_single_constant(model);
    ASSERT_NE(constant, nullptr);
    EXPECT_EQ(constant->get_element_type(), ov::element::f4e2m1);
    EXPECT_EQ(constant->get_shape(), ov::Shape({6}));
    EXPECT_EQ(constant->cast_vector<float>(), std::vector<float>({0.0f, 0.5f, 1.0f, -1.0f, 6.0f, -6.0f}));
}

TEST(ONNXFeFloatTypes, float4e2m1_initializer) {
    const auto model = convert_model("float4e2m1_initializer.onnx");

    const auto constant = get_single_constant(model);
    ASSERT_NE(constant, nullptr);
    EXPECT_EQ(constant->get_element_type(), ov::element::f4e2m1);
    EXPECT_EQ(constant->get_shape(), ov::Shape({6}));
    EXPECT_EQ(constant->cast_vector<float>(), std::vector<float>({0.0f, 0.5f, 1.0f, -1.0f, 6.0f, -6.0f}));
}

TEST(ONNXFeFloatTypes, cast_float32_to_float4e2m1) {
    const auto model = convert_model("cast_float32_to_float4e2m1.onnx");

    ASSERT_EQ(model->get_results().size(), 1);
    EXPECT_EQ(model->get_results()[0]->get_element_type(), ov::element::f4e2m1);
}

TEST(ONNXFeFloatTypes, float8e8m0_input) {
    const auto model = convert_model("float8e8m0_input.onnx");

    ASSERT_EQ(model->get_parameters().size(), 1);
    EXPECT_EQ(model->get_parameters()[0]->get_element_type(), ov::element::f8e8m0);
    EXPECT_EQ(model->get_parameters()[0]->get_partial_shape(), ov::PartialShape({6}));
    ASSERT_EQ(model->get_results().size(), 2);
    EXPECT_EQ(model->get_results()[1]->get_element_type(), ov::element::f8e8m0);
}

TEST(ONNXFeFloatTypes, float8e8m0_constant) {
    const auto model = convert_model("float8e8m0_constant.onnx");

    const auto constant = get_single_constant(model);
    ASSERT_NE(constant, nullptr);
    EXPECT_EQ(constant->get_element_type(), ov::element::f8e8m0);
    EXPECT_EQ(constant->get_shape(), ov::Shape({6}));

    // f8e8m0 stores an exponent only: value = 2^(bits - 127), 0xFF encodes NaN
    const auto values = constant->cast_vector<float>();
    ASSERT_EQ(values.size(), 6);
    EXPECT_EQ(values[0], 1.0f);
    EXPECT_EQ(values[1], 2.0f);
    EXPECT_EQ(values[2], 0.5f);
    EXPECT_EQ(values[3], 4.0f);
    EXPECT_EQ(values[4], 0.25f);
    EXPECT_TRUE(std::isnan(values[5]));
}

TEST(ONNXFeFloatTypes, float8e8m0_initializer) {
    const auto model = convert_model("float8e8m0_initializer.onnx");

    const auto constant = get_single_constant(model);
    ASSERT_NE(constant, nullptr);
    EXPECT_EQ(constant->get_element_type(), ov::element::f8e8m0);
    EXPECT_EQ(constant->get_shape(), ov::Shape({6}));

    const auto values = constant->cast_vector<float>();
    ASSERT_EQ(values.size(), 6);
    EXPECT_EQ(values[0], 1.0f);
    EXPECT_EQ(values[1], 2.0f);
    EXPECT_EQ(values[2], 0.5f);
    EXPECT_EQ(values[3], 4.0f);
    EXPECT_EQ(values[4], 0.25f);
    EXPECT_TRUE(std::isnan(values[5]));
}

TEST(ONNXFeFloatTypes, cast_float32_to_float8e8m0) {
    const auto model = convert_model("cast_float32_to_float8e8m0.onnx");

    ASSERT_EQ(model->get_results().size(), 1);
    EXPECT_EQ(model->get_results()[0]->get_element_type(), ov::element::f8e8m0);
}

TEST(ONNXFeFloatTypes, dequantize_linear_float8e8m0_scale) {
    const auto model = convert_model("dequantize_linear_f8e8m0_scale.onnx");

    ASSERT_EQ(model->get_results().size(), 1);
    EXPECT_EQ(model->get_results()[0]->get_element_type(), ov::element::f32);

    ov::test::TestCase test_case(model);
    test_case.add_expected_output<float>(ov::Shape{6}, {0.0f, 1.0f, 2.0f, -2.0f, 12.0f, -12.0f});
    test_case.run();
}

TEST(ONNXFeFloatTypes, quantize_dequantize_linear_float4e2m1) {
    const auto model = convert_model("quant_dequant_f4e2m1.onnx");

    ASSERT_EQ(model->get_results().size(), 1);
    EXPECT_EQ(model->get_results()[0]->get_element_type(), ov::element::f32);

    ov::test::TestCase test_case(model);
    test_case.add_input<float>(ov::Shape{6}, {0.0f, 1.0f, 2.0f, -2.0f, 12.0f, 20.0f});
    // 20.0 / 2.0 = 10.0 saturates to the maximal f4e2m1 value (6.0)
    test_case.add_expected_output<float>(ov::Shape{6}, {0.0f, 1.0f, 2.0f, -2.0f, 12.0f, 12.0f});
    test_case.run();
}

TEST(ONNXFeFloatTypes, quantize_linear_float4e2m1_zero_point) {
    const auto model = convert_model("quantize_linear_f4e2m1_zero_point.onnx");

    ASSERT_EQ(model->get_results().size(), 1);
    EXPECT_EQ(model->get_results()[0]->get_element_type(), ov::element::f4e2m1);
}

TEST(ONNXFeFloatTypes, dequantize_linear_invalid_output_dtype) {
    // "output_dtype" set to INT32, which is not a valid output type of DequantizeLinear.
    // Without validation this would silently produce a graph doing the dequantization in i32.
    try {
        convert_model("dequantize_linear_invalid_output_dtype.onnx");
        FAIL() << "Expected an exception for an unsupported \"output_dtype\" attribute";
    } catch (const std::exception& e) {
        EXPECT_NE(std::string{e.what()}.find(
                      "The \"output_dtype\" attribute of DequantizeLinear must be one of the supported types"),
                  std::string::npos)
            << e.what();
    }
}
