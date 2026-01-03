// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Regression test for GitHub issue #33164:
// u8 Subtract must wrap around (e.g., 3 - 4 = 255), not saturate to 0.
// https://github.com/openvinotoolkit/openvino/issues/33164

#include <gtest/gtest.h>

#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/openvino.hpp"

namespace ov {
namespace test {
namespace {

class SubtractU8WrapAroundTest : public ::testing::Test {
protected:
    void SetUp() override {
        core = std::make_shared<ov::Core>();
    }

    std::shared_ptr<ov::Core> core;
};

// Test that u8 subtraction wraps around instead of saturating.
// This is a regression test for https://github.com/openvinotoolkit/openvino/issues/33164
TEST_F(SubtractU8WrapAroundTest, WrapAroundBehavior) {
    // Create a simple model: out = a - b (both u8)
    auto a = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::Shape{4});
    auto b = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::Shape{4});
    auto subtract = std::make_shared<ov::op::v1::Subtract>(a, b);
    auto result = std::make_shared<ov::op::v0::Result>(subtract);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{a, b});

    // Compile for CPU
    auto compiled_model = core->compile_model(model, "CPU");
    auto infer_request = compiled_model.create_infer_request();

    // Test cases that exercise underflow:
    // Input A: [3, 0, 1, 5]
    // Input B: [4, 1, 2, 3]
    // Expected with wrap-around (mod 256): [255, 255, 255, 2]
    // Wrong saturation result would be:    [0,   0,   0,   2]
    std::vector<uint8_t> input_a = {3, 0, 1, 5};
    std::vector<uint8_t> input_b = {4, 1, 2, 3};
    std::vector<uint8_t> expected = {255, 255, 255, 2};

    auto tensor_a = ov::Tensor(ov::element::u8, {4}, input_a.data());
    auto tensor_b = ov::Tensor(ov::element::u8, {4}, input_b.data());

    infer_request.set_tensor(a, tensor_a);
    infer_request.set_tensor(b, tensor_b);
    infer_request.infer();

    auto output = infer_request.get_output_tensor(0);
    auto output_data = output.data<uint8_t>();

    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(output_data[i], expected[i])
            << "Mismatch at index " << i << ": got " << static_cast<int>(output_data[i]) << ", expected "
            << static_cast<int>(expected[i]) << ". u8 subtraction should wrap around (mod 256), not saturate to 0.";
    }
}

// Test with larger tensor to exercise vector path in JIT
TEST_F(SubtractU8WrapAroundTest, WrapAroundBehaviorLargeVector) {
    const size_t size = 64;  // Large enough to trigger vectorized JIT path

    auto a = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::Shape{size});
    auto b = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::Shape{size});
    auto subtract = std::make_shared<ov::op::v1::Subtract>(a, b);
    auto result = std::make_shared<ov::op::v0::Result>(subtract);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{a, b});

    auto compiled_model = core->compile_model(model, "CPU");
    auto infer_request = compiled_model.create_infer_request();

    std::vector<uint8_t> input_a(size);
    std::vector<uint8_t> input_b(size);
    std::vector<uint8_t> expected(size);

    for (size_t i = 0; i < size; ++i) {
        input_a[i] = static_cast<uint8_t>(i % 10);        // 0-9 repeating
        input_b[i] = static_cast<uint8_t>((i % 10) + 1);  // 1-10 repeating
        // Each result should be -1 mod 256 = 255, except when a >= b
        expected[i] = static_cast<uint8_t>((256 + input_a[i] - input_b[i]) % 256);
    }

    auto tensor_a = ov::Tensor(ov::element::u8, {size}, input_a.data());
    auto tensor_b = ov::Tensor(ov::element::u8, {size}, input_b.data());

    infer_request.set_tensor(a, tensor_a);
    infer_request.set_tensor(b, tensor_b);
    infer_request.infer();

    auto output = infer_request.get_output_tensor(0);
    auto output_data = output.data<uint8_t>();

    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(output_data[i], expected[i])
            << "Mismatch at index " << i << ": got " << static_cast<int>(output_data[i]) << ", expected "
            << static_cast<int>(expected[i]);
    }
}

// Test with 4D tensor to match typical NN tensor shapes
TEST_F(SubtractU8WrapAroundTest, WrapAroundBehavior4D) {
    auto a = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::Shape{1, 2, 2, 2});
    auto b = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::Shape{1, 2, 2, 2});
    auto subtract = std::make_shared<ov::op::v1::Subtract>(a, b);
    auto result = std::make_shared<ov::op::v0::Result>(subtract);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{a, b});

    auto compiled_model = core->compile_model(model, "CPU");
    auto infer_request = compiled_model.create_infer_request();

    // All zeros minus all ones should give all 255s (wrap-around)
    std::vector<uint8_t> input_a(8, 0);
    std::vector<uint8_t> input_b(8, 1);
    std::vector<uint8_t> expected(8, 255);

    auto tensor_a = ov::Tensor(ov::element::u8, {1, 2, 2, 2}, input_a.data());
    auto tensor_b = ov::Tensor(ov::element::u8, {1, 2, 2, 2}, input_b.data());

    infer_request.set_tensor(a, tensor_a);
    infer_request.set_tensor(b, tensor_b);
    infer_request.infer();

    auto output = infer_request.get_output_tensor(0);
    auto output_data = output.data<uint8_t>();

    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(output_data[i], expected[i])
            << "4D tensor mismatch at index " << i << ": got " << static_cast<int>(output_data[i]) << ", expected "
            << static_cast<int>(expected[i]);
    }
}

}  // namespace
}  // namespace test
}  // namespace ov
