// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <memory>

#include "openvino/core/model.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/tensor.hpp"

namespace {

TEST(CPUInputTensorInitialization, NewRequestDoesNotExposePreviousInput) {
    constexpr size_t count = 4096;
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{count});
    auto model = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(input)},
                                             ov::ParameterVector{input});
    ov::Core core;
    auto compiled_model = core.compile_model(model, "CPU");

    {
        auto previous_request = compiled_model.create_infer_request();
        auto tensor = previous_request.get_input_tensor();
        std::fill_n(tensor.data<float>(), count, 123.25F);
        previous_request.infer();
        EXPECT_EQ(previous_request.get_output_tensor().data<const float>()[count - 1], 123.25F);
    }

    auto next_request = compiled_model.create_infer_request();
    auto input_tensor = next_request.get_input_tensor();
    for (size_t i = 0; i < count; ++i) {
        EXPECT_EQ(input_tensor.data<const float>()[i], 0.0F) << "at index " << i;
    }
    next_request.infer();
    auto output_tensor = next_request.get_output_tensor();
    for (size_t i = 0; i < count; ++i) {
        EXPECT_EQ(output_tensor.data<const float>()[i], 0.0F) << "at index " << i;
    }

    // Explicitly written input data must remain available for subsequent inference on the same request.
    std::fill_n(input_tensor.data<float>(), count, 42.0F);
    next_request.infer();
    EXPECT_EQ(next_request.get_output_tensor().data<const float>()[count - 1], 42.0F);
}

TEST(CPUInputTensorInitialization, OnlyDefaultInputsAreCleared) {
    constexpr size_t count = 256;
    auto first = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{count});
    auto second = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{count});
    auto model = std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(first), std::make_shared<ov::op::v0::Result>(second)},
        ov::ParameterVector{first, second});
    ov::Core core;
    auto compiled_model = core.compile_model(model, "CPU");

    {
        auto previous_request = compiled_model.create_infer_request();
        std::fill_n(previous_request.get_input_tensor(1).data<int32_t>(), count, 12345);
        previous_request.infer();
    }

    auto request = compiled_model.create_infer_request();
    ov::Tensor supplied(ov::element::i32, ov::Shape{count});
    std::fill_n(supplied.data<int32_t>(), count, 67890);
    request.set_input_tensor(0, supplied);
    request.infer();

    for (size_t i = 0; i < count; ++i) {
        EXPECT_EQ(supplied.data<const int32_t>()[i], 67890);
        EXPECT_EQ(request.get_output_tensor(0).data<const int32_t>()[i], 67890);
        EXPECT_EQ(request.get_output_tensor(1).data<const int32_t>()[i], 0);
    }
}

}  // namespace
