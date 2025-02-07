// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "istft_shape_inference.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using ov::op::v0::Constant;
using ov::op::v0::Parameter;
using testing::HasSubstr;

class ISTFTShapeInferenceTest : public OpStaticShapeInferenceTest<op::v16::ISTFT> {};

TEST_F(ISTFTShapeInferenceTest, all_input_as_params_3D_data_no_signal_len) {
    const auto data_type = element::f32;
    const auto step_size_type = element::i32;
    const auto in_signal = std::make_shared<Parameter>(data_type, ov::PartialShape{-1, -1, -1});
    const auto in_window = std::make_shared<Parameter>(data_type, ov::PartialShape{-1});
    const auto in_frame_size = std::make_shared<Parameter>(step_size_type, ov::Shape{});
    const auto in_frame_step = std::make_shared<Parameter>(step_size_type, ov::Shape{});

    constexpr bool center = false;
    constexpr bool normalized = false;
    const auto op = make_op(in_signal, in_window, in_frame_size, in_frame_step, center, normalized);

    std::vector<StaticShape> static_input_shapes = {StaticShape{9, 3, 2},
                                                    StaticShape{16},
                                                    StaticShape{},
                                                    StaticShape{}};
    int32_t frame_size = 16;
    int32_t frame_step = 16;

    auto const_data = std::unordered_map<size_t, Tensor>{{2, {element::i32, ov::Shape{}, &frame_size}},
                                                         {3, {element::i32, ov::Shape{}, &frame_step}}};
    auto acc = make_tensor_accessor(const_data);
    auto static_output_shapes = shape_infer(op.get(), static_input_shapes, acc);
    ASSERT_EQ(static_output_shapes[0], StaticShape({48}));
}

TEST_F(ISTFTShapeInferenceTest, all_input_as_params_3D_data_with_signal_len) {
    const auto data_type = element::f32;
    const auto step_size_type = element::i32;
    const auto in_signal = std::make_shared<Parameter>(data_type, ov::PartialShape{-1, -1, -1});
    const auto in_window = std::make_shared<Parameter>(data_type, ov::PartialShape{-1});
    const auto in_frame_size = std::make_shared<Parameter>(step_size_type, ov::Shape{});
    const auto in_frame_step = std::make_shared<Parameter>(step_size_type, ov::Shape{});
    const auto in_signal_len = std::make_shared<Parameter>(data_type, ov::PartialShape{-1});

    constexpr bool center = false;
    constexpr bool normalized = false;
    const auto op = make_op(in_signal, in_window, in_frame_size, in_frame_step, center, normalized);

    std::vector<StaticShape> static_input_shapes = {StaticShape{9, 3, 2},
                                                    StaticShape{16},
                                                    StaticShape{},
                                                    StaticShape{},
                                                    StaticShape{1}};
    int32_t frame_size = 16;
    int32_t frame_step = 16;
    int32_t signal_len = 60;

    auto const_data = std::unordered_map<size_t, Tensor>{{2, {element::i32, ov::Shape{}, &frame_size}},
                                                         {3, {element::i32, ov::Shape{}, &frame_step}},
                                                         {4, {element::i32, ov::Shape{1}, &signal_len}}};
    auto acc = make_tensor_accessor(const_data);
    auto static_output_shapes = shape_infer(op.get(), static_input_shapes, acc);
    ASSERT_EQ(static_output_shapes[0], StaticShape{static_cast<size_t>(signal_len)});
}
