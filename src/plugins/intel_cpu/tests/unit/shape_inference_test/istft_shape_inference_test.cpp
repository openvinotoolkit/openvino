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

TEST_F(ISTFTShapeInferenceTest, all_input_as_params_4D_data_no_signal_len) {
    const auto data_type = element::f32;
    const auto step_size_type = element::i32;
    const auto in_signal = std::make_shared<Parameter>(data_type, ov::PartialShape{-1, -1, -1, -1});
    const auto in_window = std::make_shared<Parameter>(data_type, ov::PartialShape{-1});
    const auto in_frame_size = std::make_shared<Parameter>(step_size_type, ov::Shape{});
    const auto in_frame_step = std::make_shared<Parameter>(step_size_type, ov::Shape{});

    constexpr bool center = false;
    constexpr bool normalized = false;
    const auto op = make_op(in_signal, in_window, in_frame_size, in_frame_step, center, normalized);

    std::vector<StaticShape> static_input_shapes = {StaticShape{8, 9, 3, 2},
                                                    StaticShape{16},
                                                    StaticShape{},
                                                    StaticShape{}};
    int32_t frame_size = 16;
    int32_t frame_step = 16;

    auto const_data = std::unordered_map<size_t, Tensor>{{2, {element::i32, ov::Shape{}, &frame_size}},
                                                         {3, {element::i32, ov::Shape{}, &frame_step}}};
    auto acc = make_tensor_accessor(const_data);
    auto static_output_shapes = shape_infer(op.get(), static_input_shapes, acc);
    ASSERT_EQ(static_output_shapes[0], StaticShape({8, 48}));
}

TEST_F(ISTFTShapeInferenceTest, all_input_as_params_3D_data_no_signal_len_center) {
    const auto data_type = element::f32;
    const auto step_size_type = element::i32;
    const auto in_signal = std::make_shared<Parameter>(data_type, ov::PartialShape{-1, -1, -1});
    const auto in_window = std::make_shared<Parameter>(data_type, ov::PartialShape{-1});
    const auto in_frame_size = std::make_shared<Parameter>(step_size_type, ov::Shape{});
    const auto in_frame_step = std::make_shared<Parameter>(step_size_type, ov::Shape{});

    constexpr bool center = true;
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
    ASSERT_EQ(static_output_shapes[0], StaticShape({32}));
}

TEST_F(ISTFTShapeInferenceTest, all_input_as_params_3D_data_with_signal_len) {
    const auto data_type = element::f32;
    const auto step_size_type = element::i32;
    const auto in_signal = std::make_shared<Parameter>(data_type, ov::PartialShape{-1, -1, -1});
    const auto in_window = std::make_shared<Parameter>(data_type, ov::PartialShape{-1});
    const auto in_frame_size = std::make_shared<Parameter>(step_size_type, ov::Shape{});
    const auto in_frame_step = std::make_shared<Parameter>(step_size_type, ov::Shape{});
    const auto in_signal_len = std::make_shared<Parameter>(step_size_type, ov::PartialShape{-1});

    constexpr bool center = false;
    constexpr bool normalized = false;
    const auto op = make_op(in_signal, in_window, in_frame_size, in_frame_step, in_signal_len, center, normalized);

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

TEST_F(ISTFTShapeInferenceTest, all_input_as_params_3D_data_with_signal_len_center) {
    const auto data_type = element::f32;
    const auto step_size_type = element::i32;
    const auto in_signal = std::make_shared<Parameter>(data_type, ov::PartialShape{-1, -1, -1});
    const auto in_window = std::make_shared<Parameter>(data_type, ov::PartialShape{-1});
    const auto in_frame_size = std::make_shared<Parameter>(step_size_type, ov::Shape{});
    const auto in_frame_step = std::make_shared<Parameter>(step_size_type, ov::Shape{});
    const auto in_signal_len = std::make_shared<Parameter>(step_size_type, ov::PartialShape{-1});

    constexpr bool center = true;
    constexpr bool normalized = false;
    const auto op = make_op(in_signal, in_window, in_frame_size, in_frame_step, in_signal_len, center, normalized);

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

TEST_F(ISTFTShapeInferenceTest, all_input_as_params_3D_data_with_signal_len_i64) {
    const auto data_type = element::f32;
    const auto step_size_type = element::i64;
    const auto in_signal = std::make_shared<Parameter>(data_type, ov::PartialShape{-1, -1, -1});
    const auto in_window = std::make_shared<Parameter>(data_type, ov::PartialShape{-1});
    const auto in_frame_size = std::make_shared<Parameter>(step_size_type, ov::Shape{});
    const auto in_frame_step = std::make_shared<Parameter>(step_size_type, ov::Shape{});
    const auto in_signal_len = std::make_shared<Parameter>(step_size_type, ov::PartialShape{-1});

    constexpr bool center = false;
    constexpr bool normalized = false;
    const auto op = make_op(in_signal, in_window, in_frame_size, in_frame_step, in_signal_len, center, normalized);

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

TEST_F(ISTFTShapeInferenceTest, all_input_as_params_4D_data_with_signal_len) {
    const auto data_type = element::f32;
    const auto step_size_type = element::i32;
    const auto in_signal = std::make_shared<Parameter>(data_type, ov::PartialShape::dynamic());
    const auto in_window = std::make_shared<Parameter>(data_type, ov::PartialShape{-1});
    const auto in_frame_size = std::make_shared<Parameter>(step_size_type, ov::Shape{});
    const auto in_frame_step = std::make_shared<Parameter>(step_size_type, ov::Shape{});
    const auto in_signal_len = std::make_shared<Parameter>(step_size_type, ov::PartialShape{-1});

    constexpr bool center = false;
    constexpr bool normalized = false;
    const auto op = make_op(in_signal, in_window, in_frame_size, in_frame_step, in_signal_len, center, normalized);

    std::vector<StaticShape> static_input_shapes = {StaticShape{8, 9, 3, 2},
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
    ASSERT_EQ(static_output_shapes[0], (StaticShape{8, static_cast<size_t>(signal_len)}));
}

TEST_F(ISTFTShapeInferenceTest, inputs_const_3D_data_with_signal_len) {
    const auto data_type = element::f32;
    const auto step_size_type = element::i32;
    const auto in_signal = std::make_shared<Parameter>(data_type, ov::PartialShape{-1, -1, -1});
    const auto in_window = std::make_shared<Parameter>(data_type, ov::PartialShape{-1});

    int32_t frame_size = 16;
    int32_t frame_step = 16;
    int32_t signal_len = 60;

    const auto in_frame_size = std::make_shared<Constant>(step_size_type, ov::Shape{}, frame_size);
    const auto in_frame_step = std::make_shared<Constant>(step_size_type, ov::Shape{}, frame_step);
    const auto in_signal_len = std::make_shared<Constant>(step_size_type, ov::Shape{}, signal_len);

    constexpr bool center = false;
    constexpr bool normalized = false;
    const auto op = make_op(in_signal, in_window, in_frame_size, in_frame_step, in_signal_len, center, normalized);

    std::vector<StaticShape> static_input_shapes = {StaticShape{9, 3, 2},
                                                    StaticShape{16},
                                                    StaticShape{},
                                                    StaticShape{},
                                                    StaticShape{}};

    auto acc = make_tensor_accessor();
    auto static_output_shapes = shape_infer(op.get(), static_input_shapes, acc);
    ASSERT_EQ(static_output_shapes[0], StaticShape{static_cast<size_t>(signal_len)});
}

TEST_F(ISTFTShapeInferenceTest, frame_size_incompatible_value) {
    const auto data_type = element::f32;
    const auto step_size_type = element::i32;
    const auto in_data = std::make_shared<Parameter>(data_type, PartialShape{-1, -1, -1});
    const auto in_window = std::make_shared<Parameter>(data_type, PartialShape{-1});
    const auto in_frame_size = std::make_shared<Parameter>(step_size_type, ov::Shape{});
    const auto in_frame_step = std::make_shared<Parameter>(step_size_type, ov::Shape{});

    constexpr bool center = false;
    constexpr bool normalized = false;
    const auto op = make_op(in_data, in_window, in_frame_size, in_frame_step, center, normalized);

    std::vector<StaticShape> static_input_shapes = {StaticShape{9, 3, 2},
                                                    StaticShape{16},
                                                    StaticShape{},
                                                    StaticShape{},
                                                    StaticShape{}};
    int32_t frame_size = -1;
    int32_t frame_step = 16;

    auto const_data = std::unordered_map<size_t, Tensor>{{2, {element::i32, ov::Shape{}, &frame_size}},
                                                         {3, {element::i32, ov::Shape{}, &frame_step}}};
    auto acc = make_tensor_accessor(const_data);
    OV_EXPECT_THROW(std::ignore = shape_infer(op.get(), static_input_shapes, acc),
                    NodeValidationFailure,
                    HasSubstr("Provided frame size must be greater than zero, but got: -1"));
}

TEST_F(ISTFTShapeInferenceTest, frame_step_incompatible_value) {
    const auto data_type = element::f32;
    const auto step_size_type = element::i32;
    const auto in_data = std::make_shared<Parameter>(data_type, PartialShape{-1, -1, -1});
    const auto in_window = std::make_shared<Parameter>(data_type, PartialShape{-1});
    const auto in_frame_size = std::make_shared<Parameter>(step_size_type, ov::Shape{});
    const auto in_frame_step = std::make_shared<Parameter>(step_size_type, ov::Shape{});

    constexpr bool center = false;
    constexpr bool normalized = false;
    const auto op = make_op(in_data, in_window, in_frame_size, in_frame_step, center, normalized);

    std::vector<StaticShape> static_input_shapes = {StaticShape{9, 3, 2},
                                                    StaticShape{16},
                                                    StaticShape{},
                                                    StaticShape{},
                                                    StaticShape{}};
    int32_t frame_size = 16;
    int32_t frame_step = -1;

    auto const_data = std::unordered_map<size_t, Tensor>{{2, {element::i32, ov::Shape{}, &frame_size}},
                                                         {3, {element::i32, ov::Shape{}, &frame_step}}};
    auto acc = make_tensor_accessor(const_data);
    OV_EXPECT_THROW(std::ignore = shape_infer(op.get(), static_input_shapes, acc),
                    NodeValidationFailure,
                    HasSubstr("frame step must be greater than zero, but got: -1"));
}

TEST_F(ISTFTShapeInferenceTest, window_incompatible_dim_with_frame_size) {
    const auto data_type = element::f32;
    const auto step_size_type = element::i32;
    const auto in_data = std::make_shared<Parameter>(data_type, PartialShape{-1, -1, -1});
    const auto in_window = std::make_shared<Parameter>(data_type, PartialShape{-1});
    const auto in_frame_size = std::make_shared<Parameter>(step_size_type, ov::Shape{});
    const auto in_frame_step = std::make_shared<Parameter>(step_size_type, ov::Shape{});

    constexpr bool center = false;
    constexpr bool normalized = false;
    const auto op = make_op(in_data, in_window, in_frame_size, in_frame_step, center, normalized);

    std::vector<StaticShape> static_input_shapes = {StaticShape{9, 3, 2},
                                                    StaticShape{16},
                                                    StaticShape{},
                                                    StaticShape{},
                                                    StaticShape{}};
    int32_t frame_size = 8;
    int32_t frame_step = 4;

    auto const_data = std::unordered_map<size_t, Tensor>{{2, {element::i32, ov::Shape{}, &frame_size}},
                                                         {3, {element::i32, ov::Shape{}, &frame_step}}};
    auto acc = make_tensor_accessor(const_data);

    OV_EXPECT_THROW(std::ignore = shape_infer(op.get(), static_input_shapes, acc),
                    NodeValidationFailure,
                    HasSubstr("Window input dimension must be in range [1, 8]"));
}

TEST_F(ISTFTShapeInferenceTest, data_shape_incompatible_dim_with_frame_size) {
    const auto data_type = element::f32;
    const auto step_size_type = element::i32;
    const auto in_data = std::make_shared<Parameter>(data_type, PartialShape{-1, -1, -1});
    const auto in_window = std::make_shared<Parameter>(data_type, PartialShape{-1});
    const auto in_frame_size = std::make_shared<Parameter>(step_size_type, ov::Shape{});
    const auto in_frame_step = std::make_shared<Parameter>(step_size_type, ov::Shape{});

    constexpr bool center = false;
    constexpr bool normalized = false;
    const auto op = make_op(in_data, in_window, in_frame_size, in_frame_step, center, normalized);

    std::vector<StaticShape> static_input_shapes = {StaticShape{9, 3, 2},
                                                    StaticShape{16},
                                                    StaticShape{},
                                                    StaticShape{},
                                                    StaticShape{}};
    int32_t frame_size = 31;
    int32_t frame_step = 11;

    auto const_data = std::unordered_map<size_t, Tensor>{{2, {element::i32, ov::Shape{}, &frame_size}},
                                                         {3, {element::i32, ov::Shape{}, &frame_step}}};
    auto acc = make_tensor_accessor(const_data);

    OV_EXPECT_THROW(std::ignore = shape_infer(op.get(), static_input_shapes, acc),
                    NodeValidationFailure,
                    HasSubstr("The dimension at data_shape[-3] must be equal to: (frame_size // 2 + 1)"));
}