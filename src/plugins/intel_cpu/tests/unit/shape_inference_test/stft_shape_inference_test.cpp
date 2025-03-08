// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "stft_shape_inference.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using ov::op::v0::Constant;
using ov::op::v0::Parameter;
using testing::HasSubstr;

class STFTShapeInferenceTest : public OpStaticShapeInferenceTest<op::v15::STFT> {};

TEST_F(STFTShapeInferenceTest, all_input_as_params_1D_signal) {
    const auto data_type = element::f32;
    const auto step_size_type = element::i32;
    const auto in_signal = std::make_shared<Parameter>(data_type, ov::PartialShape{-1});
    const auto in_window = std::make_shared<Parameter>(data_type, ov::PartialShape{-1});
    const auto in_frame_size = std::make_shared<Parameter>(step_size_type, ov::Shape{});
    const auto in_frame_step = std::make_shared<Parameter>(step_size_type, ov::Shape{});
    const auto op = make_op(in_signal, in_window, in_frame_size, in_frame_step, true);

    std::vector<StaticShape> static_input_shapes = {StaticShape{48}, StaticShape{16}, StaticShape{}, StaticShape{}};
    int32_t frame_size = 16;
    int32_t frame_step = 16;

    auto const_data = std::unordered_map<size_t, Tensor>{{2, {element::i32, ov::Shape{}, &frame_size}},
                                                         {3, {element::i32, ov::Shape{}, &frame_step}}};
    auto acc = make_tensor_accessor(const_data);
    auto static_output_shapes = shape_infer(op.get(), static_input_shapes, acc);
    ASSERT_EQ(static_output_shapes[0], StaticShape({9, 3, 2}));
}

TEST_F(STFTShapeInferenceTest, all_input_as_params) {
    const auto data_type = element::f32;
    const auto step_size_type = element::i32;
    const auto in_signal = std::make_shared<Parameter>(data_type, ov::PartialShape{-1, -1});
    const auto in_window = std::make_shared<Parameter>(data_type, ov::PartialShape{-1});
    const auto in_frame_size = std::make_shared<Parameter>(step_size_type, ov::Shape{});
    const auto in_frame_step = std::make_shared<Parameter>(step_size_type, ov::Shape{});
    const auto op = make_op(in_signal, in_window, in_frame_size, in_frame_step, true);

    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 48}, StaticShape{16}, StaticShape{}, StaticShape{}};
    int32_t frame_size = 16;
    int32_t frame_step = 16;

    auto const_data = std::unordered_map<size_t, Tensor>{{2, {element::i32, ov::Shape{}, &frame_size}},
                                                         {3, {element::i32, ov::Shape{}, &frame_step}}};
    auto acc = make_tensor_accessor(const_data);
    auto static_output_shapes = shape_infer(op.get(), static_input_shapes, acc);
    ASSERT_EQ(static_output_shapes[0], StaticShape({1, 9, 3, 2}));
}

TEST_F(STFTShapeInferenceTest, all_input_as_params_equal_dims) {
    const auto data_type = element::f32;
    const auto step_size_type = element::i32;
    const auto in_signal = std::make_shared<Parameter>(data_type, ov::PartialShape{-1, -1});
    const auto in_window = std::make_shared<Parameter>(data_type, ov::PartialShape{-1});
    const auto in_frame_size = std::make_shared<Parameter>(step_size_type, ov::Shape{});
    const auto in_frame_step = std::make_shared<Parameter>(step_size_type, ov::Shape{});
    const auto op = make_op(in_signal, in_window, in_frame_size, in_frame_step, true);

    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 16}, StaticShape{16}, StaticShape{}, StaticShape{}};
    int32_t frame_size = 16;
    int32_t frame_step = 16;

    auto const_data = std::unordered_map<size_t, Tensor>{{2, {element::i32, ov::Shape{}, &frame_size}},
                                                         {3, {element::i32, ov::Shape{}, &frame_step}}};
    auto acc = make_tensor_accessor(const_data);
    auto static_output_shapes = shape_infer(op.get(), static_input_shapes, acc);
    ASSERT_EQ(static_output_shapes[0], StaticShape({1, 9, 1, 2}));
}

TEST_F(STFTShapeInferenceTest, frame_inputs_as_const) {
    const auto data_type = element::f32;
    const auto step_size_type = element::i32;
    const auto in_signal = std::make_shared<Parameter>(data_type, ov::PartialShape{-1, -1});
    const auto in_window = std::make_shared<Parameter>(data_type, ov::PartialShape{-1});

    int32_t frame_size = 16;
    int32_t frame_step = 16;

    const auto in_frame_size = std::make_shared<Constant>(step_size_type, ov::Shape{}, frame_size);
    const auto in_frame_step = std::make_shared<Constant>(step_size_type, ov::Shape{}, frame_step);
    const auto op = make_op(in_signal, in_window, in_frame_size, in_frame_step, true);

    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 48}, StaticShape{16}, StaticShape{}, StaticShape{}};

    auto acc = make_tensor_accessor();
    auto static_output_shapes = shape_infer(op.get(), static_input_shapes, acc);
    ASSERT_EQ(static_output_shapes[0], StaticShape({1, 9, 3, 2}));
}

TEST_F(STFTShapeInferenceTest, frame_size_incompatible_value_big) {
    const auto data_type = element::f32;
    const auto step_size_type = element::i32;
    const auto signal = std::make_shared<Parameter>(data_type, PartialShape{-1, -1});
    const auto window = std::make_shared<Parameter>(data_type, PartialShape{-1});
    const auto in_frame_size = std::make_shared<Parameter>(step_size_type, ov::Shape{});
    const auto in_frame_step = std::make_shared<Parameter>(step_size_type, ov::Shape{});

    const auto op = make_op(signal, window, in_frame_size, in_frame_step, true);

    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 48}, StaticShape{16}, StaticShape{}, StaticShape{}};
    int32_t frame_size = 49;
    int32_t frame_step = 16;

    auto const_data = std::unordered_map<size_t, Tensor>{{2, {element::i32, ov::Shape{}, &frame_size}},
                                                         {3, {element::i32, ov::Shape{}, &frame_step}}};
    auto acc = make_tensor_accessor(const_data);
    OV_EXPECT_THROW(std::ignore = shape_infer(op.get(), static_input_shapes, acc),
                    NodeValidationFailure,
                    HasSubstr("Provided frame size is 49 but must be in range [1, 48]"));
}

TEST_F(STFTShapeInferenceTest, frame_size_incompatible_value_small) {
    const auto data_type = element::f32;
    const auto step_size_type = element::i32;
    const auto signal = std::make_shared<Parameter>(data_type, PartialShape{-1, -1});
    const auto window = std::make_shared<Parameter>(data_type, PartialShape{-1});
    const auto in_frame_size = std::make_shared<Parameter>(step_size_type, ov::Shape{});
    const auto in_frame_step = std::make_shared<Parameter>(step_size_type, ov::Shape{});

    const auto op = make_op(signal, window, in_frame_size, in_frame_step, true);

    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 48}, StaticShape{16}, StaticShape{}, StaticShape{}};
    int32_t frame_size = -1;
    int32_t frame_step = 16;

    auto const_data = std::unordered_map<size_t, Tensor>{{2, {element::i32, ov::Shape{}, &frame_size}},
                                                         {3, {element::i32, ov::Shape{}, &frame_step}}};
    auto acc = make_tensor_accessor(const_data);
    OV_EXPECT_THROW(std::ignore = shape_infer(op.get(), static_input_shapes, acc),
                    NodeValidationFailure,
                    HasSubstr("Provided frame size is -1 but must be in range [1, 48]"));
}

TEST_F(STFTShapeInferenceTest, frame_step_incompatible_value) {
    const auto data_type = element::f32;
    const auto step_size_type = element::i32;
    const auto signal = std::make_shared<Parameter>(data_type, PartialShape{-1, -1});
    const auto window = std::make_shared<Parameter>(data_type, PartialShape{-1});
    const auto in_frame_size = std::make_shared<Parameter>(step_size_type, ov::Shape{});
    const auto in_frame_step = std::make_shared<Parameter>(step_size_type, ov::Shape{});

    const auto op = make_op(signal, window, in_frame_size, in_frame_step, true);

    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 48}, StaticShape{16}, StaticShape{}, StaticShape{}};
    int32_t frame_size = 16;
    int32_t frame_step = -1;

    auto const_data = std::unordered_map<size_t, Tensor>{{2, {element::i32, ov::Shape{}, &frame_size}},
                                                         {3, {element::i32, ov::Shape{}, &frame_step}}};
    auto acc = make_tensor_accessor(const_data);
    OV_EXPECT_THROW(std::ignore = shape_infer(op.get(), static_input_shapes, acc),
                    NodeValidationFailure,
                    HasSubstr("Provided frame step is -1 but must be greater than zero"));
}

TEST_F(STFTShapeInferenceTest, window_incompatible_dim_with_frame_size) {
    const auto data_type = element::f32;
    const auto step_size_type = element::i32;
    const auto signal = std::make_shared<Parameter>(data_type, PartialShape{-1, -1});
    const auto window = std::make_shared<Parameter>(data_type, PartialShape{-1});
    const auto in_frame_size = std::make_shared<Parameter>(step_size_type, ov::Shape{});
    const auto in_frame_step = std::make_shared<Parameter>(step_size_type, ov::Shape{});

    const auto op = make_op(signal, window, in_frame_size, in_frame_step, true);

    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 48}, StaticShape{16}, StaticShape{}, StaticShape{}};
    int32_t frame_size = 8;
    int32_t frame_step = 4;

    auto const_data = std::unordered_map<size_t, Tensor>{{2, {element::i32, ov::Shape{}, &frame_size}},
                                                         {3, {element::i32, ov::Shape{}, &frame_step}}};
    auto acc = make_tensor_accessor(const_data);

    OV_EXPECT_THROW(std::ignore = shape_infer(op.get(), static_input_shapes, acc),
                    NodeValidationFailure,
                    HasSubstr("Window input dimension must be in range [1, 8]"));
}
