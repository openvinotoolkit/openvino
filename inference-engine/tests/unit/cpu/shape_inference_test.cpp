// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <openvino/core/coordinate_diff.hpp>
#include <openvino/op/convolution.hpp>
#include <openvino/op/parameter.hpp>
#include <convolution_shape_inference.hpp>
#include <assign_shape_inference.hpp>
#include <read_value_shape_inference.hpp>
#include <openvino/op/ops.hpp>
#include "utils/shape_inference/static_shape.hpp"

using namespace ov;

TEST(StaticShapeInferenceTest, ConvolutionTest) {
    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto filters = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});

    auto conv =
            std::make_shared<op::v1::Convolution>(data, filters, strides, pads_begin, pads_end, dilations, auto_pad);

    std::vector<PartialShape> input_shapes = {PartialShape{3, 6, 5, 5}, PartialShape{7, 6, 3, 3}}, output_shapes = {PartialShape{}};
    shape_infer(conv.get(), input_shapes, output_shapes);

    ASSERT_EQ(output_shapes[0], PartialShape({3, 7, 5, 5}));
    ASSERT_EQ(conv->get_pads_begin(), (CoordinateDiff{1, 1}));
    ASSERT_EQ(conv->get_pads_end(), (CoordinateDiff{1, 1}));

    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 6, 5, 5}, StaticShape{7, 6, 3, 3}}, static_output_shapes = {StaticShape{}};
    shape_infer(conv.get(), static_input_shapes, static_output_shapes);

    ASSERT_EQ(static_output_shapes[0], StaticShape({3, 7, 5, 5}));
    ASSERT_EQ(conv->get_pads_begin(), (CoordinateDiff{1, 1}));
    ASSERT_EQ(conv->get_pads_end(), (CoordinateDiff{1, 1}));
}

TEST(StaticShapeInferenceTest, AssignTest) {
  auto input = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
  auto read_value = std::make_shared<op::v3::ReadValue>(input, "variable_id");
  auto assign = std::make_shared<op::v3::Assign>(read_value, "variable_id");
  //Test PartialShape
  std::vector<PartialShape> input_shapes = {PartialShape{1, 2, 64, 64}},
                            output_shapes = {PartialShape{}};
  shape_infer(assign.get(), input_shapes, output_shapes);
  ASSERT_EQ(output_shapes[0], (PartialShape{1, 2, 64, 64}));
  //Test StaticShape
  std::vector<StaticShape> static_input_shapes = {StaticShape{1, 2, 64, 64}},
                           static_output_shapes = {StaticShape{}};
  shape_infer(assign.get(), static_input_shapes, static_output_shapes);
  ASSERT_EQ(static_input_shapes[0], (StaticShape{1, 2, 64, 64}));
}

TEST(StaticShapeInferenceTest, ReadValueTest) {
  auto input = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
  auto read_value = std::make_shared<op::v3::ReadValue>(input, "variable_id");
  //Test PartialShape
  std::vector<PartialShape> input_shapes = {PartialShape{1, 2, 64, 64}},
                            output_shapes = {PartialShape{}};
  shape_infer(read_value.get(), input_shapes, output_shapes);
  ASSERT_EQ(output_shapes[0], (PartialShape{1, 2, 64, 64}));
  //Test StaticShape
  std::vector<StaticShape> static_input_shapes = {StaticShape{1, 2, 64, 64}},
                           static_output_shapes = {StaticShape{}};
  shape_infer(read_value.get(), static_input_shapes, static_output_shapes);
  ASSERT_EQ(static_output_shapes[0], (StaticShape{1, 2, 64, 64}));
}

#if 0
TEST(StaticShapeInferenceTest, ConvolutionTimeTest) {
    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto auto_pad = op::PadType::SAME_LOWER;
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{3, 6, 5, 5});
    auto filters = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{7, 6, 3, 3});
    auto conv =
            std::make_shared<op::v1::Convolution>(data, filters, strides, pads_begin, pads_end, dilations, auto_pad);
    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 6, 5, 5}, StaticShape{7, 6, 3, 3}}, static_output_shapes = {StaticShape{}};

    auto before = std::chrono::high_resolution_clock::now();
    auto after = std::chrono::high_resolution_clock::now();

    std::cout << conv << std::endl;
    auto convolution_time_sum = 0;
    for (size_t i = 0; i < 10; ++i) {
        before = std::chrono::high_resolution_clock::now();
        shape_infer(conv.get(), static_input_shapes, static_output_shapes);
        after = std::chrono::high_resolution_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::nanoseconds>(after - before).count();
        std::cout << diff << " ns" << std::endl;
        convolution_time_sum += diff;
    }

    // other operation creation and time measurements: ReLU is an example
    auto relu = std::make_shared<op::v0::Relu>(data);
    std::cout << relu << std::endl;
    auto other_op_time_sum = 0;
    for (size_t i = 0; i < 10; ++i) {
        before = std::chrono::high_resolution_clock::now();
        relu->validate_and_infer_types();
        after = std::chrono::high_resolution_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::nanoseconds>(after - before).count();
        std::cout << diff << " ns" << std::endl;
        other_op_time_sum += diff;
    }
    std::cout << (convolution_time_sum >= other_op_time_sum ? "ON PAR WITH CONVOLUTION: " : "LONGER THAN CONVOLUTION ")
              << 1. * other_op_time_sum / convolution_time_sum << std::endl;
}
#endif