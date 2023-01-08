// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/core/coordinate_diff.hpp>
#include <openvino/op/convolution.hpp>
#include <openvino/op/group_conv.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/relu.hpp>
#include <openvino/op/constant.hpp>
#include <utils/shape_inference/shape_inference.hpp>
#include <utils/shape_inference/static_shape.hpp>

using namespace ov;
using namespace ov::intel_cpu;

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

    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 6, 5, 5}, StaticShape{7, 6, 3, 3}},
                             static_output_shapes = {StaticShape{}};
    shape_inference(conv.get(), static_input_shapes, static_output_shapes);

    ASSERT_EQ(static_output_shapes[0], StaticShape({3, 7, 5, 5}));
}


TEST(StaticShapeInferenceTest, GroupConvolutionTest) {
    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto filters = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1, -1});

    auto conv =
        std::make_shared<op::v1::GroupConvolution>(data, filters, strides, pads_begin, pads_end, dilations, auto_pad);

    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 4, 5, 5}, StaticShape{2, 1, 2, 3, 3}},
                             static_output_shapes = {StaticShape{}};
    shape_inference(conv.get(), static_input_shapes, static_output_shapes);

    ASSERT_EQ(static_output_shapes[0], StaticShape({1, 2, 5, 5}));
}

TEST(StaticShapeInferenceTest, ConvolutionBackPropDataTest) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto filters = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});

    const Strides strides{2, 2};
    const Strides dilations{1, 1};
    const CoordinateDiff padding_begin{1, 1};
    const CoordinateDiff padding_end{1, 1};
    const CoordinateDiff output_padding{1, 1};
    const op::PadType auto_pad = op::PadType::SAME_LOWER;

    auto output_shape = std::make_shared<op::v0::Constant>(
            ov::element::i64, ov::Shape{2}, std::vector<int64_t>({3, 3}));
    auto conv = std::make_shared<op::v1::ConvolutionBackpropData>(data,
                                                                  filters,
                                                                  output_shape,
                                                                  strides,
                                                                  padding_begin,
                                                                  padding_end,
                                                                  dilations,
                                                                  auto_pad,
                                                                  output_padding);

    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 16, 2, 2}, StaticShape{16, 6, 3, 3}, StaticShape{2}},
                             static_output_shapes = {StaticShape{}};
    shape_inference(conv.get(), static_input_shapes, static_output_shapes);

    ASSERT_EQ(static_output_shapes[0], StaticShape({1, 6, 3, 3}));
}

TEST(StaticShapeInferenceTest, GroupConvolutionBackPropDataTest) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto filters = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1, -1});

    const Strides strides{2, 2};
    const Strides dilations{1, 1};
    const CoordinateDiff padding_begin{1, 1};
    const CoordinateDiff padding_end{1, 1};
    const CoordinateDiff output_padding{1, 1};
    const op::PadType auto_pad = op::PadType::SAME_LOWER;

    auto output_shape = std::make_shared<op::v0::Constant>(
            ov::element::i64, ov::Shape{2}, std::vector<int64_t>({3, 3}));
    auto conv = std::make_shared<op::v1::GroupConvolutionBackpropData>(data,
                                                                  filters,
                                                                  output_shape,
                                                                  strides,
                                                                  padding_begin,
                                                                  padding_end,
                                                                  dilations,
                                                                  auto_pad,
                                                                  output_padding);

    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 16, 2, 2}, StaticShape{4, 4, 6, 3, 3}, StaticShape{2}},
                             static_output_shapes = {StaticShape{}};
    shape_inference(conv.get(), static_input_shapes, static_output_shapes);

    ASSERT_EQ(static_output_shapes[0], StaticShape({1, 24, 3, 3}));
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
        shape_inference(conv.get(), static_input_shapes, static_output_shapes);
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