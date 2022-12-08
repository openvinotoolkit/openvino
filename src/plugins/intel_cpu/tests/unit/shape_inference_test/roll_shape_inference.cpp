// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/op/constant.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/roll.hpp>
#include <utils/shape_inference/shape_inference.hpp>
#include <utils/shape_inference/static_shape.hpp>

using namespace ov::intel_cpu;

TEST(StaticShapeInferenceTest, RollTest) {
    auto arg =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic()});
    auto shift = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{ov::Dimension::dynamic()});
    auto axes = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{ov::Dimension::dynamic()});

    auto roll = std::make_shared<ov::op::v7::Roll>(arg, shift, axes);

    int32_t axes_val[] = {0, 1, -1};
    auto axes_tensor = std::make_shared<ngraph::runtime::HostTensor>(ov::element::i32, ov::Shape{3}, axes_val);

    std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>> constant_data;
    constant_data[2] = axes_tensor;

    const std::vector<StaticShape> input_shapes = {StaticShape{3, 3, 3},
                                                   StaticShape{3},
                                                   StaticShape{3}};
    std::vector<StaticShape> output_shapes = {StaticShape{}};
    shape_inference(roll.get(), input_shapes, output_shapes, constant_data);
    ASSERT_EQ(output_shapes[0], input_shapes[0]);
}

TEST(StaticShapeInferenceTest, RollTestWithConstAxis) {
    auto arg =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic()});
    auto shift = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{ov::Dimension::dynamic()});
    auto axes = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{3}, std::vector<int32_t>{0, 1, -1});
    auto roll = std::make_shared<ov::op::v7::Roll>(arg, shift, axes);

    const std::vector<StaticShape> input_shapes = {StaticShape{3, 3, 3},
                                                   StaticShape{3},
                                                   StaticShape{3}};
    std::vector<StaticShape> output_shapes = {StaticShape{}};
    shape_inference(roll.get(), input_shapes, output_shapes);
    ASSERT_EQ(output_shapes[0], input_shapes[0]);
}
