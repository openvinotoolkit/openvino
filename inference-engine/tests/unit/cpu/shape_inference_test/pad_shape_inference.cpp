// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/core/coordinate_diff.hpp>
#include <openvino/op/convolution.hpp>
#include <openvino/op/ops.hpp>
#include <openvino/op/parameter.hpp>
#include <pad_shape_inference.hpp>

#include "utils/shape_inference/static_shape.hpp"

using namespace ov;

TEST(StaticShapeInferenceTest, Padv1) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());

    const auto pads_begin = ov::op::v0::Constant::create(element::i64, ov::Shape{4}, {3, 2, 1, 0});
    const auto pads_end = ov::op::v0::Constant::create(element::i64, ov::Shape{4}, {0, 1, 2, 3});
    const auto pad_val = ov::op::v0::Constant::create(element::f32, ov::Shape{}, {2112});

    const auto pad = std::make_shared<ov::op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::CONSTANT);
    auto f = std::make_shared<Function>(pad, ParameterVector{data});

    std::vector<PartialShape> input_shapes = {PartialShape{3, 6, 5, 5}, ov::Shape{4}, ov::Shape{4}, ov::Shape{}};
    std::vector<PartialShape> output_shapes = {PartialShape::dynamic()};
    shape_infer(pad.get(), input_shapes, output_shapes);

    ASSERT_EQ(output_shapes.size(), 1);
    ASSERT_EQ(output_shapes[0], PartialShape({6, 9, 8, 8}));

    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 6, 5, 5},
                                                    StaticShape{4},
                                                    StaticShape{4},
                                                    StaticShape(std::initializer_list<StaticDimension>{})};
    std::vector<StaticShape> static_output_shapes = {StaticShape{}};
    shape_infer(pad.get(), static_input_shapes, static_output_shapes);

    ASSERT_EQ(static_output_shapes.size(), 1);
    ASSERT_EQ(static_output_shapes[0], StaticShape({6, 9, 8, 8}));
}
