// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/core/coordinate_diff.hpp>
#include <openvino/op/convolution.hpp>
#include <openvino/op/ops.hpp>
#include <openvino/op/parameter.hpp>
#include <topk_shape_inference.hpp>

#include "utils/shape_inference/static_shape.hpp"

using namespace ov;

TEST(StaticShapeInferenceTest, TopKv3) {
    const auto data_shape = PartialShape::dynamic();
    const auto data = std::make_shared<op::v0::Parameter>(element::f32, data_shape);
    const auto k = op::v0::Constant::create(element::i64, ov::Shape{}, {2});
    const int64_t axis = 1;

    const auto topk = std::make_shared<op::v3::TopK>(data, k, axis, "max", "value");

    std::vector<PartialShape> input_shapes = {PartialShape{1, 10, 100}, PartialShape{}};
    std::vector<PartialShape> output_shapes = {PartialShape{}, PartialShape{}};
    shape_infer(topk.get(), input_shapes, output_shapes);

    ASSERT_EQ(output_shapes[0], PartialShape({1, 2, 100}));
    ASSERT_EQ(output_shapes[1], PartialShape({1, 2, 100}));

    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 10, 100}, StaticShape{}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{}, StaticShape{}};
    shape_infer(topk.get(), static_input_shapes, static_output_shapes);

    ASSERT_EQ(static_output_shapes[0], StaticShape({1, 2, 100}));
    ASSERT_EQ(static_output_shapes[1], StaticShape({1, 2, 100}));
}
