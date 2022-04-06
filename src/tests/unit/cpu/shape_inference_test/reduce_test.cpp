// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/core/coordinate_diff.hpp>
#include <openvino/op/reduce_mean.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/constant.hpp>
#include <utils/shape_inference/shape_inference.hpp>
#include <utils/shape_inference/static_shape.hpp>

using namespace ov;
using namespace ov::intel_cpu;

TEST(StaticShapeInferenceTest, ReduceTest) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto axes = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{2}, std::vector<int32_t>{1, 3});

    auto reduce =
            std::make_shared<op::v1::ReduceMean>(data, axes, true);

    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 6, 5, 5}, StaticShape{2}},
            static_output_shapes = {StaticShape{}};
    shape_inference(reduce.get(), static_input_shapes, static_output_shapes);

    ASSERT_EQ(static_output_shapes[0], StaticShape({3, 1, 5, 1}));
}
