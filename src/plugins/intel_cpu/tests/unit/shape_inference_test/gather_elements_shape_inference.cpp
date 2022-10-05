// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>

#include <gather_elements_shape_inference.hpp>
#include <openvino/op/ops.hpp>
#include <openvino/op/parameter.hpp>
#include <utils/shape_inference/shape_inference.hpp>
#include <utils/shape_inference/static_shape.hpp>

using namespace ov;
using namespace ov::intel_cpu;

TEST(StaticShapeInferenceTest, GatherElementsTest) {
    int64_t axis = -1;
    auto D = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto I = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1, -1, -1, -1});
    auto GE = std::make_shared<op::v6::GatherElements>(D, I, axis);
    // Test StaticShape
    std::vector<StaticShape> static_input_shapes = {StaticShape{300, 3, 10, 1}, StaticShape{300, 3, 10, 33333}},
                             static_output_shapes = {StaticShape{}};
    shape_inference(GE.get(), static_input_shapes, static_output_shapes);
    ASSERT_EQ(static_output_shapes[0], (StaticShape{300, 3, 10, 33333}));
}