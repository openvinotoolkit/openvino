// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <gather_shape_inference.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/ops.hpp>
#include <openvino/op/parameter.hpp>

#include "utils/shape_inference/static_shape.hpp"

using namespace ov;

TEST(StaticShapeInferenceTest, GatherTest) {
    auto P = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1});
    auto I = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1, -1});
    auto A = op::v0::Constant::create(element::i64, Shape{}, {0});
    auto G = std::make_shared<op::v1::Gather>(P, I, A);
    std::vector<PartialShape> input_shapes = {PartialShape{3, 2}, PartialShape{2, 2}, PartialShape{1}},
                              output_shapes = {PartialShape{}};
    shape_infer(G.get(), input_shapes, output_shapes, {});
    ASSERT_EQ(output_shapes[0], (PartialShape{2, 2, 2}));
    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 2}, StaticShape{2, 2}, StaticShape{1}},
                             static_output_shapes = {StaticShape{}};
    shape_infer(G.get(), static_input_shapes, static_output_shapes, {});
    ASSERT_EQ(static_output_shapes[0], (StaticShape{2, 2, 2}));
}