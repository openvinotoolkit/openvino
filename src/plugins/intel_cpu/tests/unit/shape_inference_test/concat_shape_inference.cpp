// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <concat_shape_inference.hpp>

#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;

TEST(StaticShapeInferenceTest, ConcatTest) {
    auto P1 = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1});
    auto P2 = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1});
    auto concat = std::make_shared<op::v0::Concat>(NodeVector{P1, P2}, 1);
    // Test StaticShape
    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 4, 5}, StaticShape{3, 2, 5}},
                             static_output_shapes = {StaticShape{}};
    shape_inference(concat.get(), static_input_shapes, static_output_shapes);
    ASSERT_EQ(static_output_shapes[0], StaticShape({3, 6, 5}));
}

TEST(StaticShapeInferenceTest, ConcatNegativeAxisTest) {
    auto P1 = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1});
    auto P2 = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1});
    auto concat = std::make_shared<op::v0::Concat>(NodeVector{P1, P2}, -3);
    // Test StaticShape
    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 4, 5}, StaticShape{2, 4, 5}},
                             static_output_shapes = {StaticShape{}};
    shape_inference(concat.get(), static_input_shapes, static_output_shapes);
    ASSERT_EQ(static_output_shapes[0], StaticShape({5, 4, 5}));
}
