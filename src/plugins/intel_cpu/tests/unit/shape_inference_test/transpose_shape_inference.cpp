// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <transpose_shape_inference.hpp>

#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;

TEST(StaticShapeInferenceTest, TransposeTest) {
    auto p = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1});
    auto order = std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape{3}, std::vector<int32_t>{2, 1, 0});
    auto transpose = std::make_shared<op::v1::Transpose>(p, order);
    // Test StaticShape
    std::vector<StaticShape> static_input_shapes = {StaticShape{3, 4, 5}, StaticShape{3}},
            static_output_shapes = {StaticShape{}};
    shape_inference(transpose.get(), static_input_shapes, static_output_shapes);
    ASSERT_EQ(static_output_shapes[0], StaticShape({5, 4, 3}));
}
