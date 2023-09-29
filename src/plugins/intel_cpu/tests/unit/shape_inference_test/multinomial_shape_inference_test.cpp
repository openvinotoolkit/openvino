// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "multinomial_shape_inference.hpp"

#include <gtest/gtest.h>

#include "dimension_util.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;

TEST(StaticShapeInferenceTest, MultinomialDefaultShapeInferenceTest) {
    auto param0 = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{4});
    auto param1 = ov::op::v0::Constant::create(element::i32, Shape{1}, {2});
    auto multinomial = std::make_shared<op::v13::Multinomial>(param0, param1, ov::element::i32, false, false, 0, 0);
    // Test Static Shape
    std::vector<StaticShape> static_input_shapes = {StaticShape{4}, StaticShape{1}};
    auto const_data = std::unordered_map<size_t, ov::Tensor>{{1, {element::i32, Shape{1}, {2}}}};
    auto acc = make_tensor_accessor(const_data);
    auto static_output_shapes = shape_infer(multinomial.get(), static_input_shapes, acc);
    ASSERT_EQ(static_output_shapes[0], StaticShape({2}));
}