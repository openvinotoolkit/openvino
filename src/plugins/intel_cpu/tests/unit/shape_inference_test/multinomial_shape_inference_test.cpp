// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "multinomial_shape_inference.hpp"

#include <gtest/gtest.h>
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;

TEST(StaticShapeInferenceTest, MultinomialDefaultShapeInferenceTest) {
    auto probs = std::make_shared<op::v0::Parameter>(element::f32, Shape{4});
    auto num_elements = std::make_shared<op::v0::Parameter>(element::f32, Shape{1});
    auto multinomial = std::make_shared<op::v13::Multinomial>(probs, num_elements, element::i32, false, false, 0, 0);
    // Test Static Shape
    std::vector<StaticShape> static_input_shapes = {StaticShape{4}, StaticShape{1}};
    int32_t num_elements_val = 2;
    auto const_data = std::map<size_t, HostTensorPtr>{
        {1, std::make_shared<HostTensor>(element::i32, Shape{1}, &num_elements_val)}
    };
    auto acc = make_tensor_accessor(const_data);
    auto static_output_shapes = shape_infer(multinomial.get(), static_input_shapes, acc);
    ASSERT_EQ(static_output_shapes[0], StaticShape({2}));
}
