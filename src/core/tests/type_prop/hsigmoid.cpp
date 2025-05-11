// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/hsigmoid.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/type_prop.hpp"

using namespace std;
using namespace ov;

TEST(type_prop, hsigmoid) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 6});
    auto hsigmoid_func = make_shared<op::v5::HSigmoid>(data);
    EXPECT_EQ(hsigmoid_func->get_element_type(), element::f32);
    EXPECT_EQ(hsigmoid_func->get_shape(), data->get_output_shape(0));
}

TEST(type_prop, hsigmoid_partial) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, Dimension::dynamic(), 6});
    auto hsigmoid_func = make_shared<op::v5::HSigmoid>(data);
    EXPECT_EQ(hsigmoid_func->get_element_type(), element::f32);
    ASSERT_TRUE(hsigmoid_func->get_output_partial_shape(0).same_scheme(data->get_output_partial_shape(0)));

    // rank unknown
    auto hsigmoid_partial =
        make_shared<op::v5::HSigmoid>(make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic()));
    ASSERT_TRUE(hsigmoid_partial->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, hsigmoid_partial_static_rank) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, Dimension::dynamic(), 6});
    auto hsigmoid_func = make_shared<op::v5::HSigmoid>(data);
    EXPECT_EQ(hsigmoid_func->get_element_type(), element::f32);
    ASSERT_TRUE(hsigmoid_func->get_output_partial_shape(0).same_scheme(data->get_output_partial_shape(0)));
    ASSERT_TRUE(hsigmoid_func->get_output_partial_shape(0).rank().is_static());
}
