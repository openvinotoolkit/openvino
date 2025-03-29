// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/sigmoid.hpp"

#include "common_test_utils/type_prop.hpp"

using namespace std;
using namespace ov;

TEST(type_prop, sigmoid) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 6});
    auto sigmoid_func = make_shared<op::v0::Sigmoid>(data);
    EXPECT_EQ(sigmoid_func->get_element_type(), element::f32);
    EXPECT_EQ(sigmoid_func->get_shape(), data->get_output_shape(0));
}

TEST(type_prop, sigmoid_partial) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, Dimension::dynamic(), 6});
    auto sigmoid_func = make_shared<op::v0::Sigmoid>(data);
    EXPECT_EQ(sigmoid_func->get_element_type(), element::f32);
    ASSERT_TRUE(sigmoid_func->get_output_partial_shape(0).same_scheme(data->get_output_partial_shape(0)));

    // rank unknown
    auto sigmoid_partial =
        make_shared<op::v0::Sigmoid>(make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic()));
    ASSERT_TRUE(sigmoid_partial->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, sigmoid_partial_static_rank) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, Dimension::dynamic(), 6});
    auto sigmoid_func = make_shared<op::v0::Sigmoid>(data);
    EXPECT_EQ(sigmoid_func->get_element_type(), element::f32);
    ASSERT_TRUE(sigmoid_func->get_output_partial_shape(0).same_scheme(data->get_output_partial_shape(0)));
    ASSERT_TRUE(sigmoid_func->get_output_partial_shape(0).rank().is_static());
}
