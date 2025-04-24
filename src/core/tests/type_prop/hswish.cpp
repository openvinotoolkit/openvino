// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/hswish.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/type_prop.hpp"

using namespace std;
using namespace ov;

TEST(type_prop, hswish) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 6});
    auto hswish_func = make_shared<op::v4::HSwish>(data);
    EXPECT_EQ(hswish_func->get_element_type(), element::f32);
    EXPECT_EQ(hswish_func->get_shape(), data->get_output_shape(0));
}

TEST(type_prop, hswish_partial) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, Dimension::dynamic(), 6});
    auto hswish_func = make_shared<op::v4::HSwish>(data);
    EXPECT_EQ(hswish_func->get_element_type(), element::f32);
    ASSERT_TRUE(hswish_func->get_output_partial_shape(0).same_scheme(data->get_output_partial_shape(0)));

    // rank unknown
    auto hswish_partial =
        make_shared<op::v4::HSwish>(make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic()));
    ASSERT_TRUE(hswish_partial->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, hswish_partial_static_rank) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, Dimension::dynamic(), 6});
    auto hswish_func = make_shared<op::v4::HSwish>(data);
    EXPECT_EQ(hswish_func->get_element_type(), element::f32);
    ASSERT_TRUE(hswish_func->get_output_partial_shape(0).same_scheme(data->get_output_partial_shape(0)));
    ASSERT_TRUE(hswish_func->get_output_partial_shape(0).rank().is_static());
}
