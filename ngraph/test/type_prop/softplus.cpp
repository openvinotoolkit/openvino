// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, softplus)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 3, 6});
    auto softplus_func = make_shared<op::v4::SoftPlus>(data);
    EXPECT_EQ(softplus_func->get_element_type(), element::f32);
    EXPECT_EQ(softplus_func->get_shape(), (Shape{1, 3, 6}));
}

TEST(type_prop, softplus_partial)
{
    auto data = make_shared<op::Parameter>(element::f32, PartialShape{1, Dimension::dynamic(), 6});
    auto softplus_func = make_shared<op::v4::SoftPlus>(data);
    EXPECT_EQ(softplus_func->get_element_type(), element::f32);
    ASSERT_TRUE(softplus_func->get_output_partial_shape(0).same_scheme(
        (PartialShape{1, Dimension::dynamic(), 6})));

    // rank unknown
    auto softplus_partial = make_shared<op::v4::SoftPlus>(
        make_shared<op::Parameter>(element::f32, PartialShape::dynamic()));
    ASSERT_TRUE(softplus_partial->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, softplus_partial_static_rank)
{
    auto data = make_shared<op::Parameter>(element::f32, PartialShape{1, Dimension::dynamic(), 6});
    auto softplus_func = make_shared<op::v4::SoftPlus>(data);
    EXPECT_EQ(softplus_func->get_element_type(), element::f32);
    ASSERT_TRUE(softplus_func->get_output_partial_shape(0).same_scheme(
        (PartialShape{1, Dimension::dynamic(), 6})));
    ASSERT_TRUE(softplus_func->get_output_partial_shape(0).rank().is_static());
}
