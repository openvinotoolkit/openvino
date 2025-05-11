// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/mvn.hpp"

#include "common_test_utils/type_prop.hpp"

using namespace std;
using namespace ov;

// ------------------------------ V0 ------------------------------

TEST(type_prop, mvn) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 6});
    auto mvn_func = make_shared<op::v0::MVN>(data);
    EXPECT_EQ(mvn_func->get_element_type(), element::f32);
    EXPECT_EQ(mvn_func->get_shape(), (Shape{1, 3, 6}));
}

TEST(type_prop, mvn_partial) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, Dimension::dynamic(), 6});
    auto mvn_func = make_shared<op::v0::MVN>(data);
    EXPECT_EQ(mvn_func->get_element_type(), element::f32);
    EXPECT_EQ(mvn_func->get_reduction_axes(), (AxisSet{1, 2}));
    ASSERT_TRUE(mvn_func->get_output_partial_shape(0).same_scheme((PartialShape{1, Dimension::dynamic(), 6})));

    // across_channels = false
    EXPECT_EQ(make_shared<op::v0::MVN>(data, false)->get_reduction_axes(), (AxisSet{2}));

    // rank unknown
    auto mvn_partial =
        make_shared<op::v0::MVN>(make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic()));
    EXPECT_EQ(mvn_partial->get_reduction_axes(), AxisSet{});
    ASSERT_TRUE(mvn_partial->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

// ------------------------------ V6 ------------------------------

TEST(type_prop, mvn_6) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto axes = make_shared<ov::op::v0::Parameter>(element::i64, Shape{3});

    auto mvn_func = make_shared<op::v6::MVN>(data, axes, true, 1e-6f, op::MVNEpsMode::INSIDE_SQRT);
    EXPECT_EQ(mvn_func->get_element_type(), element::f32);
    EXPECT_EQ(mvn_func->get_shape(), (Shape{1, 2, 3, 4}));
}

TEST(type_prop, mvn_6_partial) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, Dimension::dynamic(), 5, 6});
    auto axes = make_shared<ov::op::v0::Parameter>(element::i64, Shape{3});
    auto mvn_func = make_shared<op::v6::MVN>(data, axes, true, 1e-6f, op::MVNEpsMode::INSIDE_SQRT);
    EXPECT_EQ(mvn_func->get_element_type(), element::f32);
    ASSERT_TRUE(mvn_func->get_output_partial_shape(0).same_scheme((PartialShape{1, Dimension::dynamic(), 5, 6})));

    // rank unknown
    auto mvn_partial =
        make_shared<op::v6::MVN>(make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic()),
                                 axes,
                                 true,
                                 1e-6f,
                                 op::MVNEpsMode::INSIDE_SQRT);
    ASSERT_TRUE(mvn_partial->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}
