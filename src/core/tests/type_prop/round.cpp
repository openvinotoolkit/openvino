// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/round.hpp"

#include "common_test_utils/type_prop.hpp"
#include "openvino/op/parameter.hpp"

using namespace std;
using namespace ov;

TEST(type_prop, rounding_to_even) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 6});
    auto round_func = make_shared<op::v5::Round>(data, op::v5::Round::RoundMode::HALF_TO_EVEN);
    EXPECT_EQ(round_func->get_element_type(), element::f32);
    EXPECT_EQ(round_func->get_shape(), (Shape{1, 3, 6}));
}

TEST(type_prop, rounding_away) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 6});
    auto round_func = make_shared<op::v5::Round>(data, op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO);
    EXPECT_EQ(round_func->get_element_type(), element::f32);
    EXPECT_EQ(round_func->get_shape(), (Shape{1, 3, 6}));
}

TEST(type_prop, rounding_to_even_partial) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, Dimension::dynamic(), 6});
    auto round_func = make_shared<op::v5::Round>(data, op::v5::Round::RoundMode::HALF_TO_EVEN);
    EXPECT_EQ(round_func->get_element_type(), element::f32);
    ASSERT_TRUE(round_func->get_output_partial_shape(0).same_scheme((PartialShape{1, Dimension::dynamic(), 6})));

    // rank unknown
    auto round_partial =
        make_shared<op::v5::Round>(make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic()),
                                   op::v5::Round::RoundMode::HALF_TO_EVEN);
    ASSERT_TRUE(round_partial->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, rounding_away_partial) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, Dimension::dynamic(), 6});
    auto round_func = make_shared<op::v5::Round>(data, op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO);
    EXPECT_EQ(round_func->get_element_type(), element::f32);
    ASSERT_TRUE(round_func->get_output_partial_shape(0).same_scheme((PartialShape{1, Dimension::dynamic(), 6})));

    // rank unknown
    auto round_partial =
        make_shared<op::v5::Round>(make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic()),
                                   op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO);
    ASSERT_TRUE(round_partial->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, rounding_to_even_partial_static_rank) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, Dimension::dynamic(), 6});
    auto round_func = make_shared<op::v5::Round>(data, op::v5::Round::RoundMode::HALF_TO_EVEN);
    EXPECT_EQ(round_func->get_element_type(), element::f32);
    ASSERT_TRUE(round_func->get_output_partial_shape(0).same_scheme((PartialShape{1, Dimension::dynamic(), 6})));
    ASSERT_TRUE(round_func->get_output_partial_shape(0).rank().is_static());
}

TEST(type_prop, rounding_away_partial_static_rank) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, Dimension::dynamic(), 6});
    auto round_func = make_shared<op::v5::Round>(data, op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO);
    EXPECT_EQ(round_func->get_element_type(), element::f32);
    ASSERT_TRUE(round_func->get_output_partial_shape(0).same_scheme((PartialShape{1, Dimension::dynamic(), 6})));
    ASSERT_TRUE(round_func->get_output_partial_shape(0).rank().is_static());
}
