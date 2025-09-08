// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/mish.hpp"

#include "common_test_utils/type_prop.hpp"

using namespace std;
using namespace ov;

TEST(type_prop, mish) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 6});
    auto mish_func = make_shared<op::v4::Mish>(data);
    EXPECT_EQ(mish_func->get_element_type(), element::f32);
    EXPECT_EQ(mish_func->get_shape(), (Shape{1, 3, 6}));
}

TEST(type_prop, mish_partial) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, Dimension::dynamic(), 6});
    auto mish_func = make_shared<op::v4::Mish>(data);
    EXPECT_EQ(mish_func->get_element_type(), element::f32);
    ASSERT_TRUE(mish_func->get_output_partial_shape(0).same_scheme((PartialShape{1, Dimension::dynamic(), 6})));

    // rank unknown
    auto mish_partial =
        make_shared<op::v4::Mish>(make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic()));
    ASSERT_TRUE(mish_partial->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, mish_partial_static_rank) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, Dimension::dynamic(), 6});
    auto mish_func = make_shared<op::v4::Mish>(data);
    EXPECT_EQ(mish_func->get_element_type(), element::f32);
    ASSERT_TRUE(mish_func->get_output_partial_shape(0).same_scheme((PartialShape{1, Dimension::dynamic(), 6})));
    ASSERT_TRUE(mish_func->get_output_partial_shape(0).rank().is_static());
}

TEST(type_prop, mish_incompatible_dtype_i32) {
    auto data = make_shared<ov::op::v0::Parameter>(element::i32, Shape{1, 3, 6});
    ASSERT_THROW(const auto unused = std::make_shared<op::v4::Mish>(data), ov::NodeValidationFailure);
}

TEST(type_prop, mish_incompatible_dtype_u32) {
    auto data = make_shared<ov::op::v0::Parameter>(element::u32, Shape{1, 3, 6});
    ASSERT_THROW(const auto unused = std::make_shared<op::v4::Mish>(data), ov::NodeValidationFailure);
}

TEST(type_prop, mish_incompatible_dtype_boolean) {
    auto data = make_shared<ov::op::v0::Parameter>(element::boolean, Shape{1, 3, 6});
    ASSERT_THROW(const auto unused = std::make_shared<op::v4::Mish>(data), ov::NodeValidationFailure);
}
