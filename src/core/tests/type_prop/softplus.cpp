// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/softplus.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/type_prop.hpp"

using namespace std;
using namespace ov;

TEST(type_prop, softplus) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 6});
    auto softplus_func = make_shared<op::v4::SoftPlus>(data);
    EXPECT_EQ(softplus_func->get_element_type(), element::f32);
    EXPECT_EQ(softplus_func->get_shape(), (Shape{1, 3, 6}));
}

TEST(type_prop, softplus_partial) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, Dimension::dynamic(), 6});
    auto softplus_func = make_shared<op::v4::SoftPlus>(data);
    EXPECT_EQ(softplus_func->get_element_type(), element::f32);
    ASSERT_TRUE(softplus_func->get_output_partial_shape(0).same_scheme((PartialShape{1, Dimension::dynamic(), 6})));

    // rank unknown
    auto softplus_partial =
        make_shared<op::v4::SoftPlus>(make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic()));
    ASSERT_TRUE(softplus_partial->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, softplus_partial_static_rank) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, Dimension::dynamic(), 6});
    auto softplus_func = make_shared<op::v4::SoftPlus>(data);
    EXPECT_EQ(softplus_func->get_element_type(), element::f32);
    ASSERT_TRUE(softplus_func->get_output_partial_shape(0).same_scheme((PartialShape{1, Dimension::dynamic(), 6})));
    ASSERT_TRUE(softplus_func->get_output_partial_shape(0).rank().is_static());
}

TEST(type_prop, softplus_invalid_element_type) {
    auto data = make_shared<ov::op::v0::Parameter>(element::i32, Shape{2, 2});

    try {
        auto softplus = make_shared<op::v4::SoftPlus>(data);
        // Input element type is boolean
        FAIL() << "Invalid int element type for input not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Input element type must be float");
    } catch (...) {
        FAIL() << "Numeric element type node validation check failed for unexpected reason";
    }
}
