// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/non_zero.hpp"

#include "common_test_utils/type_prop.hpp"

using namespace std;
using namespace ov;

TEST(type_prop, non_zero) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 3, 224, 224});
    auto non_zero = make_shared<op::v3::NonZero>(data);
    EXPECT_EQ(non_zero->get_element_type(), element::i64);
    ASSERT_EQ(non_zero->get_output_partial_shape(0), (PartialShape{4, {0, 451584}}));
}

TEST(type_prop, non_zero_partial_input) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{{3, 4}, {5, 6}, {7, 8}});
    auto non_zero = make_shared<op::v3::NonZero>(data);
    EXPECT_EQ(non_zero->get_element_type(), element::i64);
    ASSERT_EQ(non_zero->get_output_partial_shape(0), (PartialShape{3, {0, 192}}));
}

TEST(type_prop, non_zero_partial_with_negative) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{{3, 4}, {5, 6}, -1});
    auto non_zero = make_shared<op::v3::NonZero>(data);
    EXPECT_EQ(non_zero->get_element_type(), element::i64);
    ASSERT_EQ(non_zero->get_output_partial_shape(0), (PartialShape{3, -1}));
}

TEST(type_prop, non_zero_dynamic) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto non_zero = make_shared<op::v3::NonZero>(data);
    EXPECT_EQ(non_zero->get_element_type(), element::i64);
    EXPECT_TRUE(
        non_zero->get_output_partial_shape(0).same_scheme(PartialShape{Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop, non_zero_output_type) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto non_zero = make_shared<op::v3::NonZero>(data, element::i32);

    ASSERT_EQ(non_zero->get_output_element_type(0), element::i32);
    ASSERT_EQ(non_zero->get_output_partial_shape(0), (PartialShape{4, {0, 24}}));
}

TEST(type_prop, non_zero_string_output_type) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto non_zero = make_shared<op::v3::NonZero>(data, "i32");

    ASSERT_EQ(non_zero->get_output_element_type(0), element::i32);
    ASSERT_EQ(non_zero->get_output_partial_shape(0), (PartialShape{4, {0, 24}}));
}

TEST(type_prop, non_zero_bool_input_type) {
    auto data = make_shared<ov::op::v0::Parameter>(element::boolean, Shape{1, 2, 3, 4});
    auto non_zero = make_shared<op::v3::NonZero>(data, element::i32);

    ASSERT_EQ(non_zero->get_output_element_type(0), element::i32);
    ASSERT_EQ(non_zero->get_output_partial_shape(0), (PartialShape{4, {0, 24}}));
}

TEST(type_prop, non_zero_fail_index_element_type) {
    // Deduce type
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    try {
        auto non_zero = make_shared<ov::op::v3::NonZero>(data, element::i16);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid output type not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Output type must be i32 or i64"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}
