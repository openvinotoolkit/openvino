// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/constant.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/type_prop.hpp"

using namespace std;

TEST(type_prop, scalar_constant_deduce_float32) {
    auto c = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {208});
    ASSERT_EQ(c->get_element_type(), ov::element::f32);
    ASSERT_EQ(c->get_shape(), (ov::Shape{}));
}

TEST(type_prop, scalar_constant_deduce_bool) {
    auto c = ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{}, {1});
    ASSERT_EQ(c->get_element_type(), ov::element::boolean);
    ASSERT_EQ(c->get_shape(), (ov::Shape{}));
}

TEST(type_prop, tensor_constant_deduce_float32) {
    auto c = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{2, 2}, {208, 208, 208, 208});
    ASSERT_EQ(c->get_element_type(), ov::element::f32);
    ASSERT_EQ(c->get_shape(), (ov::Shape{2, 2}));
}

TEST(type_prop, tensor_constant_deduce_bool) {
    auto c = ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{2, 2}, {1, 1, 1, 1});
    ASSERT_EQ(c->get_element_type(), ov::element::boolean);
    ASSERT_EQ(c->get_shape(), (ov::Shape{2, 2}));
}

TEST(type_prop, tensor_constant_deduce_string) {
    auto c =
        ov::op::v0::Constant::create(ov::element::string, ov::Shape{2, 2}, vector<std::string>{"1", "2", "3", "4"});
    ASSERT_EQ(c->get_element_type(), ov::element::string);
    ASSERT_EQ(c->get_shape(), (ov::Shape{2, 2}));
}

TEST(type_prop, tensor_constant_bad_count) {
    try {
        auto c = ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{2, 2}, {1, 1, 1});
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect number of literals not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Did not get the expected number of literals for a "
                                         "constant of shape [2,2] (got 3, expected 1 or 4)"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, constant_zero_elements_one_string) {
    auto c = make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2, 0, 2, 2}, std::vector<std::string>{"42"});
    ASSERT_EQ(c->get_element_type(), ov::element::i64);
    ASSERT_EQ(c->get_shape(), (ov::Shape{2, 0, 2, 2}));
}

TEST(type_prop, constant_zero_elements_ov_string) {
    auto c =
        make_shared<ov::op::v0::Constant>(ov::element::string, ov::Shape{2, 0, 2, 2}, std::vector<std::string>{"42"});
    ASSERT_EQ(c->get_element_type(), ov::element::string);
    ASSERT_EQ(c->get_shape(), (ov::Shape{2, 0, 2, 2}));
}
