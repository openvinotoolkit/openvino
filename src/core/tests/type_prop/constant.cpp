// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, scalar_constant_deduce_float32) {
    auto c = op::Constant::create(element::f32, Shape{}, {208});
    ASSERT_EQ(c->get_element_type(), element::f32);
    ASSERT_EQ(c->get_shape(), (Shape{}));
}

TEST(type_prop, scalar_constant_deduce_bool) {
    auto c = op::Constant::create(element::boolean, Shape{}, {1});
    ASSERT_EQ(c->get_element_type(), element::boolean);
    ASSERT_EQ(c->get_shape(), (Shape{}));
}

TEST(type_prop, tensor_constant_deduce_float32) {
    auto c = op::Constant::create(element::f32, Shape{2, 2}, {208, 208, 208, 208});
    ASSERT_EQ(c->get_element_type(), element::f32);
    ASSERT_EQ(c->get_shape(), (Shape{2, 2}));
}

TEST(type_prop, tensor_constant_deduce_bool) {
    auto c = op::Constant::create(element::boolean, Shape{2, 2}, {1, 1, 1, 1});
    ASSERT_EQ(c->get_element_type(), element::boolean);
    ASSERT_EQ(c->get_shape(), (Shape{2, 2}));
}

TEST(type_prop, tensor_constant_bad_count) {
    try {
        auto c = op::Constant::create(element::boolean, Shape{2, 2}, {1, 1, 1});
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect number of literals not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Did not get the expected number of literals for a "
                                         "constant of shape {2, 2} (got 3, expected 1 or 4)"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, constant_zero_elements_one_string) {
    auto c = make_shared<op::Constant>(element::i64, Shape{2, 0, 2, 2}, std::vector<std::string>{"42"});
    ASSERT_EQ(c->get_element_type(), element::i64);
    ASSERT_EQ(c->get_shape(), (Shape{2, 0, 2, 2}));
}
