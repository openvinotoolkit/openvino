// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

TEST(type_prop, reverse_1d_deduce) {
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5});
    auto rev =
        make_shared<op::v1::Reverse>(param, op::Constant::create(element::i64, {1}, {0}), op::v1::Reverse::Mode::INDEX);

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5}));
}

TEST(type_prop, reverse_2d_deduce_0) {
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6});
    auto rev =
        make_shared<op::v1::Reverse>(param, op::Constant::create(element::i64, {1}, {0}), op::v1::Reverse::Mode::INDEX);

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5, 6}));
}

TEST(type_prop, reverse_2d_deduce_1) {
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6});
    auto rev =
        make_shared<op::v1::Reverse>(param, op::Constant::create(element::i64, {1}, {1}), op::v1::Reverse::Mode::INDEX);

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5, 6}));
}

TEST(type_prop, reverse_2d_deduce_01) {
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6});
    auto rev = make_shared<op::v1::Reverse>(param,
                                            op::Constant::create(element::i64, {2}, {0, 1}),
                                            op::v1::Reverse::Mode::INDEX);

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5, 6}));
}

TEST(type_prop, reverse_3d_deduce_0) {
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6, 7});
    auto rev =
        make_shared<op::v1::Reverse>(param, op::Constant::create(element::i64, {1}, {0}), op::v1::Reverse::Mode::INDEX);

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5, 6, 7}));
}

TEST(type_prop, reverse_3d_deduce_1) {
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6, 7});
    auto rev =
        make_shared<op::v1::Reverse>(param, op::Constant::create(element::i64, {1}, {1}), op::v1::Reverse::Mode::INDEX);

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5, 6, 7}));
}

TEST(type_prop, reverse_3d_deduce_2) {
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6, 7});
    auto rev =
        make_shared<op::v1::Reverse>(param, op::Constant::create(element::i64, {1}, {2}), op::v1::Reverse::Mode::INDEX);

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5, 6, 7}));
}

TEST(type_prop, reverse_3d_deduce_01) {
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6, 7});
    auto rev = make_shared<op::v1::Reverse>(param,
                                            op::Constant::create(element::i64, {2}, {0, 1}),
                                            op::v1::Reverse::Mode::INDEX);

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5, 6, 7}));
}

TEST(type_prop, reverse_3d_deduce_02) {
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6, 7});
    auto rev = make_shared<op::v1::Reverse>(param,
                                            op::Constant::create(element::i64, {2}, {0, 2}),
                                            op::v1::Reverse::Mode::INDEX);

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5, 6, 7}));
}

TEST(type_prop, reverse_3d_deduce_12) {
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6, 7});
    auto rev = make_shared<op::v1::Reverse>(param,
                                            op::Constant::create(element::i64, {2}, {1, 2}),
                                            op::v1::Reverse::Mode::INDEX);

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5, 6, 7}));
}

TEST(type_prop, reverse_3d_deduce_012) {
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6, 7});
    auto rev = make_shared<op::v1::Reverse>(param,
                                            op::Constant::create(element::i64, {3}, {0, 1, 2}),
                                            op::v1::Reverse::Mode::INDEX);

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5, 6, 7}));
}

TEST(type_prop, reverse_3d_deduce_oob) {
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6, 7});
    try {
        auto rev = make_shared<op::v1::Reverse>(param,
                                                op::Constant::create(element::i64, {3}, {0, 3, 2}),
                                                op::v1::Reverse::Mode::INDEX);

        // Should have thrown, so fail if it didn't
        FAIL() << "Axis out of bounds not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Some of the provided axes (AxisSet{0, 2, 3}) are out of bounds (input rank: 3)."));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

//
// If the input rank is dynamic, we should pass unconditionally.
//
TEST(type_prop, reverse_partial_rank_dynamic) {
    auto param = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto rev = make_shared<op::v1::Reverse>(param,
                                            op::Constant::create(element::i64, {4}, {0, 2, 1776, 90909}),
                                            op::v1::Reverse::Mode::INDEX);

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_TRUE(rev->get_output_partial_shape(0).rank().is_dynamic());
}

//
// If the input rank is static but the shape is dynamic, we should pass if the axis indices are
// in bounds.
//
TEST(type_prop, reverse_partial_rank_static_dynamic_axes_ok) {
    PartialShape param_shape{Dimension::dynamic(), Dimension::dynamic(), 2, 3};
    auto param = make_shared<op::Parameter>(element::f32, param_shape);
    auto rev = make_shared<op::v1::Reverse>(param,
                                            op::Constant::create(element::i64, {2}, {0, 2}),
                                            op::v1::Reverse::Mode::INDEX);

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_TRUE(rev->get_output_partial_shape(0).same_scheme(param_shape));
}

TEST(type_prop, reverse_partial_rank_static_dynamic_axes_oob) {
    PartialShape param_shape{Dimension::dynamic(), Dimension::dynamic(), 2, 3};
    auto param = make_shared<op::Parameter>(element::f32, param_shape);
    try {
        auto rev = make_shared<op::v1::Reverse>(param,
                                                op::Constant::create(element::i64, {3}, {0, 4, 2}),
                                                op::v1::Reverse::Mode::INDEX);

        // Should have thrown, so fail if it didn't
        FAIL() << "Axis out of bounds not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Some of the provided axes (AxisSet{0, 2, 4}) are out of bounds (input rank: 4)."));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}
