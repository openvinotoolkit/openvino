// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset7.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, roll_output_shape_type_test) {
    auto arg = make_shared<opset7::Parameter>(element::f32, Shape{3, 3, 4, 1, 5});
    auto shift = make_shared<opset7::Parameter>(element::i32, Shape{2});
    auto axes = make_shared<opset7::Parameter>(element::i64, Shape{2});

    auto r = make_shared<opset7::Roll>(arg, shift, axes);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape{3, 3, 4, 1, 5}));
}

TEST(type_prop, roll_axis_const_test) {
    auto arg = make_shared<opset7::Parameter>(element::f32, Shape{3, 3, 3});
    auto shift = make_shared<opset7::Parameter>(element::i32, Shape{3});
    auto axes = opset7::Constant::create(element::i64, Shape{3}, {0, 1, -1});

    auto r = make_shared<opset7::Roll>(arg, shift, axes);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape{3, 3, 3}));
}

TEST(type_prop, roll_incorrect_axis_test) {
    auto arg = make_shared<opset7::Parameter>(element::f32, Shape{3, 3});
    auto shift = make_shared<opset7::Parameter>(element::i32, Shape{2});
    auto axes = opset7::Constant::create(element::i64, Shape{2}, {0, 2});

    try {
        auto r = make_shared<opset7::Roll>(arg, shift, axes);
        // Should have thrown, so fail if it didn't
        FAIL() << "Unexpected pass with invalid axes and shift.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Axes must be less than data tensor rank."));
    } catch (...) {
        FAIL() << "Check failed for unexpected reason";
    }
}

TEST(type_prop, roll_incorrect_negative_axis_test) {
    auto arg = make_shared<opset7::Parameter>(element::f32, Shape{3, 3});
    auto shift = make_shared<opset7::Parameter>(element::i32, Shape{2});
    auto axes = opset7::Constant::create(element::i64, Shape{2}, {0, -5});

    try {
        auto r = make_shared<opset7::Roll>(arg, shift, axes);
        // Should have thrown, so fail if it didn't
        FAIL() << "Unexpected pass with invalid axes and shift.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Axes must be positive or equal to zero."));
    } catch (...) {
        FAIL() << "Check failed for unexpected reason";
    }
}

TEST(type_prop, roll_axis_scalar_test) {
    auto arg = make_shared<opset7::Parameter>(element::i32, Shape{3, 3, 4});
    auto shift = opset7::Constant::create(element::i64, Shape{}, {5});
    auto axes = make_shared<opset7::Parameter>(element::i32, Shape{3});

    auto r = make_shared<opset7::Roll>(arg, shift, axes);

    EXPECT_EQ(r->get_output_element_type(0), element::i32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape{3, 3, 4}));
}

TEST(type_prop, roll_invalid_axes_check) {
    auto arg = make_shared<opset7::Parameter>(element::f32, Shape{3, 3, 4, 1, 5});
    auto shift = make_shared<opset7::Parameter>(element::i32, Shape{3});
    auto axes = make_shared<opset7::Parameter>(element::i64, Shape{1});

    try {
        auto r = make_shared<opset7::Roll>(arg, shift, axes);
        // Should have thrown, so fail if it didn't
        FAIL() << "Unexpected pass with invalid axes and shift.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("If shift is a 1D vector, axes must be a 1D tensor of the same size."));
    } catch (...) {
        FAIL() << "Check failed for unexpected reason";
    }
}

TEST(type_prop, roll_dynamic_shape) {
    auto arg = make_shared<opset7::Parameter>(element::f32, PartialShape{Dimension::dynamic(), Dimension::dynamic()});
    auto shift = make_shared<opset7::Parameter>(element::i64, PartialShape{Dimension::dynamic()});
    auto axes = make_shared<opset7::Parameter>(element::i32, PartialShape{Dimension::dynamic()});

    auto r = make_shared<opset7::Roll>(arg, shift, axes);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(2)));
}

TEST(type_prop, roll_dynamic_ranks) {
    auto arg = make_shared<opset7::Parameter>(element::f32, PartialShape::dynamic());
    auto shift = make_shared<opset7::Parameter>(element::i64, PartialShape::dynamic());
    auto axes = make_shared<opset7::Parameter>(element::i32, PartialShape::dynamic());

    auto r = make_shared<opset7::Roll>(arg, shift, axes);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, roll_dynamic_axes_static_shift) {
    auto arg = make_shared<opset7::Parameter>(element::i32, Shape{3, 3, 4, 2});
    auto shift = opset7::Constant::create(element::i64, Shape{}, {5});
    auto axes = make_shared<opset7::Parameter>(element::i32, PartialShape{Dimension::dynamic()});

    auto r = make_shared<opset7::Roll>(arg, shift, axes);

    EXPECT_EQ(r->get_output_element_type(0), element::i32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(Shape{3, 3, 4, 2}));
}

TEST(type_prop, roll_scatic_axes_dynamic_shift) {
    auto arg = make_shared<opset7::Parameter>(element::i32, Shape{1, 2, 4});
    auto shift = make_shared<opset7::Parameter>(element::i64, PartialShape{Dimension::dynamic()});
    auto axes = make_shared<opset7::Parameter>(element::i32, Shape{3});

    auto r = make_shared<opset7::Roll>(arg, shift, axes);

    EXPECT_EQ(r->get_output_element_type(0), element::i32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(Shape{1, 2, 4}));
}

TEST(type_prop, roll_scatic_axes_dynamic_data) {
    auto arg = make_shared<opset7::Parameter>(element::f32, PartialShape{Dimension::dynamic(), Dimension::dynamic()});
    auto shift = opset7::Constant::create(element::i64, Shape{}, {5});
    auto axes = make_shared<opset7::Parameter>(element::i32, PartialShape{Dimension::dynamic()});

    auto r = make_shared<opset7::Roll>(arg, shift, axes);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(2)));
}
