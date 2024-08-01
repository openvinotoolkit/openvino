// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/type_prop.hpp"
#include "gmock/gmock.h"
#include "openvino/opsets/opset7.hpp"

using namespace std;
using namespace ov;
using namespace ov::opset7;
using namespace testing;

class TypePropRollV7Test : public TypePropOpTest<op::v7::Roll> {};

TEST(type_prop, roll_output_shape_type_test) {
    auto arg_shape = PartialShape{3, 3, 4, 1, 5};
    auto symbols = set_shape_symbols(arg_shape);
    auto arg = make_shared<opset7::Parameter>(element::f32, arg_shape);
    auto shift = make_shared<opset7::Parameter>(element::i32, Shape{2});
    auto axes = make_shared<opset7::Parameter>(element::i64, Shape{2});

    auto r = make_shared<opset7::Roll>(arg, shift, axes);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_EQ(r->get_output_partial_shape(0), PartialShape({3, 3, 4, 1, 5}));
    EXPECT_THAT(get_shape_symbols(r->get_output_partial_shape(0)), symbols);
}

TEST(type_prop, roll_axis_const_test) {
    auto arg = make_shared<opset7::Parameter>(element::f32, Shape{3, 3, 3});
    auto shift = make_shared<opset7::Parameter>(element::i32, Shape{3});
    auto axes = opset7::Constant::create(element::i64, Shape{3}, {0, 1, -1});

    auto r = make_shared<opset7::Roll>(arg, shift, axes);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_EQ(r->get_output_partial_shape(0), PartialShape({3, 3, 3}));
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
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Axis 2 out of the tensor rank range"));
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
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Axis -5 out of the tensor rank range"));
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
    EXPECT_EQ(r->get_output_partial_shape(0), PartialShape({3, 3, 4}));
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
    EXPECT_EQ(r->get_output_partial_shape(0), PartialShape::dynamic(2));
}

TEST(type_prop, roll_dynamic_ranks) {
    auto arg = make_shared<opset7::Parameter>(element::f32, PartialShape::dynamic());
    auto shift = make_shared<opset7::Parameter>(element::i64, PartialShape::dynamic());
    auto axes = make_shared<opset7::Parameter>(element::i32, PartialShape::dynamic());

    auto r = make_shared<opset7::Roll>(arg, shift, axes);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_EQ(r->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST(type_prop, roll_dynamic_axes_static_shift) {
    auto arg = make_shared<opset7::Parameter>(element::i32, Shape{3, 3, 4, 2});
    auto shift = opset7::Constant::create(element::i64, Shape{}, {5});
    auto axes = make_shared<opset7::Parameter>(element::i32, PartialShape{Dimension::dynamic()});

    auto r = make_shared<opset7::Roll>(arg, shift, axes);

    EXPECT_EQ(r->get_output_element_type(0), element::i32);
    EXPECT_EQ(r->get_output_partial_shape(0), PartialShape({3, 3, 4, 2}));
}

TEST(type_prop, roll_static_axes_dynamic_shift) {
    auto arg = make_shared<opset7::Parameter>(element::i32, Shape{1, 2, 4});
    auto shift = make_shared<opset7::Parameter>(element::i64, PartialShape{Dimension::dynamic()});
    auto axes = make_shared<opset7::Parameter>(element::i32, Shape{3});

    auto r = make_shared<opset7::Roll>(arg, shift, axes);

    EXPECT_EQ(r->get_output_element_type(0), element::i32);
    EXPECT_EQ(r->get_output_shape(0), Shape({1, 2, 4}));
}

TEST_F(TypePropRollV7Test, static_axes_dynamic_data) {
    auto arg_shape = PartialShape{-1, -1};
    auto symbols = set_shape_symbols(arg_shape);
    const auto arg = make_shared<Parameter>(element::f32, arg_shape);
    const auto shift = Constant::create(element::i64, Shape{}, {5});
    const auto axes = make_shared<Parameter>(element::i32, PartialShape{Dimension::dynamic()});

    const auto op = make_op(arg, shift, axes);

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape::dynamic(2));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), symbols);
}

TEST_F(TypePropRollV7Test, const_shift_axes_and_interval_dim_on_arg_shape) {
    auto arg_shape = PartialShape{{2, 5}, {-1, 10}, {4, -1}, -1};
    auto symbols = set_shape_symbols(arg_shape);
    const auto arg = make_shared<Parameter>(element::f32, arg_shape);
    const auto shift = Constant::create(element::i64, Shape{}, {5});
    const auto axes = Constant::create(element::i64, Shape{2}, {0, 1});

    const auto op = make_op(arg, shift, axes);

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), arg_shape);
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), symbols);
}

TEST_F(TypePropRollV7Test, default_ctor) {
    const auto arg_shape = PartialShape{{3, 5}, -1, 10};
    const auto arg = make_shared<Parameter>(element::f32, arg_shape);
    const auto shift = Constant::create(element::i64, Shape{}, {5});
    const auto axes = Constant::create(element::i64, Shape{2}, {0, 1});

    const auto op = make_op();
    op->set_arguments(OutputVector{arg, shift, axes});
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_input_size(), 3);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), arg_shape);
}
