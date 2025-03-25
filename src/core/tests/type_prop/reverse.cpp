// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reverse.hpp"

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/opsets/opset10.hpp"

using namespace std;
using namespace ov;
using namespace testing;

class TypePropReverseV1Test : public TypePropOpTest<op::v1::Reverse> {};

TEST(type_prop, reverse_1d_deduce) {
    // Deduce type
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5});
    auto rev = make_shared<op::v1::Reverse>(param,
                                            ov::op::v0::Constant::create(element::i64, {1}, {0}),
                                            op::v1::Reverse::Mode::INDEX);

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5}));
}

TEST(type_prop, reverse_2d_deduce_0) {
    // Deduce type
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 6});
    auto rev = make_shared<op::v1::Reverse>(param,
                                            ov::op::v0::Constant::create(element::i64, {1}, {0}),
                                            op::v1::Reverse::Mode::INDEX);

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5, 6}));
}

TEST(type_prop, reverse_2d_deduce_1) {
    // Deduce type
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 6});
    auto rev = make_shared<op::v1::Reverse>(param,
                                            ov::op::v0::Constant::create(element::i64, {1}, {1}),
                                            op::v1::Reverse::Mode::INDEX);

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5, 6}));
}

TEST(type_prop, reverse_2d_deduce_01) {
    // Deduce type
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 6});
    auto rev = make_shared<op::v1::Reverse>(param,
                                            ov::op::v0::Constant::create(element::i64, {2}, {0, 1}),
                                            op::v1::Reverse::Mode::INDEX);

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5, 6}));
}

TEST(type_prop, reverse_3d_deduce_0) {
    // Deduce type
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 6, 7});
    auto rev = make_shared<op::v1::Reverse>(param,
                                            ov::op::v0::Constant::create(element::i64, {1}, {0}),
                                            op::v1::Reverse::Mode::INDEX);

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5, 6, 7}));
}

TEST(type_prop, reverse_3d_deduce_1) {
    // Deduce type
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 6, 7});
    auto rev = make_shared<op::v1::Reverse>(param,
                                            ov::op::v0::Constant::create(element::i64, {1}, {1}),
                                            op::v1::Reverse::Mode::INDEX);

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5, 6, 7}));
}

TEST(type_prop, reverse_3d_deduce_2) {
    // Deduce type
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 6, 7});
    auto rev = make_shared<op::v1::Reverse>(param,
                                            ov::op::v0::Constant::create(element::i64, {1}, {2}),
                                            op::v1::Reverse::Mode::INDEX);

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5, 6, 7}));
}

TEST(type_prop, reverse_3d_deduce_01) {
    // Deduce type
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 6, 7});
    auto rev = make_shared<op::v1::Reverse>(param,
                                            ov::op::v0::Constant::create(element::i64, {2}, {0, 1}),
                                            op::v1::Reverse::Mode::INDEX);

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5, 6, 7}));
}

TEST(type_prop, reverse_3d_deduce_02) {
    // Deduce type
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 6, 7});
    auto rev = make_shared<op::v1::Reverse>(param,
                                            ov::op::v0::Constant::create(element::i64, {2}, {0, 2}),
                                            op::v1::Reverse::Mode::INDEX);

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5, 6, 7}));
}

TEST(type_prop, reverse_3d_deduce_12) {
    // Deduce type
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 6, 7});
    auto rev = make_shared<op::v1::Reverse>(param,
                                            ov::op::v0::Constant::create(element::i64, {2}, {1, 2}),
                                            op::v1::Reverse::Mode::INDEX);

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5, 6, 7}));
}

TEST(type_prop, reverse_3d_deduce_012) {
    // Deduce type
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 6, 7});
    auto rev = make_shared<op::v1::Reverse>(param,
                                            ov::op::v0::Constant::create(element::i64, {3}, {0, 1, 2}),
                                            op::v1::Reverse::Mode::INDEX);

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5, 6, 7}));
}

TEST(type_prop, reverse_3d_deduce_oob) {
    // Deduce type
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 6, 7});
    try {
        auto rev = make_shared<op::v1::Reverse>(param,
                                                ov::op::v0::Constant::create(element::i64, {3}, {0, 3, 2}),
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
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto rev = make_shared<op::v1::Reverse>(param,
                                            ov::op::v0::Constant::create(element::i64, {4}, {0, 2, 1776, 90909}),
                                            op::v1::Reverse::Mode::INDEX);

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_output_partial_shape(0), PartialShape::dynamic());
}

using namespace ov::opset10;

//
// If the input rank is static but the shape is dynamic, we should pass if the axis indices are
// in bounds.
//
TEST_F(TypePropReverseV1Test, partial_rank_static_dynamic_axes_ok) {
    PartialShape param_shape{Dimension::dynamic(), {10, 300}, 2, 3};
    auto symbols = set_shape_symbols(param_shape);
    auto param = make_shared<Parameter>(element::f32, param_shape);
    auto rev = make_op(param, Constant::create(element::i64, {2}, {0, 2}), op::v1::Reverse::Mode::INDEX);

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_output_partial_shape(0), param_shape);
    EXPECT_THAT(get_shape_symbols(rev->get_output_partial_shape(0)), symbols);
}

TEST_F(TypePropReverseV1Test, axes_index_is_not_1d_tensor) {
    PartialShape param_shape{Dimension::dynamic(), Dimension::dynamic(), 2, 3};
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, param_shape);
    auto axes = make_shared<Parameter>(element::i64, PartialShape{2, 3});

    OV_EXPECT_THROW(auto op = make_op(param, axes, op::v1::Reverse::Mode::INDEX),
                    NodeValidationFailure,
                    HasSubstr("The reversed_axes input must be a 1D tensor"));
}

TEST_F(TypePropReverseV1Test, axes_mask_is_not_1d_tensor) {
    PartialShape param_shape{Dimension::dynamic(), Dimension::dynamic(), 2, 3};
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, param_shape);
    auto axes = make_shared<Parameter>(element::boolean, PartialShape{2, 3});

    OV_EXPECT_THROW(auto op = make_op(param, axes, op::v1::Reverse::Mode::MASK),
                    NodeValidationFailure,
                    HasSubstr("The reversed_axes input must be a 1D tensor"));
}

TEST_F(TypePropReverseV1Test, axes_mask_length_lt_input_rank) {
    PartialShape param_shape{Dimension::dynamic(), Dimension::dynamic(), 2, 3};
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, param_shape);
    auto axes = make_shared<Parameter>(element::boolean, PartialShape{2});

    OV_EXPECT_THROW(
        auto op = make_op(param, axes, op::v1::Reverse::Mode::MASK),
        NodeValidationFailure,
        HasSubstr("The number of elements in the reversed_axes tensor (2) must match the input data tensor rank"));
}

TEST_F(TypePropReverseV1Test, axes_mask_length_gt_input_rank) {
    PartialShape param_shape{Dimension::dynamic(), Dimension::dynamic(), 2, 3};
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, param_shape);
    auto axes = make_shared<Parameter>(element::boolean, PartialShape{5});

    OV_EXPECT_THROW(
        auto op = make_op(param, axes, op::v1::Reverse::Mode::MASK),
        NodeValidationFailure,
        HasSubstr("The number of elements in the reversed_axes tensor (5) must match the input data tensor rank"));
}

TEST_F(TypePropReverseV1Test, axes_index_is_scalar) {
    PartialShape param_shape{2, {2, 10}, 8};
    auto param = make_shared<Parameter>(element::f32, param_shape);
    auto axes = make_shared<Parameter>(element::i64, PartialShape{});

    OV_EXPECT_THROW(auto op = make_op(param, axes, op::v1::Reverse::Mode::INDEX),
                    NodeValidationFailure,
                    HasSubstr("The reversed_axes input must be a 1D tensor"));
}

TEST_F(TypePropReverseV1Test, axes_mask_is_scalar) {
    PartialShape param_shape{2, {2, 10}, 8};
    auto param = make_shared<Parameter>(element::f32, param_shape);
    auto axes = make_shared<Parameter>(element::boolean, PartialShape{});

    OV_EXPECT_THROW(auto op = make_op(param, axes, op::v1::Reverse::Mode::MASK),
                    NodeValidationFailure,
                    HasSubstr("The reversed_axes input must be a 1D tensor"));
}

TEST_F(TypePropReverseV1Test, axes_mask_not_boolean_type) {
    PartialShape param_shape{2, {2, 10}, 8};
    auto param = make_shared<Parameter>(element::f32, param_shape);
    auto axes = make_shared<Parameter>(element::i32, PartialShape{4});

    OV_EXPECT_THROW(auto op = make_op(param, axes, op::v1::Reverse::Mode::MASK),
                    NodeValidationFailure,
                    HasSubstr("In 'mask' mode the second input must contain boolean values"));
}

TEST_F(TypePropReverseV1Test, axes_index_not_integer_type) {
    PartialShape param_shape{2, {2, 10}, 8};
    auto param = make_shared<Parameter>(element::f32, param_shape);
    auto axes = make_shared<Parameter>(element::f32, PartialShape{4});

    OV_EXPECT_THROW(auto op = make_op(param, axes, op::v1::Reverse::Mode::INDEX),
                    NodeValidationFailure,
                    HasSubstr("In 'index' mode the second input must contain integer values"));
}

TEST_F(TypePropReverseV1Test, param_static_rank_partial_shape_axes_out_of_input_rank) {
    PartialShape param_shape{Dimension::dynamic(), Dimension::dynamic(), 2, 3};
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, param_shape);

    OV_EXPECT_THROW(
        auto op = make_op(param, Constant::create(element::i64, {3}, {0, 4, 2}), op::v1::Reverse::Mode::INDEX),
        NodeValidationFailure,
        HasSubstr("Some of the provided axes (AxisSet{0, 2, 4}) are out of bounds (input rank: 4)."));
}

TEST_F(TypePropReverseV1Test, param_static_rank_partial_shape_axes_negatives) {
    PartialShape param_shape{-1, {2, -1}, {-1, 3}, 5};
    auto symbols = set_shape_symbols(param_shape);
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, param_shape);

    auto op = make_op(param, Constant::create(element::i64, {3}, {0, -1, 2}), op::v1::Reverse::Mode::INDEX);

    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), param_shape);
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), symbols);
}

TEST_F(TypePropReverseV1Test, more_axes_index_than_input_rank) {
    PartialShape param_shape{-1, {2, -1}, {-1, 3}, 5};
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, param_shape);

    auto op = make_op(param, Constant::create(element::i64, {7}, {0, -1, 1, 2, 3, 3, 2}), op::v1::Reverse::Mode::INDEX);

    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), param_shape);
}

TEST_F(TypePropReverseV1Test, axes_index_is_dynamic) {
    PartialShape param_shape{2, {2, 10}, 8};
    auto param = make_shared<Parameter>(element::f32, param_shape);
    auto axes = make_shared<Parameter>(element::i64, PartialShape::dynamic());

    auto op = make_op(param, axes, op::v1::Reverse::Mode::INDEX);

    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), param_shape);
}

TEST_F(TypePropReverseV1Test, axes_index_interval_1d_tensor) {
    PartialShape param_shape{2, {2, 10}, 8};
    auto param = make_shared<Parameter>(element::f32, param_shape);
    auto axes = make_shared<Parameter>(element::i64, PartialShape{{2, 4}});

    auto op = make_op(param, axes, op::v1::Reverse::Mode::INDEX);

    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), param_shape);
}

TEST_F(TypePropReverseV1Test, default_ctor) {
    PartialShape param_shape{2, {2, 10}, 8};
    auto param = make_shared<Parameter>(element::f32, param_shape);
    auto axes = Constant::create(element::i64, Shape{3}, {2, 0, 1});

    auto op = make_op();
    op->set_arguments(OutputVector{param, axes});
    op->set_mode(op::v1::Reverse::Mode::INDEX);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_mode(), op::v1::Reverse::Mode::INDEX);
    EXPECT_EQ(op->get_input_size(), 2);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), param_shape);
}
