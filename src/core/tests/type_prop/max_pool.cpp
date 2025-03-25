// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/max_pool.hpp"

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "dimension_util.hpp"
#include "openvino/op/parameter.hpp"

using namespace std;
using namespace ov;
using namespace testing;

template <class TOp>
class MaxPoolOperator : public TypePropOpTest<TOp> {};

TYPED_TEST_SUITE_P(MaxPoolOperator);

TEST(type_prop, max_pool_default_ctor) {
    PartialShape arg_shape{1, 3, 32};
    auto symbols = set_shape_symbols(arg_shape);
    const Strides strides{1};
    const Shape pads_begin{2};
    const Shape pads_end{2};
    const Shape kernel_shape{2};

    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, arg_shape);

    auto mp = make_shared<op::v1::MaxPool>();
    mp->set_argument(0, arg);
    mp->set_pads_begin(pads_begin);
    mp->set_pads_end(pads_end);
    mp->set_kernel(kernel_shape);
    mp->set_strides(strides);
    mp->set_rounding_type(op::RoundingType::CEIL);
    mp->set_auto_pad(op::PadType::VALID);
    mp->validate_and_infer_types();

    EXPECT_EQ(mp->get_output_partial_shape(0), PartialShape({1, 3, 31}));
    EXPECT_THAT(get_shape_symbols(mp->get_output_partial_shape(0)), ElementsAre(symbols[0], symbols[1], nullptr));
    EXPECT_EQ(mp->get_pads_begin(), (Shape{0}));
    EXPECT_EQ(mp->get_pads_end(), (Shape{0}));
}

TEST(type_prop, max_pool_valid_auto_padding) {
    PartialShape arg_shape{1, 3, {10, 32}};
    auto symbols = set_shape_symbols(arg_shape);
    const Strides strides{1};
    const Shape pads_begin{2};
    const Shape pads_end{2};
    const Shape kernel_shape{2};
    const auto rounding_mode = op::RoundingType::FLOOR;
    const auto auto_pad = op::PadType::VALID;

    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, arg_shape);
    auto mp = make_shared<op::v1::MaxPool>(arg, strides, pads_begin, pads_end, kernel_shape, rounding_mode, auto_pad);
    EXPECT_EQ(mp->get_output_partial_shape(0), PartialShape({1, 3, {9, 31}}));
    EXPECT_THAT(get_shape_symbols(mp->get_output_partial_shape(0)), ElementsAre(symbols[0], symbols[1], nullptr));
    EXPECT_EQ(mp->get_pads_begin(), (Shape{0}));
    EXPECT_EQ(mp->get_pads_end(), (Shape{0}));
}

TEST(type_prop, max_pool_1D_auto_padding) {
    const PartialShape arg_shape{1, 3, 32};
    const Strides strides{1};
    const Shape pads_begin{0};
    const Shape pads_end{0};
    const Shape kernel_shape{2};
    const auto rounding_mode = op::RoundingType::FLOOR;
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, arg_shape);
    auto mp = make_shared<op::v1::MaxPool>(arg, strides, pads_begin, pads_end, kernel_shape, rounding_mode, auto_pad);

    EXPECT_EQ(mp->get_output_partial_shape(0), PartialShape({1, 3, 32}));
    EXPECT_EQ(mp->get_pads_begin(), (Shape{1}));
    EXPECT_EQ(mp->get_pads_end(), (Shape{0}));
}

TEST(type_prop, max_pool_2D_auto_padding) {
    const PartialShape arg_shape{1, 3, 32, 32};
    const Strides strides{1, 1};
    const Shape pads_begin{0, 0};
    const Shape pads_end{0, 0};
    const Shape kernel_shape{2, 2};
    const auto rounding_mode = op::RoundingType::FLOOR;
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, arg_shape);
    auto mp = make_shared<op::v1::MaxPool>(arg, strides, pads_begin, pads_end, kernel_shape, rounding_mode, auto_pad);

    EXPECT_EQ(mp->get_output_partial_shape(0), PartialShape({1, 3, 32, 32}));
    EXPECT_EQ(mp->get_pads_begin(), (Shape{1, 1}));
    EXPECT_EQ(mp->get_pads_end(), (Shape{0, 0}));
}

TEST(type_prop, max_pool_auto_padding_1D_nc_dims_dynamic_same_lower) {
    const PartialShape arg_shape{Dimension::dynamic(), 32, 32};
    const Strides strides{1};
    const Shape pads_begin{0};
    const Shape pads_end{0};
    const Shape kernel_shape{2};
    const auto rounding_mode = op::RoundingType::FLOOR;
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, arg_shape);
    auto mp = make_shared<op::v1::MaxPool>(arg, strides, pads_begin, pads_end, kernel_shape, rounding_mode, auto_pad);

    EXPECT_EQ(mp->get_output_partial_shape(0), PartialShape({Dimension::dynamic(), 32, 32}));
    EXPECT_EQ(mp->get_pads_begin(), (Shape{1}));
    EXPECT_EQ(mp->get_pads_end(), (Shape{0}));
}

TEST(type_prop, max_pool_auto_padding_2D_nc_dims_dynamic_same_lower) {
    const PartialShape arg_shape{Dimension::dynamic(), Dimension::dynamic(), 32, 32};
    const Strides strides{1, 1};
    const Shape pads_begin{0, 0};
    const Shape pads_end{0, 0};
    const Shape kernel_shape{2, 2};
    const auto rounding_mode = op::RoundingType::FLOOR;
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, arg_shape);
    auto mp = make_shared<op::v1::MaxPool>(arg, strides, pads_begin, pads_end, kernel_shape, rounding_mode, auto_pad);

    EXPECT_EQ(mp->get_output_partial_shape(0), PartialShape({Dimension::dynamic(), Dimension::dynamic(), 32, 32}));
    EXPECT_EQ(mp->get_pads_begin(), (Shape{1, 1}));
    EXPECT_EQ(mp->get_pads_end(), (Shape{0, 0}));
}

TEST(type_prop, max_pool_auto_padding_nc_dims_dynamic_same_upper) {
    PartialShape arg_shape{Dimension::dynamic(), Dimension::dynamic(), 32, 32};
    auto symbols = set_shape_symbols(arg_shape);
    const Strides strides{1, 1};
    const Shape pads_begin{0, 0};
    const Shape pads_end{0, 0};
    const Shape kernel_shape{2, 2};
    const auto rounding_mode = op::RoundingType::FLOOR;
    const auto auto_pad = op::PadType::SAME_UPPER;

    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, arg_shape);
    auto mp = make_shared<op::v1::MaxPool>(arg, strides, pads_begin, pads_end, kernel_shape, rounding_mode, auto_pad);

    EXPECT_EQ(mp->get_output_partial_shape(0), PartialShape({Dimension::dynamic(), Dimension::dynamic(), 32, 32}));
    EXPECT_THAT(get_shape_symbols(mp->get_output_partial_shape(0)),
                ElementsAre(symbols[0], symbols[1], nullptr, nullptr));
    EXPECT_EQ(mp->get_pads_begin(), (Shape{0, 0}));
    EXPECT_EQ(mp->get_pads_end(), (Shape{1, 1}));
}

TEST(type_prop, max_pool_auto_padding_interval_dims_same_upper) {
    PartialShape arg_shape{{1, 2}, {2, 3}, {16, 32}, {11, 32}};
    auto symbols = set_shape_symbols(arg_shape);
    const Strides strides{1, 1};
    const Shape pads_begin{0, 0};
    const Shape pads_end{0, 0};
    const Shape kernel_shape{2, 2};
    const auto rounding_mode = op::RoundingType::FLOOR;
    const auto auto_pad = op::PadType::SAME_UPPER;

    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, arg_shape);
    auto mp = make_shared<op::v1::MaxPool>(arg, strides, pads_begin, pads_end, kernel_shape, rounding_mode, auto_pad);

    EXPECT_EQ(mp->get_output_partial_shape(0), PartialShape({{1, 2}, {2, 3}, -1, -1}));
    EXPECT_THAT(get_shape_symbols(mp->get_output_partial_shape(0)),
                ElementsAre(symbols[0], symbols[1], nullptr, nullptr));
    EXPECT_EQ(mp->get_pads_begin(), (Shape{0, 0}));
    EXPECT_EQ(mp->get_pads_end(), (Shape{0, 0}));
}

TEST(type_prop, max_pool_auto_padding_spatial_dims_dynamic) {
    const PartialShape arg_shape{1, 3, 32, Dimension::dynamic()};
    const Strides strides{1, 1};
    const Shape pads_begin{0, 0};
    const Shape pads_end{0, 0};
    const Shape kernel_shape{2, 2};
    const auto rounding_mode = op::RoundingType::FLOOR;
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, arg_shape);
    auto mp = make_shared<op::v1::MaxPool>(arg, strides, pads_begin, pads_end, kernel_shape, rounding_mode, auto_pad);

    EXPECT_EQ(mp->get_output_partial_shape(0), PartialShape({1, 3, 32, Dimension::dynamic()}));
    EXPECT_EQ(mp->get_pads_begin(), (Shape{1, 0}));
    EXPECT_EQ(mp->get_pads_end(), (Shape{0, 0}));
}

TEST(type_prop, max_pool_default_values) {
    const PartialShape arg_shape{1, 3, 32, 32};
    const Strides strides{1, 1};
    const Shape pads_begin{0, 0};
    const Shape pads_end{0, 0};
    const Shape kernel_shape{2, 2};

    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, arg_shape);
    auto mp = make_shared<op::v1::MaxPool>(arg, strides, pads_begin, pads_end, kernel_shape);

    EXPECT_EQ(mp->get_rounding_type(), op::RoundingType::FLOOR);
    EXPECT_EQ(mp->get_auto_pad(), op::PadType::EXPLICIT);
}

TEST(type_prop, max_pool_v1_invalid_rounding_type) {
    const PartialShape arg_shape{1, 3, 32, Dimension::dynamic()};
    const Strides strides{1, 1};
    const Shape pads_begin{0, 0};
    const Shape pads_end{0, 0};
    const Shape kernel_shape{2, 2};
    const auto rounding_mode = op::RoundingType::CEIL_TORCH;
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, arg_shape);

    EXPECT_THROW(
        std::ignore =
            make_shared<op::v1::MaxPool>(arg, strides, pads_begin, pads_end, kernel_shape, rounding_mode, auto_pad),
        ov::NodeValidationFailure);
}

TEST(type_prop, max_pool_v8_invalid_rounding_type) {
    const PartialShape arg_shape{1, 3, 32, Dimension::dynamic()};
    const Strides strides{1, 1};
    const Strides dilations{1, 1};
    const Shape pads_begin{0, 0};
    const Shape pads_end{0, 0};
    const Shape kernel_shape{2, 2};
    const auto rounding_mode = op::RoundingType::CEIL_TORCH;

    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, arg_shape);

    EXPECT_THROW(
        std::ignore =
            make_shared<op::v8::MaxPool>(arg, strides, dilations, pads_begin, pads_end, kernel_shape, rounding_mode),
        ov::NodeValidationFailure);
}

TYPED_TEST_P(MaxPoolOperator, max_pool_3D_no_dilations) {
    const PartialShape arg_shape{1, 7, 13};
    const Strides strides{1};
    const Strides dilations{1};
    const Shape pads_begin{0};
    const Shape pads_end{0};
    const Shape kernel_shape{3};

    const auto arg = make_shared<ov::op::v0::Parameter>(element::f32, arg_shape);
    const auto mp = this->make_op(arg, strides, dilations, pads_begin, pads_end, kernel_shape);

    const auto expected_output_shape = PartialShape({1, 7, 11});
    EXPECT_EQ(mp->get_output_partial_shape(0), expected_output_shape);
    EXPECT_EQ(mp->get_output_partial_shape(1), expected_output_shape);
}

TYPED_TEST_P(MaxPoolOperator, max_pool_3D_with_dilations) {
    const PartialShape arg_shape{1, 7, 13};
    const Strides strides{1};
    const Strides dilations{2};
    const Shape pads_begin{0};
    const Shape pads_end{0};
    const Shape kernel_shape{3};

    const auto arg = make_shared<ov::op::v0::Parameter>(element::f32, arg_shape);
    const auto mp = this->make_op(arg, strides, dilations, pads_begin, pads_end, kernel_shape);

    const auto expected_output_shape = PartialShape({1, 7, 9});
    EXPECT_EQ(mp->get_output_partial_shape(0), expected_output_shape);
    EXPECT_EQ(mp->get_output_partial_shape(1), expected_output_shape);
}

TYPED_TEST_P(MaxPoolOperator, max_pool_3D_with_dilations_and_padding) {
    const PartialShape arg_shape{1, 7, 13};
    const Strides strides{1};
    const Strides dilations{2};
    const Shape pads_begin{1};
    const Shape pads_end{2};
    const Shape kernel_shape{3};

    const auto arg = make_shared<ov::op::v0::Parameter>(element::f32, arg_shape);
    const auto mp = this->make_op(arg, strides, dilations, pads_begin, pads_end, kernel_shape);

    const auto expected_output_shape = PartialShape({1, 7, 12});
    EXPECT_EQ(mp->get_output_partial_shape(0), expected_output_shape);
    EXPECT_EQ(mp->get_output_partial_shape(1), expected_output_shape);
}

TYPED_TEST_P(MaxPoolOperator, max_pool_4D_no_dilations) {
    const PartialShape arg_shape{1, 3, 13, 13};
    const Strides strides{1, 1};
    const Strides dilations{1, 1};
    const Shape pads_begin{0, 0};
    const Shape pads_end{0, 0};
    const Shape kernel_shape{2, 2};

    const auto arg = make_shared<ov::op::v0::Parameter>(element::f32, arg_shape);
    const auto mp = this->make_op(arg, strides, dilations, pads_begin, pads_end, kernel_shape);

    const auto expected_output_shape = PartialShape({1, 3, 12, 12});
    EXPECT_EQ(mp->get_output_partial_shape(0), expected_output_shape);
    EXPECT_EQ(mp->get_output_partial_shape(1), expected_output_shape);
}

TYPED_TEST_P(MaxPoolOperator, max_pool_4D_with_dilations) {
    const PartialShape arg_shape{1, 3, 13, 13};
    const Strides strides{1, 1};
    const Strides dilations{2, 3};
    const Shape pads_begin{0, 0};
    const Shape pads_end{0, 0};
    const Shape kernel_shape{2, 2};

    const auto arg = make_shared<ov::op::v0::Parameter>(element::f32, arg_shape);
    const auto mp = this->make_op(arg, strides, dilations, pads_begin, pads_end, kernel_shape);

    const auto expected_output_shape = PartialShape({1, 3, 11, 10});
    EXPECT_EQ(mp->get_output_partial_shape(0), expected_output_shape);
    EXPECT_EQ(mp->get_output_partial_shape(1), expected_output_shape);
}

TYPED_TEST_P(MaxPoolOperator, max_pool_4D_dynamic_dims_with_non_zero_low_range_floor_mode) {
    PartialShape arg_shape{Dimension::dynamic(), 64, {198, ov::util::dim::inf_bound}, {198, ov::util::dim::inf_bound}};
    const Strides strides{2, 2};
    const Strides dilations{1, 1};
    const Shape pads_begin{0, 0};
    const Shape pads_end{0, 0};
    const Shape kernel_shape{2, 2};
    const auto rounding_mode = op::RoundingType::FLOOR;

    const auto arg = make_shared<ov::op::v0::Parameter>(element::f32, arg_shape);
    const auto mp = this->make_op(arg, strides, dilations, pads_begin, pads_end, kernel_shape, rounding_mode);

    const auto expected_output_shape =
        PartialShape{Dimension::dynamic(), 64, {99, ov::util::dim::inf_bound}, {99, ov::util::dim::inf_bound}};
    EXPECT_EQ(mp->get_output_partial_shape(0), expected_output_shape);
    EXPECT_EQ(mp->get_output_partial_shape(1), expected_output_shape);
}

TYPED_TEST_P(MaxPoolOperator, max_pool_4D_dynamic_dims_with_non_zero_low_range_ceil_mode) {
    PartialShape arg_shape{Dimension::dynamic(), 64, {198, ov::util::dim::inf_bound}, {198, ov::util::dim::inf_bound}};
    const Strides strides{2, 2};
    const Strides dilations{1, 1};
    const Shape pads_begin{0, 0};
    const Shape pads_end{0, 0};
    const Shape kernel_shape{2, 2};
    const auto rounding_mode = op::RoundingType::CEIL;

    const auto arg = make_shared<ov::op::v0::Parameter>(element::f32, arg_shape);
    const auto mp = this->make_op(arg, strides, dilations, pads_begin, pads_end, kernel_shape, rounding_mode);

    const auto expected_output_shape =
        PartialShape{Dimension::dynamic(), 64, {99, ov::util::dim::inf_bound}, {99, ov::util::dim::inf_bound}};
    EXPECT_EQ(mp->get_output_partial_shape(0), expected_output_shape);
    EXPECT_EQ(mp->get_output_partial_shape(1), expected_output_shape);
}

TYPED_TEST_P(MaxPoolOperator, max_pool_4D_interval_dims_with_dilations) {
    PartialShape arg_shape{{2, 3}, {1, 3}, {2, 13}, {6, 13}};
    auto symbols = set_shape_symbols(arg_shape);
    const Strides strides{1, 1};
    const Strides dilations{2, 3};
    const Shape pads_begin{0, 0};
    const Shape pads_end{0, 0};
    const Shape kernel_shape{2, 2};

    const auto arg = make_shared<ov::op::v0::Parameter>(element::f32, arg_shape);
    const auto mp = this->make_op(arg, strides, dilations, pads_begin, pads_end, kernel_shape);

    const auto expected_output_shape = PartialShape({{2, 3}, {1, 3}, {1, 11}, {3, 10}});
    EXPECT_EQ(mp->get_output_partial_shape(0), expected_output_shape);
    EXPECT_EQ(mp->get_output_partial_shape(1), expected_output_shape);
    EXPECT_THAT(get_shape_symbols(mp->get_output_partial_shape(0)),
                ElementsAre(symbols[0], symbols[1], nullptr, nullptr));
}

TYPED_TEST_P(MaxPoolOperator, max_pool_4D_with_dilations_and_auto_pad_same_upper) {
    const PartialShape arg_shape{1, 3, 13, 13};
    const Strides strides{1, 1};
    const Strides dilations{2, 3};
    const Shape pads_begin{0, 0};
    const Shape pads_end{0, 0};
    const Shape kernel_shape{3, 3};
    const auto rounding_mode = op::RoundingType::FLOOR;
    const auto auto_pad = op::PadType::SAME_UPPER;

    const auto arg = make_shared<ov::op::v0::Parameter>(element::f32, arg_shape);
    const auto mp = this->make_op(arg, strides, dilations, pads_begin, pads_end, kernel_shape, rounding_mode, auto_pad);

    const auto expected_output_shape = PartialShape({1, 3, 13, 13});
    EXPECT_EQ(mp->get_output_partial_shape(0), expected_output_shape);
    EXPECT_EQ(mp->get_output_partial_shape(1), expected_output_shape);
    EXPECT_EQ(mp->get_pads_begin(), (Shape{2, 3}));
    EXPECT_EQ(mp->get_pads_end(), (Shape{2, 3}));
}

TEST(type_prop, max_pool_v14_4D_static_dims_ceil_mode) {
    const PartialShape arg_shape{1, 3, 5, 5};
    const Strides strides{2, 2};
    const Strides dilations{1, 1};
    const Shape pads_begin{1, 1};
    const Shape pads_end{1, 1};
    const Shape kernel_shape{2, 2};
    const auto rounding_mode = op::RoundingType::CEIL;

    const auto arg = make_shared<ov::op::v0::Parameter>(element::f32, arg_shape);
    const auto mp =
        make_shared<op::v14::MaxPool>(arg, strides, dilations, pads_begin, pads_end, kernel_shape, rounding_mode);

    const auto expected_output_shape = PartialShape{1, 3, 4, 4};
    EXPECT_EQ(mp->get_output_partial_shape(0), expected_output_shape);
    EXPECT_EQ(mp->get_output_partial_shape(1), expected_output_shape);
}

TEST(type_prop, max_pool_v14_4D_static_dims_ceil_torch_mode_1) {
    const PartialShape arg_shape{1, 3, 5, 5};
    const Strides strides{2, 2};
    const Strides dilations{1, 1};
    const Shape pads_begin{1, 1};
    const Shape pads_end{1, 1};
    const Shape kernel_shape{2, 2};
    const auto rounding_mode = op::RoundingType::CEIL_TORCH;

    const auto arg = make_shared<ov::op::v0::Parameter>(element::f32, arg_shape);
    const auto mp =
        make_shared<op::v14::MaxPool>(arg, strides, dilations, pads_begin, pads_end, kernel_shape, rounding_mode);

    const auto expected_output_shape = PartialShape{1, 3, 3, 3};
    EXPECT_EQ(mp->get_output_partial_shape(0), expected_output_shape);
    EXPECT_EQ(mp->get_output_partial_shape(1), expected_output_shape);
}

TEST(type_prop, max_pool_v14_4D_static_dims_ceil_torch_mode_2) {
    const PartialShape arg_shape{1, 3, 9, 9};
    const Strides strides{2, 2};
    const Strides dilations{1, 1};
    const Shape pads_begin{1, 1};
    const Shape pads_end{1, 1};
    const Shape kernel_shape{2, 2};
    const auto rounding_mode = op::RoundingType::CEIL_TORCH;

    const auto arg = make_shared<ov::op::v0::Parameter>(element::f32, arg_shape);
    const auto mp =
        make_shared<op::v14::MaxPool>(arg, strides, dilations, pads_begin, pads_end, kernel_shape, rounding_mode);

    const auto expected_output_shape = PartialShape{1, 3, 5, 5};
    EXPECT_EQ(mp->get_output_partial_shape(0), expected_output_shape);
    EXPECT_EQ(mp->get_output_partial_shape(1), expected_output_shape);
}

TEST(type_prop, max_pool_v14_4D_dynamic_dims_with_non_zero_low_range_ceil_torch_mode) {
    PartialShape arg_shape{Dimension::dynamic(), 64, {198, ov::util::dim::inf_bound}, {198, ov::util::dim::inf_bound}};
    const Strides strides{2, 2};
    const Strides dilations{1, 1};
    const Shape pads_begin{0, 0};
    const Shape pads_end{0, 0};
    const Shape kernel_shape{2, 2};
    const auto rounding_mode = op::RoundingType::CEIL_TORCH;

    const auto arg = make_shared<ov::op::v0::Parameter>(element::f32, arg_shape);
    const auto mp =
        make_shared<op::v14::MaxPool>(arg, strides, dilations, pads_begin, pads_end, kernel_shape, rounding_mode);

    const auto expected_output_shape =
        PartialShape{Dimension::dynamic(), 64, {99, ov::util::dim::inf_bound}, {99, ov::util::dim::inf_bound}};
    EXPECT_EQ(mp->get_output_partial_shape(0), expected_output_shape);
    EXPECT_EQ(mp->get_output_partial_shape(1), expected_output_shape);
}

TEST(type_prop, max_pool_v14_4D_dynamic_dims_ceil_mode_1) {
    PartialShape arg_shape{Dimension::dynamic(), 3, {5, ov::util::dim::inf_bound}, {6, 7}};
    const Strides strides{2, 2};
    const Strides dilations{1, 1};
    const Shape pads_begin{1, 1};
    const Shape pads_end{1, 1};
    const Shape kernel_shape{2, 2};
    const auto rounding_mode = op::RoundingType::CEIL;

    const auto arg = make_shared<ov::op::v0::Parameter>(element::f32, arg_shape);
    const auto mp =
        make_shared<op::v14::MaxPool>(arg, strides, dilations, pads_begin, pads_end, kernel_shape, rounding_mode);

    const auto expected_output_shape = PartialShape{Dimension::dynamic(), 3, {4, ov::util::dim::inf_bound}, {4, 5}};
    EXPECT_EQ(mp->get_output_partial_shape(0), expected_output_shape);
    EXPECT_EQ(mp->get_output_partial_shape(1), expected_output_shape);
}

TEST(type_prop, max_pool_v14_4D_dynamic_dims_ceil_torch_mode_1) {
    PartialShape arg_shape{Dimension::dynamic(), 3, {5, ov::util::dim::inf_bound}, {6, 7}};
    const Strides strides{2, 2};
    const Strides dilations{1, 1};
    const Shape pads_begin{1, 1};
    const Shape pads_end{1, 1};
    const Shape kernel_shape{2, 2};
    const auto rounding_mode = op::RoundingType::CEIL_TORCH;

    const auto arg = make_shared<ov::op::v0::Parameter>(element::f32, arg_shape);
    const auto mp =
        make_shared<op::v14::MaxPool>(arg, strides, dilations, pads_begin, pads_end, kernel_shape, rounding_mode);

    const auto expected_output_shape = PartialShape{Dimension::dynamic(), 3, {3, ov::util::dim::inf_bound}, {4, 4}};
    EXPECT_EQ(mp->get_output_partial_shape(0), expected_output_shape);
    EXPECT_EQ(mp->get_output_partial_shape(1), expected_output_shape);
}

TEST(type_prop, max_pool_v14_4D_dynamic_dims_ceil_mode_2) {
    PartialShape arg_shape{Dimension::dynamic(), 3, {14, ov::util::dim::inf_bound}, {15, 17}};
    const Strides strides{3, 3};
    const Strides dilations{1, 1};
    const Shape pads_begin{1, 1};
    const Shape pads_end{1, 1};
    const Shape kernel_shape{3, 3};
    const auto rounding_mode = op::RoundingType::CEIL;

    const auto arg = make_shared<ov::op::v0::Parameter>(element::f32, arg_shape);
    const auto mp =
        make_shared<op::v14::MaxPool>(arg, strides, dilations, pads_begin, pads_end, kernel_shape, rounding_mode);

    const auto expected_output_shape = PartialShape{Dimension::dynamic(), 3, {6, ov::util::dim::inf_bound}, {6, 7}};
    EXPECT_EQ(mp->get_output_partial_shape(0), expected_output_shape);
    EXPECT_EQ(mp->get_output_partial_shape(1), expected_output_shape);
}

TEST(type_prop, max_pool_v14_4D_dynamic_dims_ceil_torch_mode_2) {
    PartialShape arg_shape{Dimension::dynamic(), 3, {14, ov::util::dim::inf_bound}, {15, 17}};
    const Strides strides{3, 3};
    const Strides dilations{1, 1};
    const Shape pads_begin{1, 1};
    const Shape pads_end{1, 1};
    const Shape kernel_shape{3, 3};
    const auto rounding_mode = op::RoundingType::CEIL_TORCH;

    const auto arg = make_shared<ov::op::v0::Parameter>(element::f32, arg_shape);
    const auto mp =
        make_shared<op::v14::MaxPool>(arg, strides, dilations, pads_begin, pads_end, kernel_shape, rounding_mode);

    const auto expected_output_shape = PartialShape{Dimension::dynamic(), 3, {5, ov::util::dim::inf_bound}, {6, 6}};
    EXPECT_EQ(mp->get_output_partial_shape(0), expected_output_shape);
    EXPECT_EQ(mp->get_output_partial_shape(1), expected_output_shape);
}

REGISTER_TYPED_TEST_SUITE_P(MaxPoolOperator,
                            max_pool_3D_no_dilations,
                            max_pool_3D_with_dilations,
                            max_pool_3D_with_dilations_and_padding,
                            max_pool_4D_no_dilations,
                            max_pool_4D_with_dilations,
                            max_pool_4D_dynamic_dims_with_non_zero_low_range_floor_mode,
                            max_pool_4D_dynamic_dims_with_non_zero_low_range_ceil_mode,
                            max_pool_4D_interval_dims_with_dilations,
                            max_pool_4D_with_dilations_and_auto_pad_same_upper);

using MaxPoolOpTypes = Types<ov::op::v8::MaxPool, ov::op::v14::MaxPool>;
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop, MaxPoolOperator, MaxPoolOpTypes);
