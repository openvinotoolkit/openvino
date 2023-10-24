// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/avg_pool.hpp"

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "gtest/gtest.h"

using namespace std;
using namespace testing;

TEST(type_prop, avg_pool_default_ctor) {
    ov::PartialShape arg_shape{1, 3, 32};
    set_shape_labels(arg_shape, 10);
    auto arg = make_shared<ov::op::v0::Parameter>(ov::element::f32, arg_shape);

    auto mp = make_shared<ov::op::v1::AvgPool>();
    mp->set_argument(0, arg);
    mp->set_pads_begin({2});
    mp->set_pads_end({2});
    mp->set_kernel({2});
    mp->set_strides({1});
    mp->set_rounding_type(ov::op::RoundingType::CEIL);
    mp->set_auto_pad(ov::op::PadType::SAME_LOWER);
    mp->validate_and_infer_types();

    EXPECT_TRUE(mp->get_exclude_pad());
    EXPECT_EQ(mp->get_input_size(), 1);
    EXPECT_EQ(mp->get_output_size(), 1);
    EXPECT_EQ(mp->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(mp->get_output_partial_shape(0), ov::PartialShape({1, 3, 32}));
    EXPECT_THAT(get_shape_labels(mp->get_output_partial_shape(0)), ElementsAre(10, 11, ov::no_label));
    EXPECT_EQ(mp->get_pads_begin(), (ov::Shape{1}));
    EXPECT_EQ(mp->get_pads_end(), (ov::Shape{0}));
}

TEST(type_prop, avg_pool_auto_padding) {
    const ov::PartialShape arg_shape{1, 3, 32};
    const ov::Strides strides{1};
    const ov::Shape pads_begin{0};
    const ov::Shape pads_end{0};
    const ov::Shape kernel_shape{2};
    const bool exclude_pad = false;
    const auto rounding_mode = ov::op::RoundingType::FLOOR;
    const auto auto_pad = ov::op::PadType::SAME_LOWER;

    auto arg = make_shared<ov::op::v0::Parameter>(ov::element::f32, arg_shape);
    auto mp = make_shared<ov::op::v1::AvgPool>(arg,
                                               strides,
                                               pads_begin,
                                               pads_end,
                                               kernel_shape,
                                               exclude_pad,
                                               rounding_mode,
                                               auto_pad);

    EXPECT_EQ(mp->get_output_partial_shape(0), ov::PartialShape({1, 3, 32}));
    EXPECT_EQ(mp->get_pads_begin(), (ov::Shape{1}));
    EXPECT_EQ(mp->get_pads_end(), (ov::Shape{0}));
}

TEST(type_prop, avg_pool_explicit_padding_round_ceil_dynamic_dimensions) {
    const ov::PartialShape arg_shape{-1, -1, -1};
    const ov::Strides strides{4};
    const ov::Shape pads_begin{2};
    const ov::Shape pads_end{2};
    const ov::Shape kernel_shape{4};
    const bool exclude_pad = true;
    const auto rounding_mode = ov::op::RoundingType::CEIL;
    const auto auto_pad = ov::op::PadType::EXPLICIT;

    auto arg = make_shared<ov::op::v0::Parameter>(ov::element::f32, arg_shape);
    auto mp = make_shared<ov::op::v1::AvgPool>(arg,
                                               strides,
                                               pads_begin,
                                               pads_end,
                                               kernel_shape,
                                               exclude_pad,
                                               rounding_mode,
                                               auto_pad);

    EXPECT_EQ(mp->get_output_partial_shape(0), ov::PartialShape({-1, -1, {1, -1}}));
    EXPECT_EQ(mp->get_pads_begin(), (ov::Shape{2}));
    EXPECT_EQ(mp->get_pads_end(), (ov::Shape{2}));
}

TEST(type_prop, avg_pool_auto_padding_4D_nc_dims_dynamic_same_lower) {
    const ov::PartialShape arg_shape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), 32, 32};
    const ov::Strides strides{1, 1};
    const ov::Shape pads_begin{0, 0};
    const ov::Shape pads_end{0, 0};
    const ov::Shape kernel_shape{2, 2};
    const bool exclude_pad = true;
    const auto rounding_mode = ov::op::RoundingType::FLOOR;
    const auto auto_pad = ov::op::PadType::SAME_LOWER;

    auto arg = make_shared<ov::op::v0::Parameter>(ov::element::f32, arg_shape);
    auto mp = make_shared<ov::op::v1::AvgPool>(arg,
                                               strides,
                                               pads_begin,
                                               pads_end,
                                               kernel_shape,
                                               exclude_pad,
                                               rounding_mode,
                                               auto_pad);

    EXPECT_EQ(mp->get_output_partial_shape(0),
              ov::PartialShape({ov::Dimension::dynamic(), ov::Dimension::dynamic(), 32, 32}));
    EXPECT_EQ(mp->get_pads_begin(), (ov::Shape{1, 1}));
    EXPECT_EQ(mp->get_pads_end(), (ov::Shape{0, 0}));
}

TEST(type_prop, avg_pool_auto_padding_nc_dims_dynamic_same_upper) {
    const ov::PartialShape arg_shape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), 32, 32};
    const ov::Strides strides{1, 1};
    const ov::Shape pads_begin{0, 0};
    const ov::Shape pads_end{0, 0};
    const ov::Shape kernel_shape{2, 2};
    const bool exclude_pad = false;
    const auto rounding_mode = ov::op::RoundingType::FLOOR;
    const auto auto_pad = ov::op::PadType::SAME_UPPER;

    auto arg = make_shared<ov::op::v0::Parameter>(ov::element::f32, arg_shape);
    auto mp = make_shared<ov::op::v1::AvgPool>(arg,
                                               strides,
                                               pads_begin,
                                               pads_end,
                                               kernel_shape,
                                               exclude_pad,
                                               rounding_mode,
                                               auto_pad);

    EXPECT_EQ(mp->get_output_partial_shape(0),
              ov::PartialShape({ov::Dimension::dynamic(), ov::Dimension::dynamic(), 32, 32}));
    EXPECT_EQ(mp->get_pads_begin(), (ov::Shape{0, 0}));
    EXPECT_EQ(mp->get_pads_end(), (ov::Shape{1, 1}));
}

TEST(type_prop, avg_pool_auto_padding_spatial_dims_dynamic) {
    const ov::PartialShape arg_shape{1, 3, 32, ov::Dimension::dynamic()};
    const ov::Strides strides{1, 1};
    const ov::Shape pads_begin{1, 1};
    const ov::Shape pads_end{0, 0};
    const ov::Shape kernel_shape{2, 2};
    const bool exclude_pad = true;
    const auto rounding_mode = ov::op::RoundingType::FLOOR;
    const auto auto_pad = ov::op::PadType::SAME_LOWER;

    auto arg = make_shared<ov::op::v0::Parameter>(ov::element::f32, arg_shape);
    auto mp = make_shared<ov::op::v1::AvgPool>(arg,
                                               strides,
                                               pads_begin,
                                               pads_end,
                                               kernel_shape,
                                               exclude_pad,
                                               rounding_mode,
                                               auto_pad);

    EXPECT_EQ(mp->get_output_partial_shape(0), ov::PartialShape({1, 3, 32, ov::Dimension::dynamic()}));
    EXPECT_EQ(mp->get_pads_begin(), (ov::Shape{1, 0}));
    EXPECT_EQ(mp->get_pads_end(), (ov::Shape{0, 0}));
}

TEST(type_prop, avg_pool_1d_deduce) {
    const auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{64, 3});
    const ov::Shape kernel{10};
    EXPECT_THROW(const auto unused = make_shared<ov::op::v1::AvgPool>(param,
                                                                      ov::Strides{1},
                                                                      ov::Shape{},
                                                                      ov::Shape{},
                                                                      kernel,
                                                                      true,
                                                                      ov::op::RoundingType::FLOOR),
                 ov::NodeValidationFailure);
}

TEST(type_prop, avg_pool_1d_deduce_strided) {
    const auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{64, 3});
    const ov::Shape kernel{10};
    const auto move_strides = ov::Strides{2};
    EXPECT_THROW(const auto unused = make_shared<ov::op::v1::AvgPool>(param,
                                                                      move_strides,
                                                                      ov::Shape{},
                                                                      ov::Shape{},
                                                                      kernel,
                                                                      true,
                                                                      ov::op::RoundingType::FLOOR),
                 ov::NodeValidationFailure);
}

TEST(type_prop, avg_pool_1d_deduce_strided_small_uneven) {
    const auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{64, 3});
    const ov::Shape kernel{2};
    const auto move_strides = ov::Strides{2};
    EXPECT_THROW(const auto unused = make_shared<ov::op::v1::AvgPool>(param,
                                                                      move_strides,
                                                                      ov::Shape{},
                                                                      ov::Shape{},
                                                                      kernel,
                                                                      true,
                                                                      ov::op::RoundingType::FLOOR),
                 ov::NodeValidationFailure);
}

TEST(type_prop, avg_pool_1d_deduce_strided_small_even) {
    const auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{64, 3});
    const ov::Shape kernel{2};
    const auto move_strides = ov::Strides{2};
    EXPECT_THROW(const auto unused = make_shared<ov::op::v1::AvgPool>(param,
                                                                      move_strides,
                                                                      ov::Shape{},
                                                                      ov::Shape{},
                                                                      kernel,
                                                                      true,
                                                                      ov::op::RoundingType::FLOOR),
                 ov::NodeValidationFailure);
}

TEST(type_prop, avg_pool_2d_deduce) {
    const auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{64, 3, 100, 150});
    const ov::Shape kernel{10, 20};
    const auto avg_pool = make_shared<ov::op::v1::AvgPool>(param,
                                                           ov::Strides{1, 1},
                                                           ov::Shape{0, 0},
                                                           ov::Shape{0, 0},
                                                           kernel,
                                                           true,
                                                           ov::op::RoundingType::FLOOR);

    EXPECT_EQ(avg_pool->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(avg_pool->get_output_partial_shape(0).to_shape(), (ov::Shape{64, 3, 91, 131}));

    EXPECT_EQ(avg_pool->get_strides(), (ov::Strides{1, 1}));
    EXPECT_EQ(avg_pool->get_kernel(), (ov::Shape{10, 20}));
    EXPECT_EQ(avg_pool->get_pads_begin(), (ov::Shape{0, 0}));
    EXPECT_EQ(avg_pool->get_pads_end(), (ov::Shape{0, 0}));
}

TEST(type_prop, avg_pool_2d_deduce_strided) {
    const auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{64, 3, 100, 150});
    const ov::Shape kernel{10, 20};
    const auto move_strides = ov::Strides{2, 3};
    const auto avg_pool = make_shared<ov::op::v1::AvgPool>(param,
                                                           move_strides,
                                                           ov::Shape{0, 0},
                                                           ov::Shape{0, 0},
                                                           kernel,
                                                           true,
                                                           ov::op::RoundingType::FLOOR);

    EXPECT_EQ(avg_pool->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(avg_pool->get_output_partial_shape(0).to_shape(), (ov::Shape{64, 3, 46, 44}));

    EXPECT_EQ(avg_pool->get_strides(), (ov::Strides{2, 3}));
    EXPECT_EQ(avg_pool->get_kernel(), (ov::Shape{10, 20}));
    EXPECT_EQ(avg_pool->get_pads_begin(), (ov::Shape{0, 0}));
    EXPECT_EQ(avg_pool->get_pads_end(), (ov::Shape{0, 0}));
}

TEST(type_prop, avg_pool_3d_deduce_strided_small) {
    const auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{64, 3, 7, 8, 10});
    const ov::Shape kernel{2, 3, 2};
    const auto move_strides = ov::Strides{2, 3, 4};
    const auto avg_pool = make_shared<ov::op::v1::AvgPool>(param,
                                                           move_strides,
                                                           ov::Shape{0, 0, 0},
                                                           ov::Shape{0, 0, 0},
                                                           kernel,
                                                           true,
                                                           ov::op::RoundingType::FLOOR);

    EXPECT_EQ(avg_pool->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(avg_pool->get_output_partial_shape(0).to_shape(), (ov::Shape{64, 3, 3, 2, 3}));

    EXPECT_EQ(avg_pool->get_strides(), (ov::Strides{2, 3, 4}));
    EXPECT_EQ(avg_pool->get_kernel(), (ov::Shape{2, 3, 2}));
    EXPECT_EQ(avg_pool->get_pads_begin(), (ov::Shape{0, 0, 0}));
    EXPECT_EQ(avg_pool->get_pads_end(), (ov::Shape{0, 0, 0}));
}

TEST(type_prop, avg_pool_3d_deduce_strided_padded_small) {
    const auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{64, 3, 7, 8, 10});
    const ov::Shape kernel{2, 3, 2};
    const auto move_strides = ov::Strides{2, 3, 4};
    const ov::Shape pads_begin{5, 6, 4};
    const ov::Shape pads_end{6, 4, 5};
    const auto avg_pool = make_shared<ov::op::v1::AvgPool>(param,
                                                           move_strides,
                                                           pads_begin,
                                                           pads_end,
                                                           kernel,
                                                           false,
                                                           ov::op::RoundingType::FLOOR);

    EXPECT_EQ(avg_pool->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(avg_pool->get_output_partial_shape(0).to_shape(), (ov::Shape{64, 3, 9, 6, 5}));

    EXPECT_EQ(avg_pool->get_strides(), (ov::Strides{2, 3, 4}));
    EXPECT_EQ(avg_pool->get_kernel(), (ov::Shape{2, 3, 2}));
    EXPECT_EQ(avg_pool->get_pads_begin(), (ov::Shape{5, 6, 4}));
    EXPECT_EQ(avg_pool->get_pads_end(), (ov::Shape{6, 4, 5}));
}

TEST(type_prop, avg_pool_invalid_0d_input) {
    const auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{});
    const ov::Shape kernel{};
    EXPECT_THROW(const auto unused = make_shared<ov::op::v1::AvgPool>(param,
                                                                      ov::Strides{1},
                                                                      ov::Shape{},
                                                                      ov::Shape{},
                                                                      kernel,
                                                                      true,
                                                                      ov::op::RoundingType::FLOOR),
                 ov::NodeValidationFailure);
}

TEST(type_prop, avg_pool_invalid_1d_input) {
    const auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2});
    const ov::Shape kernel{};
    EXPECT_THROW(const auto unused = make_shared<ov::op::v1::AvgPool>(param,
                                                                      ov::Strides{1},
                                                                      ov::Shape{},
                                                                      ov::Shape{},
                                                                      kernel,
                                                                      true,
                                                                      ov::op::RoundingType::FLOOR),
                 ov::NodeValidationFailure);
}

TEST(type_prop, avg_pool_invalid_2d_input) {
    const auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 6});
    const ov::Shape kernel{};
    EXPECT_THROW(const auto unused = make_shared<ov::op::v1::AvgPool>(param,
                                                                      ov::Strides{1},
                                                                      ov::Shape{},
                                                                      ov::Shape{},
                                                                      kernel,
                                                                      true,
                                                                      ov::op::RoundingType::FLOOR),
                 ov::NodeValidationFailure);
}

TEST(type_prop, avg_pool_invalid_0_batch_size) {
    const auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{0, 6});
    const ov::Shape kernel{1};
    EXPECT_THROW(const auto unused = make_shared<ov::op::v1::AvgPool>(param,
                                                                      ov::Strides{1},
                                                                      ov::Shape{},
                                                                      ov::Shape{},
                                                                      kernel,
                                                                      true,
                                                                      ov::op::RoundingType::FLOOR),
                 ov::NodeValidationFailure);
}

TEST(type_prop, avg_pool_invalid_0_channels) {
    const auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{6, 0});
    const ov::Shape kernel{1};
    EXPECT_THROW(const auto unused = make_shared<ov::op::v1::AvgPool>(param,
                                                                      ov::Strides{1},
                                                                      ov::Shape{},
                                                                      ov::Shape{},
                                                                      kernel,
                                                                      true,
                                                                      ov::op::RoundingType::FLOOR),
                 ov::NodeValidationFailure);
}

TEST(type_prop, avg_pool_invalid_wrong_number_of_window_dimensions_too_many) {
    const auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{6, 2, 10, 10});
    const ov::Shape kernel{3, 3, 3};
    EXPECT_THROW(const auto unused = make_shared<ov::op::v1::AvgPool>(param,
                                                                      ov::Strides{1},
                                                                      ov::Shape{},
                                                                      ov::Shape{},
                                                                      kernel,
                                                                      true,
                                                                      ov::op::RoundingType::FLOOR),
                 ov::NodeValidationFailure);
}

TEST(type_prop, avg_pool_invalid_wrong_number_of_window_dimensions_too_few) {
    const auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{6, 2, 10, 10});
    const ov::Shape kernel{3};
    EXPECT_THROW(const auto unused = make_shared<ov::op::v1::AvgPool>(param,
                                                                      ov::Strides{1},
                                                                      ov::Shape{},
                                                                      ov::Shape{},
                                                                      kernel,
                                                                      true,
                                                                      ov::op::RoundingType::FLOOR),
                 ov::NodeValidationFailure);
}

TEST(type_prop, avg_pool_invalid_movement_stride_rank) {
    const auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{6, 2, 10, 10});
    const ov::Shape kernel{3, 3};
    const auto move_strides = ov::Strides{2, 3, 8};
    EXPECT_THROW(const auto unused = make_shared<ov::op::v1::AvgPool>(param,
                                                                      move_strides,
                                                                      ov::Shape{},
                                                                      ov::Shape{},
                                                                      kernel,
                                                                      true,
                                                                      ov::op::RoundingType::FLOOR),
                 ov::NodeValidationFailure);
}

TEST(type_prop, avg_pool_invalid_padding_below_rank) {
    const auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{6, 2, 10, 10});
    const ov::Shape kernel{3, 3};
    const auto move_strides = ov::Strides{2, 3};
    const ov::Shape pads_begin{1, 2, 3};
    const ov::Shape pads_end{1, 2};
    EXPECT_THROW(const auto unused = make_shared<ov::op::v1::AvgPool>(param,
                                                                      move_strides,
                                                                      pads_begin,
                                                                      pads_end,
                                                                      kernel,
                                                                      true,
                                                                      ov::op::RoundingType::FLOOR),
                 ov::NodeValidationFailure);
}

TEST(type_prop, avg_pool_invalid_padding_above_rank) {
    const auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{6, 2, 10, 10});
    const ov::Shape kernel{3, 3};
    const auto move_strides = ov::Strides{2, 3};
    const ov::Shape pads_begin{1, 2};
    const ov::Shape pads_end{1, 2, 3};
    EXPECT_THROW(const auto unused = make_shared<ov::op::v1::AvgPool>(param,
                                                                      move_strides,
                                                                      pads_begin,
                                                                      pads_end,
                                                                      kernel,
                                                                      true,
                                                                      ov::op::RoundingType::FLOOR),
                 ov::NodeValidationFailure);
}

TEST(type_prop, avg_pool_invalid_input_item_size_0) {
    const auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{6, 2, 0, 10});
    const ov::Shape kernel{3, 3};
    EXPECT_THROW(const auto unused = make_shared<ov::op::v1::AvgPool>(param,
                                                                      ov::Strides{1},
                                                                      ov::Shape{},
                                                                      ov::Shape{},
                                                                      kernel,
                                                                      true,
                                                                      ov::op::RoundingType::FLOOR),
                 ov::NodeValidationFailure);
}

TEST(type_prop, avg_pool_invalid_window_size_0) {
    const auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{6, 2, 10, 10});
    const ov::Shape kernel{3, 0};
    EXPECT_THROW(const auto unused = make_shared<ov::op::v1::AvgPool>(param,
                                                                      ov::Strides{1},
                                                                      ov::Shape{},
                                                                      ov::Shape{},
                                                                      kernel,
                                                                      true,
                                                                      ov::op::RoundingType::FLOOR),
                 ov::NodeValidationFailure);
}

TEST(type_prop, avg_pool_invalid_dilated_too_large) {
    const auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{6, 2, 8, 8});
    const ov::Shape kernel{9, 9};
    EXPECT_THROW(const auto unused = make_shared<ov::op::v1::AvgPool>(param,
                                                                      ov::Strides{1},
                                                                      ov::Shape{},
                                                                      ov::Shape{},
                                                                      kernel,
                                                                      true,
                                                                      ov::op::RoundingType::FLOOR),
                 ov::NodeValidationFailure);
}

TEST(type_prop, avg_pool_larger_than_pre_padding_but_fits_in_post_padding) {
    const auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{6, 2, 8, 8});
    const ov::Shape kernel{9, 9};
    const ov::Strides window_strides{1, 1};
    const ov::Shape pads_begin{0, 0};
    const ov::Shape pads_end{1, 1};
    const auto avg_pool = make_shared<ov::op::v1::AvgPool>(param,
                                                           window_strides,
                                                           pads_begin,
                                                           pads_end,
                                                           kernel,
                                                           true,
                                                           ov::op::RoundingType::FLOOR);

    ASSERT_EQ(avg_pool->get_output_element_type(0), ov::element::f32);
    ASSERT_EQ(avg_pool->get_output_partial_shape(0).to_shape(), (ov::Shape{6, 2, 1, 1}));
}

TEST(type_prop, avg_pool_invalid_movement_stride_0) {
    const auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{6, 2, 10, 10});
    const ov::Shape kernel{3, 3};
    const auto move_strides = ov::Strides{0, 1};
    EXPECT_THROW(const auto unused = make_shared<ov::op::v1::AvgPool>(param,
                                                                      move_strides,
                                                                      ov::Shape{},
                                                                      ov::Shape{},
                                                                      kernel,
                                                                      true,
                                                                      ov::op::RoundingType::FLOOR),
                 ov::NodeValidationFailure);
}

TEST(type_prop, avg_pool_partial_rank_dynamic_ok) {
    const ov::PartialShape arg_shape{ov::PartialShape::dynamic()};
    const ov::Shape kernel{2, 3, 4, 5};
    const ov::Strides window_movement_strides{1, 1, 1, 1};
    const ov::Shape pads_begin{0, 0, 0, 0};
    const ov::Shape pads_end{0, 0, 0, 0};

    const auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, arg_shape);
    auto ap = make_shared<ov::op::v1::AvgPool>(param,
                                               window_movement_strides,
                                               pads_begin,
                                               pads_end,
                                               kernel,
                                               false,
                                               ov::op::RoundingType::FLOOR);

    EXPECT_EQ(ap->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(ap->get_output_partial_shape(0), ov::PartialShape(ov::PartialShape::dynamic(6)));
}

TEST(type_prop, avg_pool_partial_rank_dynamic_attrib_rank_mismatch) {
    const ov::PartialShape arg_shape{ov::PartialShape::dynamic()};
    const ov::Shape kernel{2, 3, 4, 5};
    const ov::Strides window_movement_strides{1, 1, 1, 1, 1};
    const ov::Shape pads_begin{0, 0, 0, 0};
    const ov::Shape pads_end{0, 0, 0, 0};

    const auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, arg_shape);

    EXPECT_THROW(const auto unused = make_shared<ov::op::v1::AvgPool>(param,
                                                                      window_movement_strides,
                                                                      pads_begin,
                                                                      pads_end,
                                                                      kernel,
                                                                      false,
                                                                      ov::op::RoundingType::FLOOR),
                 ov::NodeValidationFailure);
}

TEST(type_prop, avg_pool_partial_rank_static_dynamic_ok) {
    const ov::PartialShape arg_shape{ov::PartialShape::dynamic(5)};
    const ov::Shape kernel{2, 3, 4};
    const ov::Strides window_movement_strides{1, 1, 1};
    const ov::Shape pads_begin{0, 0, 0};
    const ov::Shape pads_end{0, 0, 0};

    const auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, arg_shape);
    auto ap = make_shared<ov::op::v1::AvgPool>(param,
                                               window_movement_strides,
                                               pads_begin,
                                               pads_end,
                                               kernel,
                                               false,
                                               ov::op::RoundingType::FLOOR);

    EXPECT_EQ(ap->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(ap->get_output_partial_shape(0), ov::PartialShape({-1, -1, {1, -1}, {1, -1}, {1, -1}}));
}

TEST(type_prop, avg_pool_partial_rank_static_dynamic_some_dims_known_ok) {
    const ov::PartialShape arg_shape{5, ov::Dimension::dynamic(), 8, ov::Dimension::dynamic(), 4};
    const ov::Shape kernel{2, 3, 4};
    const ov::Strides window_movement_strides{1, 1, 1};
    const ov::Shape pads_begin{0, 0, 0};
    const ov::Shape pads_end{0, 0, 0};

    const auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, arg_shape);
    auto ap = make_shared<ov::op::v1::AvgPool>(param,
                                               window_movement_strides,
                                               pads_begin,
                                               pads_end,
                                               kernel,
                                               false,
                                               ov::op::RoundingType::FLOOR);

    EXPECT_EQ(ap->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(ap->get_output_partial_shape(0), ov::PartialShape(ov::PartialShape{5, -1, 7, {1, -1}, 1}));
}

TEST(type_prop, avg_pool_partial_rank_static_dynamic_attrib_rank_mismatch) {
    const ov::PartialShape arg_shape{5, ov::Dimension::dynamic(), 8, ov::Dimension::dynamic(), 4};
    const ov::Shape kernel{2, 3, 4, 5};
    const ov::Strides window_movement_strides{1, 1, 1};
    const ov::Shape pads_begin{0, 0, 0};
    const ov::Shape pads_end{0, 0, 0};

    const auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, arg_shape);

    EXPECT_THROW(const auto unused = make_shared<ov::op::v1::AvgPool>(param,
                                                                      window_movement_strides,
                                                                      pads_begin,
                                                                      pads_end,
                                                                      kernel,
                                                                      true,
                                                                      ov::op::RoundingType::FLOOR),
                 ov::NodeValidationFailure);
}

TEST(type_prop, avg_pool_partial_rank_static_dynamic_window_not_too_big) {
    const ov::PartialShape arg_shape{5, ov::Dimension::dynamic(), 8, ov::Dimension::dynamic(), 4};
    const ov::Shape kernel{9, 3, 4};
    const ov::Strides window_movement_strides{1, 1, 1};
    const ov::Shape pads_begin{0, 0, 0};
    const ov::Shape pads_end{0, 0, 0};

    const auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, arg_shape);

    EXPECT_THROW(const auto unused = make_shared<ov::op::v1::AvgPool>(param,
                                                                      window_movement_strides,
                                                                      pads_begin,
                                                                      pads_end,
                                                                      kernel,
                                                                      true,
                                                                      ov::op::RoundingType::FLOOR),
                 ov::NodeValidationFailure);
}

TEST(type_prop, avg_pool_partial_rank_static_dynamic_padded_window_not_too_big) {
    const ov::PartialShape arg_shape{5, ov::Dimension::dynamic(), 8, ov::Dimension::dynamic(), 4};
    const ov::Shape kernel{9, 3, 4};
    const ov::Strides window_movement_strides{1, 1, 1};
    const ov::Shape pads_begin{0, 0, 0};
    const ov::Shape pads_end{1, 0, 0};

    const auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, arg_shape);
    auto ap = make_shared<ov::op::v1::AvgPool>(param,
                                               window_movement_strides,
                                               pads_begin,
                                               pads_end,
                                               kernel,
                                               true,
                                               ov::op::RoundingType::FLOOR);

    EXPECT_EQ(ap->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(ap->get_output_partial_shape(0),
              ov::PartialShape(ov::PartialShape{5, ov::Dimension::dynamic(), 1, {1, -1}, 1}));
}

TEST(type_prop, avg_pool_partial_rank_static_dynamic_window_in_padding) {
    const ov::PartialShape arg_shape{5, ov::Dimension::dynamic(), 8, ov::Dimension::dynamic(), 4};
    const ov::Shape kernel{9, 3, 4};
    const ov::Strides window_movement_strides{1, 1, 1};
    const ov::Shape pads_begin{0, 0, 0};
    const ov::Shape pads_end{0, 0, 0};

    const auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, arg_shape);

    EXPECT_THROW(const auto unused = make_shared<ov::op::v1::AvgPool>(param,
                                                                      window_movement_strides,
                                                                      pads_begin,
                                                                      pads_end,
                                                                      kernel,
                                                                      true,
                                                                      ov::op::RoundingType::FLOOR),
                 ov::NodeValidationFailure);
}

TEST(type_prop, avg_pool_kernel_dilation_not_compatible_with_padding_begin) {
    const ov::PartialShape arg_shape{5, -1, 8};
    const ov::Shape kernel{9};
    const ov::Strides window_movement_strides{1};
    const ov::Shape pads_begin{10};
    const ov::Shape pads_end{0};

    const auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, arg_shape);

    OV_EXPECT_THROW(const auto unused = make_shared<ov::op::v1::AvgPool>(param,
                                                                         window_movement_strides,
                                                                         pads_begin,
                                                                         pads_end,
                                                                         kernel,
                                                                         true,
                                                                         ov::op::RoundingType::FLOOR),
                    ov::NodeValidationFailure,
                    HasSubstr("Kernel after dilation is sometimes entirely in the padding area for axis 0"));
}

TEST(type_prop, avg_pool_kernel_dilation_not_compatible_with_padding_end) {
    const ov::PartialShape arg_shape{5, -1, 8};
    const ov::Shape kernel{9};
    const ov::Strides window_movement_strides{1};
    const ov::Shape pads_begin{0};
    const ov::Shape pads_end{10};

    const auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, arg_shape);

    OV_EXPECT_THROW(const auto unused = make_shared<ov::op::v1::AvgPool>(param,
                                                                         window_movement_strides,
                                                                         pads_begin,
                                                                         pads_end,
                                                                         kernel,
                                                                         true,
                                                                         ov::op::RoundingType::FLOOR),
                    ov::NodeValidationFailure,
                    HasSubstr("Kernel after dilation is sometimes entirely in the padding area for axis 0"));
}
