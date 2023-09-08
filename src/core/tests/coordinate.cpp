// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/coordinate.hpp"

#include <memory>
#include <numeric>
#include <string>

#include "common_test_utils/ndarray.hpp"
#include "common_test_utils/test_tools.hpp"
#include "gtest/gtest.h"
#include "openvino/reference/utils/coordinate_transform.hpp"

using namespace std;
using namespace ov;

OPENVINO_SUPPRESS_DEPRECATED_START
TEST(coordinate, shape0d) {
    auto ct = ov::CoordinateTransform({});
    ASSERT_EQ(shape_size(ct.get_target_shape()), 1);
    auto it = ct.begin();
    EXPECT_EQ(*it++, ov::Coordinate({}));
    EXPECT_TRUE(it == ct.end());
}

TEST(coordinate, shape1d) {
    auto ct = ov::CoordinateTransform({3});
    ASSERT_EQ(shape_size(ct.get_target_shape()), 3);
    auto it = ct.begin();
    EXPECT_EQ(*it++, Coordinate({0}));
    EXPECT_EQ(*it++, Coordinate({1}));
    EXPECT_EQ(*it++, Coordinate({2}));
    EXPECT_TRUE(it == ct.end());
}

TEST(coordinate, shape2d) {
    auto ct = ov::CoordinateTransform({2, 3});
    ASSERT_EQ(shape_size(ct.get_target_shape()), 6);
    auto it = ct.begin();
    EXPECT_EQ(*it++, Coordinate({0, 0}));
    EXPECT_EQ(*it++, Coordinate({0, 1}));
    EXPECT_EQ(*it++, Coordinate({0, 2}));
    EXPECT_EQ(*it++, Coordinate({1, 0}));
    EXPECT_EQ(*it++, Coordinate({1, 1}));
    EXPECT_EQ(*it++, Coordinate({1, 2}));
    EXPECT_TRUE(it == ct.end());
}

TEST(coordinate, shape3d) {
    auto ct = ov::CoordinateTransform({2, 3, 4});
    ASSERT_EQ(shape_size(ct.get_target_shape()), 24);
    auto it = ct.begin();
    EXPECT_EQ(*it++, Coordinate({0, 0, 0}));
    EXPECT_EQ(*it++, Coordinate({0, 0, 1}));
    EXPECT_EQ(*it++, Coordinate({0, 0, 2}));
    EXPECT_EQ(*it++, Coordinate({0, 0, 3}));
    EXPECT_EQ(*it++, Coordinate({0, 1, 0}));
    EXPECT_EQ(*it++, Coordinate({0, 1, 1}));
    EXPECT_EQ(*it++, Coordinate({0, 1, 2}));
    EXPECT_EQ(*it++, Coordinate({0, 1, 3}));
    EXPECT_EQ(*it++, Coordinate({0, 2, 0}));
    EXPECT_EQ(*it++, Coordinate({0, 2, 1}));
    EXPECT_EQ(*it++, Coordinate({0, 2, 2}));
    EXPECT_EQ(*it++, Coordinate({0, 2, 3}));
    EXPECT_EQ(*it++, Coordinate({1, 0, 0}));
    EXPECT_EQ(*it++, Coordinate({1, 0, 1}));
    EXPECT_EQ(*it++, Coordinate({1, 0, 2}));
    EXPECT_EQ(*it++, Coordinate({1, 0, 3}));
    EXPECT_EQ(*it++, Coordinate({1, 1, 0}));
    EXPECT_EQ(*it++, Coordinate({1, 1, 1}));
    EXPECT_EQ(*it++, Coordinate({1, 1, 2}));
    EXPECT_EQ(*it++, Coordinate({1, 1, 3}));
    EXPECT_EQ(*it++, Coordinate({1, 2, 0}));
    EXPECT_EQ(*it++, Coordinate({1, 2, 1}));
    EXPECT_EQ(*it++, Coordinate({1, 2, 2}));
    EXPECT_EQ(*it++, Coordinate({1, 2, 3}));
    EXPECT_TRUE(it == ct.end());
}

TEST(coordinate, zero_sized_axis) {
    auto ct = ov::CoordinateTransform({2, 0, 4});
    ASSERT_EQ(shape_size(ct.get_target_shape()), 0);
    auto it = ct.begin();
    EXPECT_TRUE(it == ct.end());
}

TEST(coordinate, corner) {
    Shape source_shape{10, 10};
    Coordinate source_start_corner = Coordinate{3, 3};
    Coordinate source_end_corner{6, 6};
    Strides source_strides = Strides(source_shape.size(), 1);
    AxisVector source_axis_order(source_shape.size());
    iota(source_axis_order.begin(), source_axis_order.end(), 0);
    CoordinateDiff target_padding_below = CoordinateDiff(source_shape.size(), 0);
    CoordinateDiff target_padding_above = CoordinateDiff(source_shape.size(), 0);
    Strides source_dilation_strides = Strides(source_shape.size(), 1);

    auto ct = ov::CoordinateTransform(source_shape,
                                      source_start_corner,
                                      source_end_corner,
                                      source_strides,
                                      source_axis_order,
                                      target_padding_below,
                                      target_padding_above,
                                      source_dilation_strides);

    ASSERT_EQ(shape_size(ct.get_target_shape()), 9);
    auto it = ct.begin();
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({3, 3}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({3, 4}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({3, 5}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({4, 3}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({4, 4}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({4, 5}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({5, 3}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({5, 4}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({5, 5}));
    EXPECT_TRUE(it == ct.end());
}

TEST(coordinate, strides) {
    Shape source_shape{10, 10};
    Coordinate source_start_corner = Coordinate{0, 0};
    Coordinate source_end_corner{source_shape};
    Strides source_strides = Strides({2, 3});
    AxisVector source_axis_order(source_shape.size());
    iota(source_axis_order.begin(), source_axis_order.end(), 0);
    CoordinateDiff target_padding_below = CoordinateDiff(source_shape.size(), 0);
    CoordinateDiff target_padding_above = CoordinateDiff(source_shape.size(), 0);
    Strides source_dilation_strides = Strides(source_shape.size(), 1);

    auto ct = ov::CoordinateTransform(source_shape,
                                      source_start_corner,
                                      source_end_corner,
                                      source_strides,
                                      source_axis_order,
                                      target_padding_below,
                                      target_padding_above,
                                      source_dilation_strides);

    ASSERT_EQ(shape_size(ct.get_target_shape()), 20);
    auto it = ct.begin();
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({0, 0}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({0, 3}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({0, 6}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({0, 9}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({2, 0}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({2, 3}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({2, 6}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({2, 9}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({4, 0}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({4, 3}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({4, 6}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({4, 9}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({6, 0}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({6, 3}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({6, 6}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({6, 9}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({8, 0}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({8, 3}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({8, 6}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({8, 9}));
    EXPECT_TRUE(it == ct.end());
}

TEST(coordinate, axis_order) {
    Shape source_shape{3, 2, 4};
    Coordinate source_start_corner = Coordinate{0, 0, 0};
    Coordinate source_end_corner{source_shape};
    Strides source_strides = Strides(source_shape.size(), 1);
    AxisVector source_axis_order({1, 2, 0});
    CoordinateDiff target_padding_below = CoordinateDiff(source_shape.size(), 0);
    CoordinateDiff target_padding_above = CoordinateDiff(source_shape.size(), 0);
    Strides source_dilation_strides = Strides(source_shape.size(), 1);

    auto ct = ov::CoordinateTransform(source_shape,
                                      source_start_corner,
                                      source_end_corner,
                                      source_strides,
                                      source_axis_order,
                                      target_padding_below,
                                      target_padding_above,
                                      source_dilation_strides);

    ASSERT_EQ(shape_size(ct.get_target_shape()), 24);
    auto it = ct.begin();
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({0, 0, 0}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({1, 0, 0}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({2, 0, 0}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({0, 0, 1}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({1, 0, 1}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({2, 0, 1}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({0, 0, 2}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({1, 0, 2}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({2, 0, 2}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({0, 0, 3}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({1, 0, 3}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({2, 0, 3}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({0, 1, 0}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({1, 1, 0}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({2, 1, 0}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({0, 1, 1}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({1, 1, 1}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({2, 1, 1}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({0, 1, 2}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({1, 1, 2}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({2, 1, 2}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({0, 1, 3}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({1, 1, 3}));
    EXPECT_EQ(ct.to_source_coordinate(*it++), Coordinate({2, 1, 3}));
    EXPECT_TRUE(it == ct.end());
}
