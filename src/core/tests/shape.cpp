// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/shape.hpp"

#include <gtest/gtest.h>

#include <memory>

#include "openvino/core/shape_util.hpp"

using namespace std;
using namespace ov;

TEST(shape, test_shape_size) {
    ASSERT_EQ(1, shape_size(ov::Shape{}));
    ASSERT_EQ(2 * 3 * 5, shape_size(Shape{2, 3, 5}));
}

TEST(shape, test_shape_strides) {
    ASSERT_EQ(Strides{}, row_major_strides(Shape{}));
    ASSERT_EQ(Strides{1}, row_major_strides(Shape{3}));
    ASSERT_EQ((Strides{7, 1}), row_major_strides(Shape{2, 7}));
    ASSERT_EQ((Strides{84, 12, 1}), row_major_strides(Shape{5, 7, 12}));
}

TEST(shape, at) {
    const auto shape = ov::Shape{100, 200, 5, 6, 7};

    EXPECT_EQ(shape.at(2), 5);
    EXPECT_EQ(shape.at(0), 100);
    EXPECT_EQ(shape.at(1), 200);
    EXPECT_EQ(shape.at(4), 7);

    EXPECT_EQ(shape.at(-3), 5);
    EXPECT_EQ(shape.at(-5), 100);
    EXPECT_EQ(shape.at(-4), 200);
    EXPECT_EQ(shape.at(-1), 7);
}

TEST(shape, subscribe_operator) {
    const auto shape = ov::Shape{100, 200, 5, 6, 7};

    EXPECT_EQ(shape[2], 5);
    EXPECT_EQ(shape[0], 100);
    EXPECT_EQ(shape[1], 200);
    EXPECT_EQ(shape[4], 7);

    EXPECT_EQ(shape[-3], 5);
    EXPECT_EQ(shape[-5], 100);
    EXPECT_EQ(shape[-4], 200);
    EXPECT_EQ(shape[-1], 7);
}

TEST(shape, at_throw_exception) {
    auto shape = ov::Shape{1, 2, 3, 4, 5, 6, 7};

    EXPECT_THROW(shape.at(7), ov::Exception);
    EXPECT_THROW(shape.at(1000), ov::Exception);
    EXPECT_THROW(shape.at(-8), ov::Exception);
    EXPECT_THROW(shape.at(-80000), ov::Exception);
}

TEST(shape, shape_size_overflow) {
    constexpr auto max = std::numeric_limits<size_t>::max();

    EXPECT_EQ(ov::util::shape_size_safe({1, 2, 3, 4, 5, 6, 7}).value(), (2 * 3 * 4 * 5 * 6 * 7));
    EXPECT_EQ(ov::util::shape_size_safe({3, max / 4, 2}), std::nullopt);
    EXPECT_EQ(ov::util::shape_size_safe({3, max / 4}).value(), (3 * (max / 4)));
}

TEST(shape, has_byte_strides) {
    EXPECT_TRUE(ov::util::has_byte_strides(element::f32, Shape{}));
    EXPECT_TRUE(ov::util::has_byte_strides(element::f32, Shape{2, 3, 5}));
    // sub-byte types require the last dimension's packed row to end on a byte boundary
    EXPECT_TRUE(ov::util::has_byte_strides(element::u4, Shape{2, 3, 8}));
    EXPECT_FALSE(ov::util::has_byte_strides(element::u4, Shape{2, 3, 3}));
    EXPECT_TRUE(ov::util::has_byte_strides(element::u1, Shape{8}));
    EXPECT_FALSE(ov::util::has_byte_strides(element::u1, Shape{10}));

    // a bitwidth that does not evenly divide a byte can never be byte-aligned
    EXPECT_FALSE(ov::util::has_byte_strides(element::u3, Shape{8}));

    // sub-byte scalar (rank 0) has no last dimension to align
    EXPECT_FALSE(ov::util::has_byte_strides(element::u4, Shape{}));
}

TEST(shape, get_dense_byte_strides) {
    EXPECT_EQ(ov::util::get_dense_byte_strides(element::f32, Shape{2, 3, 5}), Strides({60, 20, 4}));

    // last dimension is one packed row (3 bytes); outer strides count whole rows
    EXPECT_EQ(ov::util::get_dense_byte_strides(element::u4, Shape{2, 3, 8}), Strides({12, 4, 1}));

    // zero-sized last dimension collapses all strides to 0
    EXPECT_EQ(ov::util::get_dense_byte_strides(element::u4, Shape{2, 3, 0}), Strides({0, 0, 0}));
}
