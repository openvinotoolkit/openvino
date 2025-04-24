// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/shape.hpp"

#include <gtest/gtest.h>

#include <memory>

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
