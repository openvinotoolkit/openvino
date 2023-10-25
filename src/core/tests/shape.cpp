// Copyright (C) 2018-2023 Intel Corporation
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
