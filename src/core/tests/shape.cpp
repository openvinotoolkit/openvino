// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"

using namespace std;
using namespace ngraph;

TEST(shape, test_shape_size) {
    ASSERT_EQ(1, shape_size(Shape{}));
    ASSERT_EQ(2 * 3 * 5, shape_size(Shape{2, 3, 5}));
}

TEST(shape, test_shape_strides) {
    ASSERT_EQ(Strides{}, row_major_strides(Shape{}));
    ASSERT_EQ(Strides{1}, row_major_strides(Shape{3}));
    ASSERT_EQ((Strides{7, 1}), row_major_strides(Shape{2, 7}));
    ASSERT_EQ((Strides{84, 12, 1}), row_major_strides(Shape{5, 7, 12}));
}
