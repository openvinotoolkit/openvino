// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/validation_util.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;

TEST(optimize, test)
{
    auto ps = PartialShape{};
    ASSERT_TRUE(ps.rank().is_static());
    ASSERT_TRUE(ps.is_static());
    ASSERT_EQ(ps.rank().get_length(), 0);
}