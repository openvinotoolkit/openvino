// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "visitors/visitors.hpp"

TEST(attributes, value_map) {
    ov::test::ValueMap value_map;
    bool a = true;
    int8_t b = 2;
    value_map.insert("a", a);
    value_map.insert("b", b);
    bool g_a = value_map.get<bool>("a");
    int8_t g_b = value_map.get<int8_t>("b");
    EXPECT_EQ(a, g_a);
    EXPECT_EQ(b, g_b);
}
