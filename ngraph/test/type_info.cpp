// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/core/type.hpp"

TEST(type_info, compare_old_type) {
    ov::DiscreteTypeInfo type1("type1", 0);
    ov::DiscreteTypeInfo type2("type2", 0);
    ov::DiscreteTypeInfo type3("type1", 1);
    ov::DiscreteTypeInfo type4("type3", 0, &type1);
    ov::DiscreteTypeInfo type5("type3", 0, &type2);
    ASSERT_TRUE(type1 != type2);
    ASSERT_TRUE(type1 == type1);
    ASSERT_TRUE(type1 < type2);
    ASSERT_TRUE(type2 > type1);
    ASSERT_TRUE(type1 >= type1);
    ASSERT_TRUE(type1 <= type1);
    ASSERT_TRUE(type3 >= type1);
    ASSERT_TRUE(type1 <= type3);
    ASSERT_FALSE(type4 != type5);
    ASSERT_FALSE(type4 < type5);
}

TEST(type_info, compare_new_type) {
    ov::DiscreteTypeInfo type1("type1", 0, "version1");
    ov::DiscreteTypeInfo type2("type2", 0, "version1");
    ov::DiscreteTypeInfo type3("type1", 1, "version2");
    ov::DiscreteTypeInfo type4("type3", 0, "version3", &type1);
    ov::DiscreteTypeInfo type5("type3", 0, "version3", &type2);
    ASSERT_TRUE(type1 != type2);
    ASSERT_TRUE(type1 == type1);
    ASSERT_TRUE(type1 < type2);
    ASSERT_TRUE(type2 > type1);
    ASSERT_TRUE(type1 >= type1);
    ASSERT_TRUE(type1 <= type1);
    ASSERT_TRUE(type3 >= type1);
    ASSERT_TRUE(type1 <= type3);
    ASSERT_FALSE(type4 != type5);
    ASSERT_FALSE(type4 < type5);
}

TEST(type_info, compare_new_with_old_type) {
    ov::DiscreteTypeInfo type1("type1", 0, "version1");
    ov::DiscreteTypeInfo type1_o("type1", 0);
    ASSERT_TRUE(type1 == type1_o);
}
