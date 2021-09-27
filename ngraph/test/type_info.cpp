// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/core/type.hpp"
#include "openvino/opsets/opset.hpp"

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

TEST(type_info, find_in_map) {
    ov::DiscreteTypeInfo a("Mod", 1, "opset1");
    ov::DiscreteTypeInfo b("Prelu", 0, "opset1");
    ov::DiscreteTypeInfo c("Vector", 0);
    ov::DiscreteTypeInfo d("Mod", 1, "opset3");
    ov::DiscreteTypeInfo f("Mod", 2);

    std::map<ov::DiscreteTypeInfo, int> test_map;
    test_map[a] = 1;
    test_map[b] = 1;
    test_map[c] = 1;

    const auto& opset = ov::get_opset8();
    for (const auto& type : opset.get_types_info()) {
        test_map[type] = 2;
        std::string name = type.name;
        ov::DiscreteTypeInfo t(name.c_str(), 1000);
        ov::DiscreteTypeInfo t2(name.c_str(), 0);
        test_map[t] = 3;
        test_map[t2] = 4;
        std::string name1 = "a" + name;
        ov::DiscreteTypeInfo t3(name1.c_str(), 1000);
        ov::DiscreteTypeInfo t4(name1.c_str(), 0);
        test_map[t3] = 5;
        test_map[t4] = 6;
        std::string name2 = name + "z";
        ov::DiscreteTypeInfo t5(name2.c_str(), 1000);
        ov::DiscreteTypeInfo t6(name2.c_str(), 0);
        test_map[t5] = 7;
        test_map[t6] = 8;
    }

    for (const auto& it : test_map) {
        std::cout << it.first << std::endl;
        ASSERT_NE(test_map.end(), test_map.find(it.first));
    }

    ASSERT_NE(test_map.end(), test_map.find(a));
    ASSERT_NE(test_map.end(), test_map.find(b));
    ASSERT_NE(test_map.end(), test_map.find(c));
    ASSERT_EQ(test_map.end(), test_map.find(d));
    ASSERT_EQ(test_map.end(), test_map.find(f));
}
