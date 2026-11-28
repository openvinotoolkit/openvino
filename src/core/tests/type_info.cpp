// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <functional>

#include "openvino/core/type.hpp"
#include "openvino/opsets/opset.hpp"

TEST(type_info, compare_old_type) {
    ov::DiscreteTypeInfo type1("type1");
    ov::DiscreteTypeInfo type2("type2");
    ov::DiscreteTypeInfo type3("type1");
    ov::DiscreteTypeInfo type4("type3", &type1);
    ov::DiscreteTypeInfo type5("type3", &type2);
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
    ov::DiscreteTypeInfo type1("type1", "version1");
    ov::DiscreteTypeInfo type2("type2", "version1");
    ov::DiscreteTypeInfo type3("type1", "version2");
    ov::DiscreteTypeInfo type4("type3", "version3", &type1);
    ov::DiscreteTypeInfo type5("type3", "version3", &type2);
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
    ov::DiscreteTypeInfo type1("type1", "version1");
    ov::DiscreteTypeInfo type1_o("type1", "version1");
    ASSERT_TRUE(type1 == type1_o);
}

// Expected values are fixed on purpose: recomputing them with the tested algorithm would not detect its changes.
TEST(type_info, check_hash_value) {
    EXPECT_EQ(static_cast<size_t>(0x1591723d39f0c6b2ULL), ov::DiscreteTypeInfo("type1", "version1").hash());
    EXPECT_EQ(static_cast<size_t>(0xf709f69eefe342f2ULL), ov::DiscreteTypeInfo("type1").hash());
    EXPECT_EQ(static_cast<size_t>(0xe6546b64ULL), ov::DiscreteTypeInfo("").hash());
    EXPECT_EQ(static_cast<size_t>(0x16ab13a078cd7dbbULL), ov::DiscreteTypeInfo("type", "").hash());
    EXPECT_EQ(static_cast<size_t>(0x0e3866429f7ecb27ULL), ov::DiscreteTypeInfo("Add", "opset1").hash());
}

TEST(type_info, hash_properties) {
    const ov::DiscreteTypeInfo type("type1", "version1");
    const char* no_version = nullptr;

    EXPECT_EQ(type.hash(), ov::DiscreteTypeInfo("type1", "version1").hash());
    EXPECT_NE(type.hash(), ov::DiscreteTypeInfo("type2", "version1").hash());
    EXPECT_NE(type.hash(), ov::DiscreteTypeInfo("type1", "version2").hash());
    EXPECT_NE(type.hash(), ov::DiscreteTypeInfo("version1", "type1").hash());
    // parent is not a part of the type identity
    EXPECT_EQ(type.hash(), ov::DiscreteTypeInfo("type1", "version1", &type).hash());
    // missing version is the same as empty one, as in operator==
    EXPECT_EQ(ov::DiscreteTypeInfo("type1", no_version).hash(), ov::DiscreteTypeInfo("type1", "").hash());
    EXPECT_EQ(ov::DiscreteTypeInfo("type1", no_version).hash(), ov::DiscreteTypeInfo("type1").hash());
    EXPECT_EQ(type.hash(), std::hash<ov::DiscreteTypeInfo>{}(type));

    constexpr ov::DiscreteTypeInfo constant_type{"type1", "version1"};
    EXPECT_EQ(type.hash(), constant_type.hash());
    EXPECT_EQ(type, constant_type);
}

TEST(type_info, find_in_map) {
    std::vector<std::string> vector_names;
    ov::DiscreteTypeInfo a("Mod", "opset1");
    ov::DiscreteTypeInfo b("Prelu", "opset1");
    ov::DiscreteTypeInfo c("Vector");
    ov::DiscreteTypeInfo d("Mod", "opset3");
    ov::DiscreteTypeInfo f("Mod", "opset2");

    std::map<ov::DiscreteTypeInfo, int> test_map;
    test_map[a] = 1;
    test_map[b] = 1;
    test_map[c] = 1;

    const auto& opset = ov::get_opset9();
    // Reserve memory to avoid reallocation and copy of strings
    // because type info uses pointers from the original memory
    vector_names.reserve(opset.size() * 3);
    for (const auto& type : opset.get_types_info()) {
        test_map[type] = 2;
        std::string name = type.name;
        vector_names.emplace_back(name);
        ov::DiscreteTypeInfo t(vector_names.rbegin()->c_str());
        ov::DiscreteTypeInfo t2(vector_names.rbegin()->c_str());
        test_map[t] = 3;
        test_map[t2] = 4;
        std::string name1 = "a" + name;
        vector_names.emplace_back(name1);
        ov::DiscreteTypeInfo t3(vector_names.rbegin()->c_str());
        ov::DiscreteTypeInfo t4(vector_names.rbegin()->c_str());
        test_map[t3] = 5;
        test_map[t4] = 6;
        std::string name2 = name + "z";
        vector_names.emplace_back(name2);
        ov::DiscreteTypeInfo t5(vector_names.rbegin()->c_str());
        ov::DiscreteTypeInfo t6(vector_names.rbegin()->c_str());
        test_map[t5] = 7;
        test_map[t6] = 8;
    }

    for (const auto& it : test_map) {
        ASSERT_NE(test_map.end(), test_map.find(it.first));
    }

    ASSERT_NE(test_map.end(), test_map.find(a));
    ASSERT_NE(test_map.end(), test_map.find(b));
    ASSERT_NE(test_map.end(), test_map.find(c));
    ASSERT_EQ(test_map.end(), test_map.find(d));
    ASSERT_EQ(test_map.end(), test_map.find(f));
}
