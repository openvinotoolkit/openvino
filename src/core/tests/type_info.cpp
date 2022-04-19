// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/core/type.hpp"
#include "openvino/opsets/opset.hpp"
#include "openvino/util/common_util.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
TEST(type_info, compare_old_type) {
    ov::DiscreteTypeInfo type1("type1", static_cast<uint64_t>(0));
    ov::DiscreteTypeInfo type2("type2", static_cast<uint64_t>(0));
    ov::DiscreteTypeInfo type3("type1", 1ul);
    ov::DiscreteTypeInfo type4("type3", static_cast<uint64_t>(0), &type1);
    ov::DiscreteTypeInfo type5("type3", static_cast<uint64_t>(0), &type2);
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
    ov::DiscreteTypeInfo type1("type1", static_cast<uint64_t>(0), "version1");
    ov::DiscreteTypeInfo type1_o("type1", static_cast<uint64_t>(0));
    ASSERT_TRUE(type1 == type1_o);
}

TEST(type_info, check_hash_value) {
    const auto& hash_val = [](const char* name, const char* version_id, uint64_t version) -> size_t {
        size_t name_hash = name ? std::hash<std::string>()(std::string(name)) : 0;
        size_t version_hash = std::hash<decltype(version)>()(version);
        size_t version_id_hash = version_id ? std::hash<std::string>()(std::string(version_id)) : 0;
        // don't use parent for hash calculation, it is not a part of type (yet)
        return ov::util::hash_combine(std::vector<size_t>{name_hash, version_hash, version_id_hash});
    };
    ov::DiscreteTypeInfo type("type1", 0, "version1");
    ov::DiscreteTypeInfo type_old("type1", 1);
    ov::DiscreteTypeInfo type_with_version("type1", 1, "version1");
    ov::DiscreteTypeInfo type_empty_name("", static_cast<uint64_t>(0));
    ov::DiscreteTypeInfo type_empty_ver("type", static_cast<uint64_t>(0), "");
    EXPECT_EQ(hash_val(type.name, type.version_id, type.version), type.hash());
    EXPECT_EQ(hash_val(type_old.name, type_old.version_id, type_old.version), type_old.hash());
    EXPECT_EQ(hash_val(type_with_version.name, type_with_version.version_id, type_with_version.version),
              type_with_version.hash());
    EXPECT_EQ(hash_val(type_empty_name.name, type_empty_name.version_id, type_empty_name.version),
              type_empty_name.hash());
    EXPECT_EQ(hash_val(type_empty_ver.name, type_empty_ver.version_id, type_empty_ver.version), type_empty_ver.hash());
}

TEST(type_info, find_in_map) {
    std::vector<std::string> vector_names;
    ov::DiscreteTypeInfo a("Mod", 1ul, "opset1");
    ov::DiscreteTypeInfo b("Prelu", static_cast<uint64_t>(0), "opset1");
    ov::DiscreteTypeInfo c("Vector", static_cast<uint64_t>(0));
    ov::DiscreteTypeInfo d("Mod", 1ul, "opset3");
    ov::DiscreteTypeInfo f("Mod", 2ul);

    std::map<ov::DiscreteTypeInfo, int> test_map;
    test_map[a] = 1;
    test_map[b] = 1;
    test_map[c] = 1;

    const auto& opset = ov::get_opset8();
    // Reserve memory to avoid reallocation and copy of strings
    // because type info uses pointers from the original memory
    vector_names.reserve(opset.size() * 3);
    for (const auto& type : opset.get_types_info()) {
        test_map[type] = 2;
        std::string name = type.name;
        vector_names.emplace_back(name);
        ov::DiscreteTypeInfo t(vector_names.rbegin()->c_str(), 1000);
        ov::DiscreteTypeInfo t2(vector_names.rbegin()->c_str(), static_cast<uint64_t>(0));
        test_map[t] = 3;
        test_map[t2] = 4;
        std::string name1 = "a" + name;
        vector_names.emplace_back(name1);
        ov::DiscreteTypeInfo t3(vector_names.rbegin()->c_str(), 1000);
        ov::DiscreteTypeInfo t4(vector_names.rbegin()->c_str(), static_cast<uint64_t>(0));
        test_map[t3] = 5;
        test_map[t4] = 6;
        std::string name2 = name + "z";
        vector_names.emplace_back(name2);
        ov::DiscreteTypeInfo t5(vector_names.rbegin()->c_str(), 1000);
        ov::DiscreteTypeInfo t6(vector_names.rbegin()->c_str(), static_cast<uint64_t>(0));
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
