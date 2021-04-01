// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <vpu/utils/attributes_map.hpp>
#include <vpu/model/model.hpp>

using namespace testing;

TEST(VPU_AnyTest, SimpleUseCases) {
    vpu::Any any;
    ASSERT_TRUE(any.empty()) << "Empty after default constructor";

    ASSERT_NO_THROW(any.set<int>(10)) << "Can set basic type";
    ASSERT_EQ(10, any.get<int>())  << "Can get basic type";

    int val = 20;
    ASSERT_NO_THROW(any.set(&val)) << "Can set pointer";
    ASSERT_EQ(&val, any.get<int*>())  << "Can get pointer";

    std::vector<int> vec1{1, 2, 3};

    ASSERT_NO_THROW(any.set(vec1)) << "Can set C++ class by l-value";
    ASSERT_FALSE(vec1.empty()) << "Original value is not touched";
    ASSERT_EQ(vec1, any.get<std::vector<int>>()) << "Can get C++ class";

    any.get<std::vector<int>>().clear();
    ASSERT_TRUE(any.get<std::vector<int>>().empty()) << "get returns l-value";

    std::vector<int> vec2{4, 5, 6};
    std::vector<int> vec2_copy = vec2;
    ASSERT_NO_THROW(any.set(std::move(vec2))) << "Can set C++ class by r-value";
    ASSERT_TRUE(vec2.empty()) << "Original value was moved";
    ASSERT_EQ(vec2_copy, any.get<std::vector<int>>()) << "Any holds correct value";
}

TEST(VPU_AttributesMapTest, SimpleUseCases) {
    vpu::AttributesMap attrs;
    ASSERT_TRUE(attrs.empty()) << "Fresh map must be empty";

    ASSERT_NO_THROW(attrs.set<int>("int", 1)) << "Can set basic type";
    ASSERT_EQ(1, attrs.get<int>("int")) << "Can get basic type";

    int val = 20;
    ASSERT_NO_THROW(attrs.set("ptr", &val)) << "Can set pointer";
    ASSERT_EQ(&val, attrs.get<int*>("ptr"))  << "Can get pointer";

    ASSERT_NO_THROW(attrs.set<std::string>("string", "string")) << "Can set C++ class";
    ASSERT_EQ("string", attrs.get<std::string>("string")) << "Can get C++ class";

    ASSERT_EQ(-10L, attrs.getOrSet<long>("long", -10L)) << "Can getOrSet basic type";
    ASSERT_EQ(-10L, attrs.get<long>("long")) << "get after getOrSet";
    ASSERT_NO_THROW(attrs.set<long>("long", 10L)) << "Can set basic type";
    ASSERT_EQ(10L, attrs.getOrSet<long>("long", -10L)) << "getOrSet after set";

    ASSERT_TRUE(attrs.has("string")) << "has for exist key";
    ASSERT_FALSE(attrs.has("string2")) << "has for non-exist key";
    ASSERT_EQ("string2", attrs.getOrDefault<std::string>("string2", "string2")) << "getOrDefault for non-exist key";
    ASSERT_EQ("string", attrs.getOrDefault<std::string>("string", "string2")) << "getOrDefault for exist key";
    ASSERT_FALSE(attrs.has("string2")) << "getOrDefault doesn't insert new key";

    attrs.erase("int");
    attrs.erase("ptr");
    attrs.erase("string");
    attrs.erase("long");

    ASSERT_TRUE(attrs.empty()) << "Map must be empty again";
}
