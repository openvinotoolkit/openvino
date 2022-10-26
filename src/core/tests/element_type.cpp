// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/type/element_type.hpp"

#include <map>

#include "gtest/gtest.h"

using namespace ngraph;

TEST(element_type, from) {
    EXPECT_EQ(element::from<char>(), element::boolean);
    EXPECT_EQ(element::from<bool>(), element::boolean);
    EXPECT_EQ(element::from<float>(), element::f32);
    EXPECT_EQ(element::from<double>(), element::f64);
    EXPECT_EQ(element::from<int8_t>(), element::i8);
    EXPECT_EQ(element::from<int16_t>(), element::i16);
    EXPECT_EQ(element::from<int32_t>(), element::i32);
    EXPECT_EQ(element::from<int64_t>(), element::i64);
    EXPECT_EQ(element::from<uint8_t>(), element::u8);
    EXPECT_EQ(element::from<uint16_t>(), element::u16);
    EXPECT_EQ(element::from<uint32_t>(), element::u32);
    EXPECT_EQ(element::from<uint64_t>(), element::u64);
}

TEST(element_type, mapable) {
    std::map<element::Type, std::string> test_map;

    test_map.insert({element::f32, "float"});
}

TEST(element_type, merge_both_dynamic) {
    element::Type t;
    ASSERT_TRUE(element::Type::merge(t, element::dynamic, element::dynamic));
    ASSERT_TRUE(t.is_dynamic());
}

TEST(element_type, merge_left_dynamic) {
    element::Type t;
    ASSERT_TRUE(element::Type::merge(t, element::dynamic, element::u64));
    ASSERT_TRUE(t.is_static());
    ASSERT_EQ(t, element::u64);
}

TEST(element_type, merge_right_dynamic) {
    element::Type t;
    ASSERT_TRUE(element::Type::merge(t, element::i16, element::dynamic));
    ASSERT_TRUE(t.is_static());
    ASSERT_EQ(t, element::i16);
}

TEST(element_type, merge_both_static_equal) {
    element::Type t;
    ASSERT_TRUE(element::Type::merge(t, element::f64, element::f64));
    ASSERT_TRUE(t.is_static());
    ASSERT_EQ(t, element::f64);
}

TEST(element_type, merge_both_static_unequal) {
    element::Type t = element::f32;
    ASSERT_FALSE(element::Type::merge(t, element::i8, element::i16));
    ASSERT_TRUE(t.is_static());
    ASSERT_EQ(t, element::f32);
}

struct ObjectWithType {
    ObjectWithType(const element::Type& type_) : type{type_} {}
    ~ObjectWithType() {
        EXPECT_NO_THROW(type.bitwidth()) << "Could not access type information in global scope";
    }
    element::Type type;
};

using ObjectWithTypeParams = std::tuple<ObjectWithType>;

class ObjectWithTypeTests : public testing::WithParamInterface<ObjectWithTypeParams>, public ::testing::Test {};

TEST_P(ObjectWithTypeTests, construct_and_destroy_in_global_scope) {
    ASSERT_EQ(element::f32, std::get<0>(GetParam()).type);
}

INSTANTIATE_TEST_SUITE_P(f32_test_parameter, ObjectWithTypeTests, ::testing::Values(element::f32));