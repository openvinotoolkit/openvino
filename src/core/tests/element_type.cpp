// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/type/element_type.hpp"

#include <map>

#include "gtest/gtest.h"
#include "openvino/core/except.hpp"

using namespace ov;

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
    EXPECT_EQ(element::from<std::string>(), element::string);
}

TEST(element_type, from_string) {
    EXPECT_EQ(element::Type("boolean"), element::boolean);
    EXPECT_EQ(element::Type("BOOL"), element::boolean);

    EXPECT_EQ(element::Type("bf16"), element::bf16);
    EXPECT_EQ(element::Type("BF16"), element::bf16);
    EXPECT_EQ(element::Type("f16"), element::f16);
    EXPECT_EQ(element::Type("FP16"), element::f16);
    EXPECT_EQ(element::Type("f32"), element::f32);
    EXPECT_EQ(element::Type("FP32"), element::f32);
    EXPECT_EQ(element::Type("f64"), element::f64);
    EXPECT_EQ(element::Type("FP64"), element::f64);

    EXPECT_EQ(element::Type("i4"), element::i4);
    EXPECT_EQ(element::Type("I4"), element::i4);
    EXPECT_EQ(element::Type("i8"), element::i8);
    EXPECT_EQ(element::Type("I8"), element::i8);
    EXPECT_EQ(element::Type("i16"), element::i16);
    EXPECT_EQ(element::Type("I16"), element::i16);
    EXPECT_EQ(element::Type("i32"), element::i32);
    EXPECT_EQ(element::Type("I32"), element::i32);
    EXPECT_EQ(element::Type("i64"), element::i64);
    EXPECT_EQ(element::Type("I64"), element::i64);

    EXPECT_EQ(element::Type("bin"), element::u1);
    EXPECT_EQ(element::Type("BIN"), element::u1);
    EXPECT_EQ(element::Type("u1"), element::u1);
    EXPECT_EQ(element::Type("U1"), element::u1);
    EXPECT_EQ(element::Type("u4"), element::u4);
    EXPECT_EQ(element::Type("U4"), element::u4);
    EXPECT_EQ(element::Type("u8"), element::u8);
    EXPECT_EQ(element::Type("U8"), element::u8);
    EXPECT_EQ(element::Type("u16"), element::u16);
    EXPECT_EQ(element::Type("U16"), element::u16);
    EXPECT_EQ(element::Type("u32"), element::u32);
    EXPECT_EQ(element::Type("U32"), element::u32);
    EXPECT_EQ(element::Type("u64"), element::u64);
    EXPECT_EQ(element::Type("U64"), element::u64);
    EXPECT_EQ(element::Type("nf4"), element::nf4);
    EXPECT_EQ(element::Type("NF4"), element::nf4);
    EXPECT_EQ(element::Type("f8e4m3"), element::f8e4m3);
    EXPECT_EQ(element::Type("F8E4M3"), element::f8e4m3);
    EXPECT_EQ(element::Type("f8e5m2"), element::f8e5m2);
    EXPECT_EQ(element::Type("F8E5M2"), element::f8e5m2);
    EXPECT_EQ(element::Type("string"), element::string);
    EXPECT_EQ(element::Type("STRING"), element::string);
    EXPECT_EQ(element::Type("f4e2m1"), element::f4e2m1);
    EXPECT_EQ(element::Type("F4E2M1"), element::f4e2m1);
    EXPECT_EQ(element::Type("f8e8m0"), element::f8e8m0);
    EXPECT_EQ(element::Type("F8E8M0"), element::f8e8m0);
    OPENVINO_SUPPRESS_DEPRECATED_START
    EXPECT_EQ(element::Type("undefined"), element::undefined);
    EXPECT_EQ(element::Type("UNSPECIFIED"), element::undefined);
    OPENVINO_SUPPRESS_DEPRECATED_END
    EXPECT_EQ(element::Type("dynamic"), element::dynamic);

    EXPECT_THROW(element::Type("some_string"), ov::Exception);
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
