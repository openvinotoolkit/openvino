// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/type/element_type.hpp"

#include <array>
#include <map>

#include "common_test_utils/test_assertions.hpp"
#include "gtest/gtest.h"
#include "openvino/core/enum_names.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/memory_util.hpp"
#include "openvino/util/container_util.hpp"

using namespace ov;

TEST(element_type, gguf_fc_block_geometry) {
    struct BlockCase {
        element::Type type;
        const char* name;
        size_t elements;
        size_t bytes;
    };
    const std::array<BlockCase, 11> cases{{
        {element::gguf_q4_k, "gguf_q4_k", 256, 144},
        {element::gguf_q4_0, "gguf_q4_0", 32, 18},
        {element::gguf_q4_1, "gguf_q4_1", 32, 20},
        {element::gguf_q5_k, "gguf_q5_k", 256, 176},
        {element::gguf_q6_k, "gguf_q6_k", 256, 210},
        {element::gguf_q8_0, "gguf_q8_0", 32, 34},
        {element::gguf_q3_k, "gguf_q3_k", 256, 110},
        {element::gguf_iq2_xs, "gguf_iq2_xs", 256, 74},
        {element::gguf_iq2_s, "gguf_iq2_s", 256, 82},
        {element::gguf_iq3_xxs, "gguf_iq3_xxs", 256, 98},
        {element::gguf_iq3_s, "gguf_iq3_s", 256, 110},
    }};
    for (const auto& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        EXPECT_EQ(element::Type(test_case.name), test_case.type);
        EXPECT_EQ(test_case.type.get_type_name(), test_case.name);
        EXPECT_EQ(as_string(static_cast<element::Type_t>(test_case.type)), test_case.name);
        EXPECT_EQ(test_case.type.block_elem_count(), test_case.elements);
        EXPECT_EQ(test_case.type.block_byte_size(), test_case.bytes);
        EXPECT_TRUE(test_case.type.is_gguf_block());
        EXPECT_TRUE(test_case.type.is_quantized());
        EXPECT_FALSE(test_case.type.is_integral());
        EXPECT_EQ(test_case.type.size(), 1);
        EXPECT_EQ(util::get_memory_size(test_case.type, 0), 0);
        EXPECT_EQ(util::get_memory_size(test_case.type, test_case.elements), test_case.bytes);
        EXPECT_EQ(util::get_memory_size(test_case.type, test_case.elements + 1), 2 * test_case.bytes);
    }
    EXPECT_EQ(element::Type("GGUF_Q8_0"), element::gguf_q8_0);
}

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

constexpr auto element_type_cases = util::make_array(std::pair{"boolean", element::boolean},
                                                     std::pair{"BOOL", element::boolean},
                                                     std::pair{"bf16", element::bf16},
                                                     std::pair{"BF16", element::bf16},
                                                     std::pair{"f16", element::f16},
                                                     std::pair{"FP16", element::f16},
                                                     std::pair{"f32", element::f32},
                                                     std::pair{"FP32", element::f32},
                                                     std::pair{"f64", element::f64},
                                                     std::pair{"FP64", element::f64},
                                                     std::pair{"i4", element::i4},
                                                     std::pair{"I4", element::i4},
                                                     std::pair{"i8", element::i8},
                                                     std::pair{"I8", element::i8},
                                                     std::pair{"i16", element::i16},
                                                     std::pair{"I16", element::i16},
                                                     std::pair{"i32", element::i32},
                                                     std::pair{"I32", element::i32},
                                                     std::pair{"i64", element::i64},
                                                     std::pair{"I64", element::i64},
                                                     std::pair{"bin", element::u1},
                                                     std::pair{"BIN", element::u1},
                                                     std::pair{"u1", element::u1},
                                                     std::pair{"U1", element::u1},
                                                     std::pair{"u4", element::u4},
                                                     std::pair{"U4", element::u4},
                                                     std::pair{"u8", element::u8},
                                                     std::pair{"U8", element::u8},
                                                     std::pair{"u16", element::u16},
                                                     std::pair{"U16", element::u16},
                                                     std::pair{"u32", element::u32},
                                                     std::pair{"U32", element::u32},
                                                     std::pair{"u64", element::u64},
                                                     std::pair{"U64", element::u64},
                                                     std::pair{"nf4", element::nf4},
                                                     std::pair{"NF4", element::nf4},
                                                     std::pair{"f8e4m3", element::f8e4m3},
                                                     std::pair{"F8E4M3", element::f8e4m3},
                                                     std::pair{"f8e5m2", element::f8e5m2},
                                                     std::pair{"F8E5M2", element::f8e5m2},
                                                     std::pair{"string", element::string},
                                                     std::pair{"STRING", element::string},
                                                     std::pair{"f4e2m1", element::f4e2m1},
                                                     std::pair{"F4E2M1", element::f4e2m1},
                                                     std::pair{"f8e8m0", element::f8e8m0},
                                                     std::pair{"F8E8M0", element::f8e8m0},
                                                     std::pair{"gguf_q4_k", element::gguf_q4_k},
                                                     std::pair{"GGUF_Q4_K", element::gguf_q4_k},
                                                     std::pair{"dynamic", element::dynamic});

constexpr auto element_type_cases_invalid =
    util::make_array("some_string", "", "???", "12345", "not_a_type", "throw_exception");

TEST(element_type, from_string) {
    for (const auto& [str, expected] : element_type_cases) {
        EXPECT_EQ(element::Type(str), expected);
    }
}

TEST(element_type, from_string_invalid) {
    for (const auto& str : element_type_cases_invalid) {
        OV_EXPECT_THROW(std::ignore = element::Type(str), ov::Exception, testing::_);
    }
}

TEST(element_type, gguf_block_contract) {
    EXPECT_EQ(as_enum<element::Type_t>("gguf_q4_k"), element::Type_t::gguf_q4_k);
    EXPECT_EQ(as_string(element::Type_t::gguf_q4_k), "gguf_q4_k");
    EXPECT_TRUE(element::gguf_q4_k.is_static());
    EXPECT_TRUE(element::gguf_q4_k.is_quantized());
    EXPECT_FALSE(element::gguf_q4_k.is_real());
    EXPECT_FALSE(element::gguf_q4_k.is_integral());
    EXPECT_FALSE(element::gguf_q4_k.is_integral_number());
    EXPECT_EQ(static_cast<int>(element::Type_t::gguf_q4_k), static_cast<int>(element::Type_t::f8e8m0) + 1);
}

TEST(element_type, from_istringstream) {
    for (const auto& [str, expected] : element_type_cases) {
        std::istringstream ss(str);
        element::Type t;
        ss >> t;
        EXPECT_FALSE(ss.fail());
        EXPECT_EQ(t, expected);
    }
}

TEST(element_type, from_istringstream_invalid) {
    for (const auto& str : element_type_cases_invalid) {
        std::istringstream ss(str);
        element::Type t;
        EXPECT_THROW(ss >> t, ov::Exception);
    }
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
