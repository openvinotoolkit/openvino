// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/memory_util.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/data_utils.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_iterator.hpp"

namespace ov::test {

using MemsizeOverflowParam = std::tuple<element::Type, Shape, std::optional<size_t>>;

class GetMemorySizeOverflowTest : public testing::TestWithParam<MemsizeOverflowParam> {
public:
    static std::string get_test_name(const testing::TestParamInfo<MemsizeOverflowParam>& obj) {
        const auto& [type, shape, exp_size] = obj.param;
        std::ostringstream result;
        result << "type=" << type << "_shape=" << shape
               << "_exp_size=" << (exp_size ? std::to_string(*exp_size) : "overflow");
        return result.str();
    }
};

constexpr auto max_dim = std::numeric_limits<size_t>::max();

INSTANTIATE_TEST_SUITE_P(
    bit_type_precision,
    GetMemorySizeOverflowTest,
    testing::Values(std::make_tuple(element::u1, Shape{}, std::optional<size_t>(1)),
                    std::make_tuple(element::u1, Shape{8}, std::optional<size_t>(1)),
                    std::make_tuple(element::u1, Shape{9}, std::optional<size_t>(2)),
                    std::make_tuple(element::u1, Shape{3, 3}, std::optional<size_t>(2)),
                    std::make_tuple(element::u1, Shape{max_dim}, std::optional<size_t>(max_dim / 8 + 1)),
                    std::make_tuple(element::u1, Shape{max_dim - 1}, std::optional<size_t>(max_dim / 8 + 1)),
                    std::make_tuple(element::u1, Shape{2, max_dim}, std::nullopt),
                    std::make_tuple(element::u2, Shape{}, std::optional<size_t>(1)),
                    std::make_tuple(element::u2, Shape{4}, std::optional<size_t>(1)),
                    std::make_tuple(element::u2, Shape{5}, std::optional<size_t>(2)),
                    std::make_tuple(element::u2, Shape{3, 3}, std::optional<size_t>(3)),
                    std::make_tuple(element::u2, Shape{max_dim}, std::optional<size_t>(max_dim / 4 + 1)),
                    std::make_tuple(element::u2, Shape{max_dim - 1}, std::optional<size_t>(max_dim / 4 + 1)),
                    std::make_tuple(element::u2, Shape{max_dim - 3}, std::optional<size_t>(max_dim / 4)),
                    std::make_tuple(element::u2, Shape{1, 2, max_dim}, std::nullopt)),
    GetMemorySizeOverflowTest::get_test_name);

INSTANTIATE_TEST_SUITE_P(
    nibble_type_precision,
    GetMemorySizeOverflowTest,
    testing::Values(std::make_tuple(element::u4, Shape{}, std::optional<size_t>(1)),
                    std::make_tuple(element::u4, Shape{2}, std::optional<size_t>(1)),
                    std::make_tuple(element::u4, Shape{3}, std::optional<size_t>(2)),
                    std::make_tuple(element::u4, Shape{2, 2}, std::optional<size_t>(2)),
                    std::make_tuple(element::u4, Shape{3, 3, 2}, std::optional<size_t>(9)),
                    std::make_tuple(element::u4, Shape{max_dim}, std::optional<size_t>(max_dim / 2 + 1)),
                    std::make_tuple(element::u4, Shape{max_dim - 1}, std::optional<size_t>(max_dim / 2)),
                    std::make_tuple(element::u4, Shape{2, max_dim}, std::nullopt)),
    GetMemorySizeOverflowTest::get_test_name);

INSTANTIATE_TEST_SUITE_P(
    byte_type_precision,
    GetMemorySizeOverflowTest,
    testing::Values(std::make_tuple(element::u8, Shape{}, std::optional<size_t>(1)),
                    std::make_tuple(element::u8, Shape{8}, std::optional<size_t>(8)),
                    std::make_tuple(element::u8, Shape{9}, std::optional<size_t>(9)),
                    std::make_tuple(element::u8, Shape{3, 3}, std::optional<size_t>(9)),
                    std::make_tuple(element::u8, Shape{max_dim, 3}, std::nullopt),
                    std::make_tuple(element::f64, Shape{}, std::optional<size_t>(8)),
                    std::make_tuple(element::f64, Shape{12}, std::optional<size_t>(96)),
                    std::make_tuple(element::f64, Shape{max_dim / 8}, std::optional<size_t>(max_dim - 7)),
                    std::make_tuple(element::f64, Shape{2, max_dim / 16}, std::optional<size_t>(max_dim - 15)),
                    std::make_tuple(element::f64, Shape{2, max_dim / 16 - 1}, std::optional<size_t>(max_dim - 31)),
                    std::make_tuple(element::f64, Shape{1, max_dim / 8 - 1, 3}, std::nullopt),
                    std::make_tuple(element::string, Shape{}, std::optional<size_t>(sizeof(std::string))),
                    std::make_tuple(element::string, Shape{12}, std::optional<size_t>(12 * sizeof(std::string))),
                    std::make_tuple(element::string, Shape{max_dim}, std::nullopt)),
    GetMemorySizeOverflowTest::get_test_name);

TEST_P(GetMemorySizeOverflowTest, calculate_from_shape) {
    const auto& [type, shape, exp_size] = GetParam();

    EXPECT_EQ(ov::util::get_memory_size_safe(type, shape), exp_size);
}

TEST(GetMemorySizeOverflowTest, zero_number_of_elements) {
    EXPECT_EQ(ov::util::get_memory_size_safe(element::f4e2m1, 0), std::optional<size_t>(0));
    EXPECT_EQ(ov::util::get_memory_size_safe(element::i16, 0), std::optional<size_t>(0));
    EXPECT_EQ(ov::util::get_memory_size_safe(element::string, 0), std::optional<size_t>(0));
}

// element type, memory size, expected max number of elements
using MaxElementsForMemorySizeParam = std::tuple<element::Type, size_t, size_t>;

class GetMaxElementsForMemorySizeTest : public testing::TestWithParam<MaxElementsForMemorySizeParam> {
public:
    static std::string get_test_name(const testing::TestParamInfo<MaxElementsForMemorySizeParam>& obj) {
        const auto& [type, memory_size, exp_elements] = obj.param;
        std::ostringstream result;
        result << "type=" << type << "_memory_size=" << memory_size << "_exp_elements=" << exp_elements;
        return result.str();
    }
};

TEST_P(GetMaxElementsForMemorySizeTest, calculate_max_elements) {
    const auto& [type, memory_size, exp_elements] = GetParam();
    EXPECT_EQ(ov::util::get_elements_number(type, memory_size), exp_elements);
}

INSTANTIATE_TEST_SUITE_P(bit_type_precision,
                         GetMaxElementsForMemorySizeTest,
                         testing::Values(std::make_tuple(element::u1, 0, 0),
                                         std::make_tuple(element::u1, 1, 8),
                                         std::make_tuple(element::u1, 2, 16),
                                         std::make_tuple(element::u1, 3, 24),
                                         std::make_tuple(element::u1, 4, 32),
                                         std::make_tuple(element::u2, 0, 0),
                                         std::make_tuple(element::u2, 1, 4),
                                         std::make_tuple(element::u2, 2, 8),
                                         std::make_tuple(element::u2, 3, 12),
                                         std::make_tuple(element::u2, 4, 16)),
                         GetMaxElementsForMemorySizeTest::get_test_name);

INSTANTIATE_TEST_SUITE_P(nibble_type_precision,
                         GetMaxElementsForMemorySizeTest,
                         testing::Values(std::make_tuple(element::i4, 0, 0),
                                         std::make_tuple(element::u4, 0, 0),
                                         std::make_tuple(element::u4, 1, 2),
                                         std::make_tuple(element::u4, 2, 4),
                                         std::make_tuple(element::i4, 3, 6),
                                         std::make_tuple(element::i4, 4, 8)),
                         GetMaxElementsForMemorySizeTest::get_test_name);

INSTANTIATE_TEST_SUITE_P(split_bit_type_precision,
                         GetMaxElementsForMemorySizeTest,
                         testing::Values(std::make_tuple(element::u3, 0, 0),
                                         std::make_tuple(element::u3, 1, 0),
                                         std::make_tuple(element::u3, 2, 0),
                                         std::make_tuple(element::u3, 3, 8),
                                         std::make_tuple(element::u3, 5, 8),
                                         std::make_tuple(element::u3, 6, 16),
                                         std::make_tuple(element::u3, 11, 24),
                                         std::make_tuple(element::u3, 12, 32),
                                         std::make_tuple(element::u6, 0, 0),
                                         std::make_tuple(element::u6, 1, 0),
                                         std::make_tuple(element::u6, 2, 0),
                                         std::make_tuple(element::u6, 3, 4),
                                         std::make_tuple(element::u6, 4, 4),
                                         std::make_tuple(element::u6, 5, 4),
                                         std::make_tuple(element::u6, 6, 8),
                                         std::make_tuple(element::u6, 11, 12),
                                         std::make_tuple(element::u6, 12, 16)),
                         GetMaxElementsForMemorySizeTest::get_test_name);

INSTANTIATE_TEST_SUITE_P(byte_type_precision,
                         GetMaxElementsForMemorySizeTest,
                         testing::Values(std::make_tuple(element::u8, 0, 0),
                                         std::make_tuple(element::u8, 1, 1),
                                         std::make_tuple(element::u8, 2, 2),
                                         std::make_tuple(element::u8, 3, 3),
                                         std::make_tuple(element::u8, 4, 4),
                                         std::make_tuple(element::i16, 0, 0),
                                         std::make_tuple(element::i16, 1, 0),
                                         std::make_tuple(element::i16, 2, 1),
                                         std::make_tuple(element::i16, 3, 1),
                                         std::make_tuple(element::i16, 4, 2),
                                         std::make_tuple(element::f32, 0, 0),
                                         std::make_tuple(element::f32, 1, 0),
                                         std::make_tuple(element::f32, 4, 1),
                                         std::make_tuple(element::f32, 7, 1),
                                         std::make_tuple(element::f32, 8, 2),
                                         std::make_tuple(element::f32, 15, 3),
                                         std::make_tuple(element::f32, 16, 4),
                                         std::make_tuple(element::f64, 0, 0),
                                         std::make_tuple(element::f64, 1, 0),
                                         std::make_tuple(element::f64, 7, 0),
                                         std::make_tuple(element::f64, 8, 1),
                                         std::make_tuple(element::f64, 15, 1),
                                         std::make_tuple(element::f64, 16, 2),
                                         std::make_tuple(element::f64, 24, 3),
                                         std::make_tuple(element::f64, 32, 4),
                                         std::make_tuple(element::f64, 33, 4)),
                         GetMaxElementsForMemorySizeTest::get_test_name);

INSTANTIATE_TEST_SUITE_P(string_type,
                         GetMaxElementsForMemorySizeTest,
                         testing::Values(std::make_tuple(element::string, 0, 0),
                                         std::make_tuple(element::string, sizeof(std::string), 1),
                                         std::make_tuple(element::string, 2 * sizeof(std::string), 2),
                                         std::make_tuple(element::string, 3 * sizeof(std::string), 3),
                                         std::make_tuple(element::string, 4 * sizeof(std::string), 4)),
                         GetMaxElementsForMemorySizeTest::get_test_name);

using AlignTestParam = std::tuple<uintptr_t, size_t, size_t>;

class AlignTest : public ::testing::TestWithParam<AlignTestParam> {};

INSTANTIATE_TEST_SUITE_P(AlignTestSuite,
                         AlignTest,
                         testing::Values(AlignTestParam{0, 0, 0},
                                         AlignTestParam{20, 0, 0},
                                         AlignTestParam{0, 64, 0},
                                         AlignTestParam{20, 64, 44},
                                         AlignTestParam{64, 64, 0},
                                         AlignTestParam{65, 64, 63},
                                         AlignTestParam{128, 64, 0},
                                         AlignTestParam{130, 64, 62},
                                         AlignTestParam{0, 100, 0},
                                         AlignTestParam{63, 100, 37},
                                         AlignTestParam{130, 100, 70}),
                         testing::PrintToStringParamName());

TEST_P(AlignTest, align_padding_size) {
    const auto& [pos, aligment, expected] = GetParam();

    const auto pad = ov::util::align_padding_size(aligment, pos);

    EXPECT_EQ(pad, expected);
}
}  // namespace ov::test
