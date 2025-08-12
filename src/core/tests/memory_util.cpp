// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/memory_util.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/data_utils.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_iterator.hpp"

namespace ov::test {

TEST(get_memory_size_overflow, safe_memory_size_i8) {
    constexpr auto max = std::numeric_limits<size_t>::max();

    EXPECT_EQ(ov::util::get_memory_size_overflow(ov::element::i8, max), max);
    // EXPECT_EQ(ov::util::get_memory_size_overflow(ov::element::i8, ov::Shape{max, 2}), std::nullopt);
}

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
    bit_type_precisoin,
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
    nibble_type_precisoin,
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
    byte_type_precisoin,
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

    EXPECT_EQ(ov::util::get_memory_size_overflow(type, shape), exp_size);
}

TEST(GetMemorySizeOverflowTest, zero_number_of_elements) {
    EXPECT_EQ(ov::util::get_memory_size_overflow(element::f4e2m1, 0), std::optional<size_t>(0));
    EXPECT_EQ(ov::util::get_memory_size_overflow(element::i16, 0), std::optional<size_t>(0));
    EXPECT_EQ(ov::util::get_memory_size_overflow(element::string, 0), std::optional<size_t>(0));
}
}  // namespace ov::test
