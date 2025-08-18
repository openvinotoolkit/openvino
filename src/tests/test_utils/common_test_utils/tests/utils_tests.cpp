// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/util/common_util.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/env_util.hpp"

using namespace testing;
using namespace ov::util;

TEST(UtilsTests, get_directory_returns_root) {
    ASSERT_EQ(get_directory("/test"), "/");
}

TEST(UtilsTests, get_directory_returns_empty) {
    ASSERT_EQ(get_directory(""), "");
}

TEST(UtilsTests, get_directory_current_dir) {
    ASSERT_EQ(get_directory("my_file.txt"), ".");
}

TEST(UtilsTests, get_directory_return_file_dir) {
    ASSERT_EQ(get_directory("../test/path/my_file.txt"), "../test/path");
}

TEST(UtilsTests, filter_lines_by_prefix) {
    auto lines = "abc\nkkb\nabpp\n";
    auto res = filter_lines_by_prefix(lines, "ab");
    ASSERT_EQ(res, "abc\nabpp\n");

    lines = "abc\nkkb\nkkabpp\n";
    res = filter_lines_by_prefix(lines, "ab");
    ASSERT_EQ(res, "abc\n");

    lines = "abc\nkkb\nabpp\n";
    res = filter_lines_by_prefix(lines, "k");
    ASSERT_EQ(res, "kkb\n");

    lines = "abc\nkkb\nabpp\n";
    res = filter_lines_by_prefix(lines, "lr");
    ASSERT_EQ(res, "");

    lines = "";
    res = filter_lines_by_prefix(lines, "ab");
    ASSERT_EQ(res, "");

    lines = "\n\n\n";
    res = filter_lines_by_prefix(lines, "ab");
    ASSERT_EQ(res, "");
}

TEST(UtilsTests, split_by_delimiter) {
    const auto line = "EliminateSplitConcat,MarkDequantization,StatefulSDPAFusion";

    const auto split_set = ov::util::split_by_delimiter(line, ',');

    const std::unordered_set<std::string> expected_set = {"EliminateSplitConcat", "MarkDequantization", "StatefulSDPAFusion"};

    ASSERT_EQ(split_set, expected_set);
}

TEST(UtilsTests, split_by_delimiter_single) {
    const auto line = "EliminateSplitConcat";

    const auto split_set = ov::util::split_by_delimiter(line, ',');

    const std::unordered_set<std::string> expected_set = {"EliminateSplitConcat"};

    ASSERT_EQ(split_set, expected_set);
}

TEST(UtilsTests, split_by_delimiter_single_with_comma) {
    const auto line = "EliminateSplitConcat,";

    const auto split_set = ov::util::split_by_delimiter(line, ',');

    const std::unordered_set<std::string> expected_set = {"EliminateSplitConcat"};

    ASSERT_EQ(split_set, expected_set);
}

TEST(UtilsTests, split_by_delimiter_empty) {
    const auto line = "";

    const auto split_set = ov::util::split_by_delimiter(line, ',');

    const std::unordered_set<std::string> expected_set = {};

    ASSERT_EQ(split_set, expected_set);
}

TEST(UtilsTests, split_by_delimiter_empty_and_comma) {
    const auto line = ",";

    const auto split_set = ov::util::split_by_delimiter(line, ',');

    const std::unordered_set<std::string> expected_set = {"", ""};

    ASSERT_EQ(split_set, expected_set);
}

TEST(UtilsTests, mul_overflow_i8_detected) {
    int8_t result;
    constexpr auto max = std::numeric_limits<int8_t>::max();
    constexpr auto min = std::numeric_limits<int8_t>::min();

    EXPECT_TRUE(mul_overflow<int8_t>(2, max, result));
    EXPECT_TRUE(mul_overflow<int8_t>(max, 3, result));
    EXPECT_TRUE(mul_overflow<int8_t>(5, 50, result));
    EXPECT_TRUE(mul_overflow<int8_t>(51, 4, result));

    EXPECT_TRUE(mul_overflow<int8_t>(max, min, result));
    EXPECT_TRUE(mul_overflow<int8_t>(max, -2, result));
    EXPECT_TRUE(mul_overflow<int8_t>(31, -5, result));
    EXPECT_TRUE(mul_overflow<int8_t>(2, min, result));

    EXPECT_TRUE(mul_overflow<int8_t>(min, max, result));
    EXPECT_TRUE(mul_overflow<int8_t>(min, 2, result));
    EXPECT_TRUE(mul_overflow<int8_t>(-2, max, result));
    EXPECT_TRUE(mul_overflow<int8_t>(-13, 12, result));

    EXPECT_TRUE(mul_overflow<int8_t>(min, -1, result));
    EXPECT_TRUE(mul_overflow<int8_t>(-1, min, result));
    EXPECT_TRUE(mul_overflow<int8_t>(-2, -64, result));
    EXPECT_TRUE(mul_overflow<int8_t>(-64, -2, result));
    EXPECT_TRUE(mul_overflow<int8_t>(-43, -3, result));
    EXPECT_TRUE(mul_overflow<int8_t>(-4, -33, result));
}

TEST(UtilsTests, mul_overflow_i8_zero_value) {
    int8_t result;
    constexpr auto max = std::numeric_limits<int8_t>::max();
    constexpr auto min = std::numeric_limits<int8_t>::min();

    ASSERT_FALSE(mul_overflow<int8_t>(0, 0, result));
    EXPECT_EQ(result, 0);

    ASSERT_FALSE(mul_overflow<int8_t>(0, max, result));
    EXPECT_EQ(result, 0);

    ASSERT_FALSE(mul_overflow<int8_t>(max, 0, result));
    EXPECT_EQ(result, 0);

    ASSERT_FALSE(mul_overflow<int8_t>(0, min, result));
    EXPECT_EQ(result, 0);

    ASSERT_FALSE(mul_overflow<int8_t>(min, 0, result));
    EXPECT_EQ(result, 0);
}

TEST(UtilsTests, mul_overflow_i8_non_zero_value) {
    int8_t result;
    constexpr auto max = std::numeric_limits<int8_t>::max();
    constexpr auto min = std::numeric_limits<int8_t>::min();

    ASSERT_FALSE(mul_overflow<int8_t>(-2, 5, result));
    EXPECT_EQ(result, -10);

    ASSERT_FALSE(mul_overflow<int8_t>(-5, 3, result));
    EXPECT_EQ(result, -15);

    ASSERT_FALSE(mul_overflow<int8_t>(max, 1, result));
    EXPECT_EQ(result, max);

    ASSERT_FALSE(mul_overflow<int8_t>(max, -1, result));
    EXPECT_EQ(result, -max);

    ASSERT_FALSE(mul_overflow<int8_t>(1, min, result));
    EXPECT_EQ(result, min);

    ASSERT_FALSE(mul_overflow<int8_t>(-6, -10, result));
    EXPECT_EQ(result, 60);
}

TEST(UtilsTests, mul_overflow_u8_detected) {
    uint8_t result;
    constexpr auto max = std::numeric_limits<uint8_t>::max();

    EXPECT_TRUE(mul_overflow<uint8_t>(2, max, result));
    EXPECT_TRUE(mul_overflow<uint8_t>(max, 3, result));
    EXPECT_TRUE(mul_overflow<uint8_t>(5, 56, result));
    EXPECT_TRUE(mul_overflow<uint8_t>(66, 4, result));
}

TEST(UtilsTests, mul_overflow_u8_zero_value) {
    uint8_t result;
    constexpr auto max = std::numeric_limits<uint8_t>::max();

    ASSERT_FALSE(mul_overflow<uint8_t>(0, 0, result));
    EXPECT_EQ(result, 0);

    ASSERT_FALSE(mul_overflow<uint8_t>(0, max, result));
    EXPECT_EQ(result, 0);

    ASSERT_FALSE(mul_overflow<uint8_t>(max, 0, result));
    EXPECT_EQ(result, 0);
}

TEST(UtilsTests, mul_overflow_u8_non_zero_value) {
    uint8_t result;
    constexpr auto max = std::numeric_limits<uint8_t>::max();

    ASSERT_FALSE(mul_overflow<uint8_t>(2, 5, result));
    EXPECT_EQ(result, 10);

    ASSERT_FALSE(mul_overflow<uint8_t>(15, 3, result));
    EXPECT_EQ(result, 45);

    ASSERT_FALSE(mul_overflow<uint8_t>(max, 1, result));
    EXPECT_EQ(result, max);
}
