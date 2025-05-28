// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/util/common_util.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/env_util.hpp"
#include "openvino/util/monitors/device_monitor.hpp"

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

TEST(UtilsTests, device_monitor) {
    std::string cpu_device_id = "";
    std::map<std::string, float> utilization;
    ASSERT_NO_THROW(utilization = get_device_utilization(cpu_device_id));
#ifdef _WIN32
    ASSERT_FALSE(utilization.empty() && utilization.count("Total") && utilization.at("Total") >= 0.0f)
        << "Expected non-empty utilization map for CPU device";
#else
    bool ret = utilization == std::map<std::string, float>{{"Total", -1.0f}};
    ASSERT_TRUE(ret) << "Expected utilization map with 'Total' key only for CPU device";
#endif

    std::string invalid_device_id = "INVALID_DEVICE_ID";
#ifdef _WIN32
    ASSERT_THROW(utilization = get_device_utilization(invalid_device_id), std::runtime_error)
        << "Expected exception for invalid device ID";
#else
    // On non-Windows platforms, we expect no exception and an empty utilization map
    ASSERT_NO_THROW(utilization = get_device_utilization(invalid_device_id))
        << "Expected no exception for invalid device ID on non-Windows platforms";
    ASSERT_TRUE(utilization.empty()) << "Expected empty utilization map for invalid device ID";
#endif
}