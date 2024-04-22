// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/util/common_util.hpp"
#include "openvino/util/file_util.hpp"

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
