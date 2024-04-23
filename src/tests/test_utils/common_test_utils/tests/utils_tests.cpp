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

class FilterLinesByPrefixTests : public testing::WithParamInterface<std::tuple<std::string, std::string, std::string>> {
protected:
    std::string str_origin;
    std::string prefix;
    std::string str_expected;
};

TEST_P(FilterLinesByPrefixTests, ChecksStringFilter) {
    std::string str_origin = std::get<0>(GetParam());
    std::string prefix = std::get<1>(GetParam());
    std::string str_expected = std::get<2>(GetParam());

    ASSERT_EQ(filter_lines_by_prefix(str_origin, prefix), str_expected);
}

INSTANTIATE_TEST_CASE_P(
        UtilsTests,
        FilterLinesByPrefixTests,
        ::testing::Values(
                std::make_tuple("abc\nkkb\nabpp\n", "ab", "abc\nabpp\n"),
                std::make_tuple("abc\nkkb\nkkabpp\n", "ab", "abc\n"),
                std::make_tuple("abc\nkkb\nabpp\n", "k", "kkb\n"),
                std::make_tuple("abc\nkkb\nabpp\n", "lr", ""),
                std::make_tuple("", "ab", ""),
                std::make_tuple("\n\n\n", "ab", "")));

