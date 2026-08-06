// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <limits>
#include <optional>
#include <string>

#include "../src/parse_output_index.hpp"

using ov::frontend::tensorflow::parse_output_index;

struct ParseOutputIndexCase {
    std::string input;
    std::optional<int> expected;
};

class ParseOutputIndexTest : public ::testing::TestWithParam<ParseOutputIndexCase> {};

TEST_P(ParseOutputIndexTest, ParsesValidAndRejectsInvalid) {
    const auto& c = GetParam();
    EXPECT_EQ(parse_output_index(c.input), c.expected) << "input=\"" << c.input << "\"";
}

INSTANTIATE_TEST_SUITE_P(Valid,
                         ParseOutputIndexTest,
                         ::testing::Values(ParseOutputIndexCase{"0", 0},
                                           ParseOutputIndexCase{"42", 42},
                                           ParseOutputIndexCase{"-7", -7},
                                           ParseOutputIndexCase{"2147483647", std::numeric_limits<int>::max()},
                                           ParseOutputIndexCase{"-2147483648", std::numeric_limits<int>::min()}));

INSTANTIATE_TEST_SUITE_P(Invalid,
                         ParseOutputIndexTest,
                         ::testing::Values(ParseOutputIndexCase{"", std::nullopt},
                                           ParseOutputIndexCase{"abc", std::nullopt},
                                           ParseOutputIndexCase{"1junk", std::nullopt},
                                           ParseOutputIndexCase{" 42", std::nullopt},
                                           ParseOutputIndexCase{"99999999999999999999", std::nullopt},
                                           ParseOutputIndexCase{"2147483648", std::nullopt},
                                           ParseOutputIndexCase{"-2147483649", std::nullopt}));
