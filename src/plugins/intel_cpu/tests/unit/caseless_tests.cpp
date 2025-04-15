// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/caseless.hpp"

#include <string>
#include <gtest/gtest.h>

using namespace ov::intel_cpu;
using CaselessTests = ::testing::Test;

TEST_F(CaselessTests, canFindCaslessInUnordered) {
    caseless_unordered_map<std::string, int> storage = {
        {"Abc", 1},
        {"bC", 2},
        {"AbcD", 3},
    };
    ASSERT_EQ(storage["abc"], 1);
    ASSERT_EQ(storage["ABC"], 1);
    ASSERT_EQ(storage["BC"], 2);
    ASSERT_EQ(storage["aBCd"], 3);
    ASSERT_EQ(storage.find("aBd"), storage.end());
    ASSERT_EQ(storage.find(""), storage.end());
}
