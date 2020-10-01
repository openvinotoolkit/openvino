// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <unordered_map>
#include "caseless.hpp"
#include "debug.h"

using namespace std;
using namespace InferenceEngine::details;

using CaselessTests = ::testing::Test;

TEST_F(CaselessTests, smoke_emptyAreEqual) {
    ASSERT_TRUE(InferenceEngine::details::equal("", ""));
}

TEST_F(CaselessTests, smoke_canIgnoreCase) {
    ASSERT_TRUE(InferenceEngine::details::equal("abc", "ABC"));
}

TEST_F(CaselessTests, smoke_emptyIsNotEqualNotEmpty) {
    ASSERT_FALSE(InferenceEngine::details::equal("", "abc"));
}

TEST_F(CaselessTests, smoke_canFindCaslessInMap) {
    caseless_map<string, int> storage = {
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

TEST_F(CaselessTests, smoke_canFindCaslessInUnordered) {
    caseless_unordered_map <string, int> storage = {
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
