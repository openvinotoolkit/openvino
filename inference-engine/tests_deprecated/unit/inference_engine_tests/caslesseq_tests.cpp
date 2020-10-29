// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <unordered_map>
#include "details/caseless.hpp"
#include "debug.h"

using namespace std;
using namespace InferenceEngine::details;

class CaselessTests : public ::testing::Test {
 protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
    }

 public:

};

TEST_F(CaselessTests, emptyAreEqual) {
    ASSERT_TRUE(InferenceEngine::details::equal("", ""));
}

TEST_F(CaselessTests, canIgnoreCase) {
    ASSERT_TRUE(InferenceEngine::details::equal("abc", "ABC"));
}

TEST_F(CaselessTests, emptyIsNotEqualNotEmpty) {
    ASSERT_FALSE(InferenceEngine::details::equal("", "abc"));
}

TEST_F(CaselessTests, canFindCaslessInMap) {
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

TEST_F(CaselessTests, canFindCaslessInUnordered) {

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
