// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <locale>
#include "range_iterator.hpp"
#include <cctype>

using namespace std;
using namespace InferenceEngine;

class RangeIteratorTests: public ::testing::Test {
 protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
    }

 public:

};

TEST_F(RangeIteratorTests, canCompareSameStringsInsensitive) {
    ASSERT_FALSE(std::lexicographical_compare(null_terminated_string("UPPer"),
                                             null_terminated_string_end(),
                                             null_terminated_string("upper"),
                                             null_terminated_string_end(), [](char a, char b) {
            std::locale loc;
            return std::tolower(a, loc) > std::tolower(b, loc);
        }));
}

TEST_F(RangeIteratorTests, canCompareNotSameStringsInsensitive) {
    ASSERT_TRUE(std::lexicographical_compare(null_terminated_string("UPPer"),
                                         null_terminated_string_end(),
                                         null_terminated_string("uppel"),
                                         null_terminated_string_end(), [](char a, char b) {
        std::locale loc;
        return std::tolower(a, loc) > std::tolower(b, loc);
    }));
    
}

TEST_F(RangeIteratorTests, cannotDereferenceEndIterator) {
    ASSERT_ANY_THROW(*null_terminated_string_end());
    ASSERT_ANY_THROW(++null_terminated_string_end());
    ASSERT_ANY_THROW(null_terminated_string_end()++);
}
