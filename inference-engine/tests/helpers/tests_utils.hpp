// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

/**
* \brief TODO: short file description
* \file file_utils.h
*/
#pragma once

#include <string>
#include <gtest/gtest.h>

namespace {
    bool strContains(std::string str, std::string substr) {
        return str.find(substr) != std::string::npos;
    }
}

#define ASSERT_STR_CONTAINS(str, substr) ASSERT_PRED2(&strContains, str, substr)
#define ASSERT_STR_DOES_NOT_CONTAIN(str, substr) ASSERT_PRED2 \
        (std::not2(std::ptr_fun<std::string, std::string, bool>(strContains)), str, substr)
#define EXPECT_STR_CONTAINS(str, substr) EXPECT_PRED2(&strContains, str, substr)
