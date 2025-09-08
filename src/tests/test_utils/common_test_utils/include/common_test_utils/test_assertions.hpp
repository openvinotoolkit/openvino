// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"
#include "openvino/util/pp.hpp"

inline bool strContains(const std::string& str, const std::string& substr) {
    return str.find(substr) != std::string::npos;
}

inline bool strDoesnotContain(const std::string& str, const std::string& substr) {
    return !strContains(str, substr);
}

#define ASSERT_STR_CONTAINS(str, substr) ASSERT_PRED2(&strContains, str, substr)

#define ASSERT_STR_DOES_NOT_CONTAIN(str, substr) ASSERT_PRED2(&strDoesnotContain, str, substr)

#define EXPECT_STR_CONTAINS(str, substr) EXPECT_PRED2(&strContains, str, substr)

#define ASSERT_STRINGEQ(lhs, rhs) compare_cpp_strings(lhs, rhs)

#define OV_ASSERT_NO_THROW(statement) OV_ASSERT_NO_THROW_(statement, GTEST_FATAL_FAILURE_)

#define OV_ASSERT_NO_THROW_(statement, fail)                              \
    GTEST_AMBIGUOUS_ELSE_BLOCKER_                                         \
    if (::testing::internal::AlwaysTrue()) {                              \
        try {                                                             \
            GTEST_SUPPRESS_UNREACHABLE_CODE_WARNING_BELOW_(statement);    \
        } catch (const std::exception& e) {                               \
            fail("Expected: " #statement " doesn't throw an exception.\n" \
                 "  Actual: it throws.")                                  \
                << e.what();                                              \
        } catch (...) {                                                   \
            fail("Expected: " #statement " doesn't throw an exception.\n" \
                 "  Actual: it throws.");                                 \
        }                                                                 \
    }

#define OV_EXPECT_THROW(statement, exp_exception, exception_what_matcher) \
    try {                                                                 \
        GTEST_SUPPRESS_UNREACHABLE_CODE_WARNING_BELOW_(statement);        \
        FAIL() << "Expected exception " << OV_PP_TOSTRING(exp_exception); \
    } catch (const exp_exception& ex) {                                   \
        EXPECT_THAT(ex.what(), exception_what_matcher);                   \
    } catch (const std::exception& e) {                                   \
        FAIL() << "Unexpected exception " << e.what();                    \
    } catch (...) {                                                       \
        FAIL() << "Unknown exception";                                    \
    }

#define OV_EXPECT_THROW_HAS_SUBSTRING(statement, exp_exception, exception_what_matcher) \
    try {                                                                               \
        GTEST_SUPPRESS_UNREACHABLE_CODE_WARNING_BELOW_(statement);                      \
        FAIL() << "Expected exception " << OV_PP_TOSTRING(exp_exception);               \
    } catch (const exp_exception& ex) {                                                 \
        EXPECT_THAT(ex.what(), ::testing::HasSubstr(exception_what_matcher));           \
    } catch (const std::exception& e) {                                                 \
        FAIL() << "Unexpected exception " << e.what();                                  \
    } catch (...) {                                                                     \
        FAIL() << "Unknown exception";                                                  \
    }

inline void compare_cpp_strings(const std::string& lhs, const std::string& rhs) {
    ASSERT_STREQ(lhs.c_str(), rhs.c_str());
}
