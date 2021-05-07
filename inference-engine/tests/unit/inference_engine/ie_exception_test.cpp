// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "ie_common.h"
#include "cpp_interfaces/exception2status.hpp"

// TODO: cover <cpp_interfaces/exception2status.hpp> and <details/ie_exception_conversion.hpp> from
//  tests/unit/inference_engine/exception_test.cpp

TEST(ExceptionTests, CanThrowUsingMacro) {
    std::string message = "Exception message!";
    ASSERT_THROW(THROW_IE_EXCEPTION << message, InferenceEngine::Exception);
}

TEST(ExceptionTests, CanThrowScoringException) {
    InferenceEngine::Exception exception{""};
    ASSERT_THROW(throw exception, InferenceEngine::Exception);
}

TEST(ExceptionTests, CanDefineExceptionContent) {
    InferenceEngine::Exception exception{""};
    ASSERT_STREQ(exception.what(), "");
}


#ifndef NDEBUG
TEST(ExceptionTests, ExceptionShowsCorrectMessageDebugVersion) {
    std::string message = "exception";
    int lineNum = 0;
    try {
        lineNum = __LINE__ + 1;
        THROW_IE_EXCEPTION << message;
    }
    catch (InferenceEngine::Exception &iex) {
        std::string ref_message = std::string {"\n"} + __FILE__ + ":" + std::to_string(lineNum) + " " + message;
        ASSERT_STREQ(iex.what(), ref_message.c_str());
    }
}
#else
TEST(ExceptionTests, ExceptionShowsCorrectMessageReleaseVersion) {
    std::string message = "exception";
    try {
        THROW_IE_EXCEPTION << message;
    }
    catch (InferenceEngine::Exception &iex) {
        std::string ref_message = message;
        ASSERT_STREQ(iex.what(), ref_message.c_str());
    }
}
#endif

TEST(ExceptionTests, ExceptionCanBeCaughtAsStandard) {
    ASSERT_THROW(THROW_IE_EXCEPTION, std::exception);
}

TEST(ExceptionTests, CanThrowStatusCode) {
    try {
        THROW_IE_EXCEPTION_WITH_STATUS(InferNotStarted);
    }
    catch (const InferenceEngine::InferNotStarted& iex) {
        ASSERT_EQ(InferenceEngine::ExceptionToStatus(iex), InferenceEngine::StatusCode::INFER_NOT_STARTED);
    }
}

#ifdef    NDEBUG  // disabled for debug as macros calls assert()
TEST(ExceptionTests, ExceptionWithAssertThrowsNothingIfTrue) {
    ASSERT_NO_THROW(IE_ASSERT(true) << "shouldn't assert if true");
}

TEST(ExceptionTests, ExceptionWithAssertThrowsNothingIfExpressionTrue) {
    ASSERT_NO_THROW(IE_ASSERT(2 > 0) << "shouldn't assert if true expression");
}

TEST(ExceptionTests, ExceptionWithAssertThrowsExceptionIfFalse) {
    ASSERT_THROW(IE_ASSERT(false), InferenceEngine::Exception);
}

TEST(ExceptionTests, ExceptionWithAssertThrowsExceptionIfFalseExpession) {
    ASSERT_THROW(IE_ASSERT(0 == 1), InferenceEngine::Exception);
}
#endif  // NDEBUG
