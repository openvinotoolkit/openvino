// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "details/ie_exception.hpp"
#include "ie_common.h"

// TODO: cover <cpp_interfaces/exception2status.hpp> and <details/ie_exception_conversion.hpp> from
//  tests/unit/inference_engine/exception_test.cpp

TEST(ExceptionTests, CopyConstructor) {
    InferenceEngine::details::InferenceEngineException exception(__FILE__, __LINE__);
    ASSERT_NO_THROW(InferenceEngine::details::InferenceEngineException {exception});
}

TEST(ExceptionTests, CanThrowUsingMacro) {
    std::string message = "Exception message!";
    ASSERT_THROW(THROW_IE_EXCEPTION << message, InferenceEngine::details::InferenceEngineException);
}

TEST(ExceptionTests, CanThrowScoringException) {
    InferenceEngine::details::InferenceEngineException exception(__FILE__, __LINE__);
    ASSERT_THROW(throw exception, InferenceEngine::details::InferenceEngineException);
}

TEST(ExceptionTests, CanDefineExceptionContent) {
    InferenceEngine::details::InferenceEngineException exception(__FILE__, __LINE__);
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
    catch (InferenceEngine::details::InferenceEngineException &iex) {
        std::string ref_message = message + "\n" + __FILE__ + ":" + std::to_string(lineNum);
        ASSERT_STREQ(iex.what(), ref_message.c_str());
    }
}
#else
TEST(ExceptionTests, ExceptionShowsCorrectMessageReleaseVersion) {
    std::string message = "exception";
    try {
        THROW_IE_EXCEPTION << message;
    }
    catch (InferenceEngine::details::InferenceEngineException &iex) {
        std::string ref_message = message;
        ASSERT_STREQ(iex.what(), ref_message.c_str());
    }
}
#endif

TEST(ExceptionTests, ExceptionCanBeCoughtAsStandard) {
    ASSERT_THROW(THROW_IE_EXCEPTION, std::exception);
}

TEST(ExceptionTests, CanThrowStatusCode) {
    try {
        THROW_IE_EXCEPTION << InferenceEngine::details::as_status << InferenceEngine::StatusCode::INFER_NOT_STARTED;
    }
    catch (const InferenceEngine::details::InferenceEngineException &iex) {
        ASSERT_TRUE(iex.hasStatus());
        ASSERT_EQ(iex.getStatus(), InferenceEngine::StatusCode::INFER_NOT_STARTED);
    }
}

TEST(ExceptionTests, HandleOnlyFirstStatus) {
    try {
        THROW_IE_EXCEPTION << InferenceEngine::details::as_status <<
                           InferenceEngine::StatusCode::NETWORK_NOT_LOADED << InferenceEngine::StatusCode::NOT_FOUND;
    }
    catch (const InferenceEngine::details::InferenceEngineException &iex) {
        ASSERT_TRUE(iex.hasStatus());
        ASSERT_EQ(iex.getStatus(), InferenceEngine::StatusCode::NETWORK_NOT_LOADED);
    }
}

TEST(ExceptionTests, IgnoreNotStatusCodeEnumAfterManip) {
    enum testEnum : int {
        FIRST = 1
    };
    try {
        THROW_IE_EXCEPTION << InferenceEngine::details::as_status << testEnum::FIRST;
    }
    catch (const InferenceEngine::details::InferenceEngineException &iex) {
        ASSERT_FALSE(iex.hasStatus());
    }
}

TEST(ExceptionTests, CanUseManipulatorStandalone) {
    auto iex = InferenceEngine::details::InferenceEngineException("filename", 1);
    as_status(iex);
    try {
        throw iex << InferenceEngine::StatusCode::NOT_IMPLEMENTED;
    } catch (const InferenceEngine::details::InferenceEngineException &iex) {
        ASSERT_TRUE(iex.hasStatus());
        ASSERT_EQ(iex.getStatus(), InferenceEngine::StatusCode::NOT_IMPLEMENTED);
    }
}

TEST(ExceptionTests, StatusCodeNotAppearInMessageAfterCatch) {
    std::string message = "Exception message!";
    std::string strStatusCode = std::to_string(InferenceEngine::StatusCode::NETWORK_NOT_LOADED);
    try {
        THROW_IE_EXCEPTION << "<unique--" << InferenceEngine::details::as_status <<
                           InferenceEngine::StatusCode::NETWORK_NOT_LOADED << "--unique>" << message;
    }
    catch (const InferenceEngine::details::InferenceEngineException &iex) {
        ASSERT_THAT(iex.what(), testing::HasSubstr(message));
        ASSERT_THAT(iex.what(), testing::Not(testing::HasSubstr("<unique--" + strStatusCode + "--unique>")));
    }
}

TEST(ExceptionTests, StatusCodeAppearInMessageAfterCatch) {
    std::string message = "Exception message!";
    std::string strStatusCode = std::to_string(InferenceEngine::StatusCode::NETWORK_NOT_LOADED);
    try {
        THROW_IE_EXCEPTION << "<unique--" << InferenceEngine::StatusCode::NETWORK_NOT_LOADED << "--unique>" << message;
    }
    catch (const InferenceEngine::details::InferenceEngineException &iex) {
        ASSERT_THAT(iex.what(), testing::HasSubstr(message));
        ASSERT_THAT(iex.what(), testing::HasSubstr("<unique--" + strStatusCode + "--unique>"));
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
    ASSERT_THROW(IE_ASSERT(false), InferenceEngine::details::InferenceEngineException);
}

TEST(ExceptionTests, ExceptionWithAssertThrowsExceptionIfFalseExpession) {
    ASSERT_THROW(IE_ASSERT(0 == 1), InferenceEngine::details::InferenceEngineException);
}
#endif  // NDEBUG
