// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "tests_utils.hpp"
#include <details/ie_exception.hpp>
#include <cpp_interfaces/exception2status.hpp>
#include <details/ie_exception_conversion.hpp>

using namespace InferenceEngine;
class ExceptionTests : public ::testing::Test {};

TEST_F(ExceptionTests, canThrowUsingMacro) {
    std::string message = "Exception message!";
    EXPECT_THROW(THROW_IE_EXCEPTION << message, InferenceEngine::details::InferenceEngineException);
}

TEST_F(ExceptionTests, canThrowScoringException) {
    InferenceEngine::details::InferenceEngineException exception(__FILE__, __LINE__);
    EXPECT_THROW(throw exception, InferenceEngine::details::InferenceEngineException);
}

TEST_F(ExceptionTests, canDefineExceptionContent) {
    InferenceEngine::details::InferenceEngineException exception(__FILE__, __LINE__);

    ASSERT_STR_CONTAINS(exception.what(), "");
}

TEST_F(ExceptionTests, exceptionShowsCorrectMessage) {
    std::string message = "Exception message!";
    try {
        THROW_IE_EXCEPTION << message;
    }
    catch (InferenceEngine::details::InferenceEngineException ex) {
        ASSERT_STR_CONTAINS(ex.what(), message);
    }
}

//TODO: enable test once info appears in exception message
TEST_F(ExceptionTests, DISABLED_exceptionShowsCorrectFileAndLineNumbers) {
    std::string message = __FILE__;
    int lineNum = 0;
    try {
        lineNum = __LINE__; THROW_IE_EXCEPTION;
    }
    catch (InferenceEngine::details::InferenceEngineException ex) {
        message += ":" + std::to_string(lineNum);
        ASSERT_STR_CONTAINS(ex.what(), message); 
    }
}

TEST_F(ExceptionTests, exceptionCanBeCoughtAsStandard) {
    ASSERT_THROW(THROW_IE_EXCEPTION, std::exception);
}

// Status Code tests
TEST_F(ExceptionTests, canThrowStatusCode) {
    try {
        THROW_IE_EXCEPTION << InferenceEngine::details::as_status << InferenceEngine::StatusCode::INFER_NOT_STARTED;
    }
    catch (const InferenceEngine::details::InferenceEngineException &iex) {
        ASSERT_TRUE(iex.hasStatus());
        ASSERT_EQ(iex.getStatus(), InferenceEngine::StatusCode::INFER_NOT_STARTED);
    }
}

TEST_F(ExceptionTests, handleOnlyFirstStatus) {
    try {
        THROW_IE_EXCEPTION<< InferenceEngine::details::as_status <<
                InferenceEngine::StatusCode::NETWORK_NOT_LOADED << InferenceEngine::StatusCode::NOT_FOUND;
    }
    catch (const InferenceEngine::details::InferenceEngineException &iex) {
        ASSERT_TRUE(iex.hasStatus());
        ASSERT_EQ(iex.getStatus(), InferenceEngine::StatusCode::NETWORK_NOT_LOADED);
    }
}

TEST_F(ExceptionTests, ignoreNotStatusCodeEnumAfterManip) {
    enum testEnum : int { FIRST = 1 };
    try {
        THROW_IE_EXCEPTION<< InferenceEngine::details::as_status << testEnum::FIRST;
    }
    catch (const InferenceEngine::details::InferenceEngineException &iex) {
        ASSERT_FALSE(iex.hasStatus());
    }
}

TEST_F(ExceptionTests, canUseManipulatorStandalone) {
    auto iex = details::InferenceEngineException("filename", 1);
    as_status(iex);
    try {
        throw iex << NOT_IMPLEMENTED;
    } catch (const InferenceEngine::details::InferenceEngineException &iex) {
        ASSERT_TRUE(iex.hasStatus());
        ASSERT_EQ(iex.getStatus(), InferenceEngine::StatusCode::NOT_IMPLEMENTED);
    }
}

TEST_F(ExceptionTests, statusCodeNotAppearInMessageAfterCatch) {
    std::string message = "Exception message!";
    std::string strStatusCode = std::to_string(StatusCode::NETWORK_NOT_LOADED);
    try {
        THROW_IE_EXCEPTION<< "<unique--" << InferenceEngine::details::as_status  <<
        InferenceEngine::StatusCode::NETWORK_NOT_LOADED << "--unique>" << message;
    }
    catch (const InferenceEngine::details::InferenceEngineException &iex) {
        ASSERT_STR_CONTAINS(iex.what(), message);
        ASSERT_STR_DOES_NOT_CONTAIN(iex.what(), "<unique--" + strStatusCode + "--unique>");
    }
}

TEST_F(ExceptionTests, statusCodeAppearInMessageAfterCatch) {
    std::string message = "Exception message!";
    std::string strStatusCode = std::to_string(StatusCode::NETWORK_NOT_LOADED);
    try {
        THROW_IE_EXCEPTION<< "<unique--" << InferenceEngine::StatusCode::NETWORK_NOT_LOADED << "--unique>" << message;
    }
    catch (const InferenceEngine::details::InferenceEngineException &iex) {
        ASSERT_STR_CONTAINS(iex.what(), message);
        ASSERT_STR_CONTAINS(iex.what(), "<unique--" + strStatusCode + "--unique>");
    }
}

template <StatusCode T>
class WrapperClass {
public:
    static InferenceEngine::StatusCode toStatusWrapper(InferenceEngine::ResponseDesc* resp) {
        TO_STATUS(THROW_IE_EXCEPTION << details::as_status << T);
    }
    static InferenceEngine::StatusCode toStatusWrapperMsg(std::string &msg, InferenceEngine::ResponseDesc* resp) {
        TO_STATUS(THROW_IE_EXCEPTION << details::as_status << T << msg);
    }
};

// TO_STATUS macros tests
TEST_F(ExceptionTests, canConvertToStatus) {
    ResponseDesc *resp = nullptr;
    ASSERT_EQ(WrapperClass<StatusCode::GENERAL_ERROR>::toStatusWrapper(resp), StatusCode::GENERAL_ERROR);
    ASSERT_EQ(WrapperClass<StatusCode::NOT_IMPLEMENTED>::toStatusWrapper(resp), StatusCode::NOT_IMPLEMENTED);
    ASSERT_EQ(WrapperClass<StatusCode::NETWORK_NOT_LOADED>::toStatusWrapper(resp), StatusCode::NETWORK_NOT_LOADED);
    ASSERT_EQ(WrapperClass<StatusCode::PARAMETER_MISMATCH>::toStatusWrapper(resp), StatusCode::PARAMETER_MISMATCH);
    ASSERT_EQ(WrapperClass<StatusCode::NOT_FOUND>::toStatusWrapper(resp), StatusCode::NOT_FOUND);
    ASSERT_EQ(WrapperClass<StatusCode::OUT_OF_BOUNDS>::toStatusWrapper(resp), StatusCode::OUT_OF_BOUNDS);
    ASSERT_EQ(WrapperClass<StatusCode::UNEXPECTED>::toStatusWrapper(resp), StatusCode::UNEXPECTED);
    ASSERT_EQ(WrapperClass<StatusCode::REQUEST_BUSY>::toStatusWrapper(resp), StatusCode::REQUEST_BUSY);
    ASSERT_EQ(WrapperClass<StatusCode::RESULT_NOT_READY>::toStatusWrapper(resp), StatusCode::RESULT_NOT_READY);
    ASSERT_EQ(WrapperClass<StatusCode::NOT_ALLOCATED>::toStatusWrapper(resp), StatusCode::NOT_ALLOCATED);
    ASSERT_EQ(WrapperClass<StatusCode::INFER_NOT_STARTED>::toStatusWrapper(resp), StatusCode::INFER_NOT_STARTED);
}

// CALL_STATUS_FNC macros tests
TEST_F(ExceptionTests, canConvertStatusToException) {
    std::shared_ptr<WrapperClass<StatusCode::INFER_NOT_STARTED>> actual;
    ASSERT_THROW(CALL_STATUS_FNC_NO_ARGS(toStatusWrapper), InferenceEngine::InferNotStarted);
}

TEST_F(ExceptionTests, throwAfterConvertStatusToClassContainMessage) {
    std::string message = "Exception message!";
    std::shared_ptr<WrapperClass<StatusCode::NOT_ALLOCATED>> actual;
    try {
        CALL_STATUS_FNC(toStatusWrapperMsg, message);
    } catch (const NotAllocated &iex) {
        ASSERT_STR_CONTAINS(iex.what(), message);
    }
}

#ifdef    NDEBUG //disabled for debug as macros calls assert()
TEST_F(ExceptionTests, exceptionWithAssertThrowsNothingIfTrue) {
    EXPECT_NO_THROW(IE_ASSERT(true) << "shouldn't assert if true");
}

TEST_F(ExceptionTests, exceptionWithAssertThrowsNothingIfExpressionTrue) {
    EXPECT_NO_THROW(IE_ASSERT(2 > 0) << "shouldn't assert if true expression");
}

TEST_F(ExceptionTests, exceptionWithAssertThrowsExceptionIfFalse) {
    EXPECT_THROW(IE_ASSERT(false), InferenceEngine::details::InferenceEngineException);
}

TEST_F(ExceptionTests, exceptionWithAssertThrowsExceptionIfFalseExpession) {
    EXPECT_THROW(IE_ASSERT(0 == 1), InferenceEngine::details::InferenceEngineException);
}
#endif	//NDEBUG
