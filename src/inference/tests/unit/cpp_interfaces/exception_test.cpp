// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cpp/exception2status.hpp>

using namespace InferenceEngine;
IE_SUPPRESS_DEPRECATED_START

using ExceptionTests = ::testing::Test;

template <StatusCode statusCode>
class WrapperClass {
public:
    static InferenceEngine::StatusCode toStatusWrapper(InferenceEngine::ResponseDesc* resp) {
        TO_STATUS(IE_EXCEPTION_SWITCH(
            statusCode,
            ExceptionType,
            InferenceEngine::details::ThrowNow<ExceptionType>{IE_LOCATION_PARAM} <<= std::stringstream{}))
    }

    static InferenceEngine::StatusCode toStatusWrapperMsg(std::string& msg, InferenceEngine::ResponseDesc* resp) {
        TO_STATUS(IE_EXCEPTION_SWITCH(
            statusCode,
            ExceptionType,
            InferenceEngine::details::ThrowNow<ExceptionType>{IE_LOCATION_PARAM} <<= std::stringstream{} << msg))
    }
};

// TO_STATUS macros tests
TEST_F(ExceptionTests, canConvertToStatus) {
    ResponseDesc* resp = nullptr;
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
    auto actual = std::make_shared<WrapperClass<StatusCode::INFER_NOT_STARTED>>();
    ASSERT_THROW(CALL_STATUS_FNC_NO_ARGS(toStatusWrapper), InferenceEngine::InferNotStarted);
}

TEST_F(ExceptionTests, canHandleNullPtr) {
    class Mock {
    public:
        StatusCode func0(ResponseDesc*) {
            return StatusCode ::OK;
        }
        StatusCode func1(int, ResponseDesc*) {
            return StatusCode ::OK;
        }
    };
    //  shared_ptr holding the nullptr
    std::shared_ptr<Mock> actual;
    //  check that accessing the nullptr thru macros throws
    ASSERT_THROW(CALL_STATUS_FNC_NO_ARGS(func0), InferenceEngine::Exception);
    ASSERT_THROW(CALL_STATUS_FNC(func1, 0), InferenceEngine::Exception);
}

TEST_F(ExceptionTests, throwAfterConvertStatusToClassContainMessage) {
    std::string refMessage = "Exception message!";
    auto actual = std::make_shared<WrapperClass<StatusCode::NOT_ALLOCATED>>();
    try {
        CALL_STATUS_FNC(toStatusWrapperMsg, refMessage);
    } catch (const NotAllocated& iex) {
        std::string actualMessage = iex.what();
        ASSERT_TRUE(actualMessage.find(refMessage) != std::string::npos);
    }
}
