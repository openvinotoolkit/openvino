// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "tests_utils.hpp"
#include <cpp_interfaces/exception2status.hpp>
#include <details/ie_exception_conversion.hpp>

using namespace InferenceEngine;

class ExceptionTests : public ::testing::Test {
};

template<StatusCode T>
class WrapperClass {
public:
    static InferenceEngine::StatusCode toStatusWrapper(InferenceEngine::ResponseDesc *resp) {
        TO_STATUS(THROW_IE_EXCEPTION << details::as_status << T);
    }

    static InferenceEngine::StatusCode toStatusWrapperMsg(std::string &msg, InferenceEngine::ResponseDesc *resp) {
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
    auto actual = std::make_shared<WrapperClass<StatusCode::INFER_NOT_STARTED>>();
    ASSERT_THROW(CALL_STATUS_FNC_NO_ARGS(toStatusWrapper), InferenceEngine::InferNotStarted);
}

TEST_F(ExceptionTests, canHandleNullPtr) {
    class Mock {
    public:
        StatusCode func0(ResponseDesc* resp) {return StatusCode ::OK;};
        StatusCode func1(int x, ResponseDesc* resp) {return StatusCode ::OK;};
    };
    //  shared_ptr holding the nullptr
    std::shared_ptr<Mock> actual;
    //  check that accessing the nullptr thru macros throws
    ASSERT_THROW(CALL_STATUS_FNC_NO_ARGS(func0), InferenceEngine::details::InferenceEngineException);
    ASSERT_THROW(CALL_STATUS_FNC(func1, 0), InferenceEngine::details::InferenceEngineException);
}

TEST_F(ExceptionTests, throwAfterConvertStatusToClassContainMessage) {
    std::string message = "Exception message!";
    auto actual = std::make_shared<WrapperClass<StatusCode::NOT_ALLOCATED>>();
    try {
        CALL_STATUS_FNC(toStatusWrapperMsg, message);
    } catch (const NotAllocated &iex) {
        ASSERT_STR_CONTAINS(iex.what(), message);
    }
}

