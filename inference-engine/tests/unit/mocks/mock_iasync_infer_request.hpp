// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * \brief header file for MockIInferRequest
 * \file mock_iinfer_request.hpp
 */
#pragma once

#include "ie_iexecutable_network.hpp"
#include <gmock/gmock-generated-function-mockers.h>

using namespace InferenceEngine;

class MockIInferRequest : public IInferRequest {
public:
    using Ptr = std::shared_ptr<MockIInferRequest>;
    MOCK_QUALIFIED_METHOD1(StartAsync, noexcept, StatusCode(ResponseDesc*));
    MOCK_QUALIFIED_METHOD2(Wait, noexcept, StatusCode(int64_t millis_timeout, ResponseDesc*));
    MOCK_QUALIFIED_METHOD2(GetUserData, noexcept, StatusCode(void**, ResponseDesc*));
    MOCK_QUALIFIED_METHOD2(SetUserData, noexcept, StatusCode(void*, ResponseDesc*));
    MOCK_QUALIFIED_METHOD1(SetCompletionCallback, noexcept, StatusCode(IInferRequest::CompletionCallback));
    MOCK_QUALIFIED_METHOD0(Release, noexcept, void ());
    MOCK_QUALIFIED_METHOD1(Infer, noexcept, StatusCode(ResponseDesc*));
    MOCK_QUALIFIED_METHOD2(GetPerformanceCounts, const noexcept,
                           StatusCode(std::map<std::string, InferenceEngineProfileInfo> &perfMap, ResponseDesc*));
    MOCK_QUALIFIED_METHOD3(GetBlob, noexcept, StatusCode(const char*, Blob::Ptr&, ResponseDesc*));
    MOCK_QUALIFIED_METHOD3(SetBlob, noexcept, StatusCode(const char*, const Blob::Ptr&, ResponseDesc*));
	MOCK_QUALIFIED_METHOD2(SetBatch, noexcept, StatusCode(int batch, ResponseDesc*));
};
