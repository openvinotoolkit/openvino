// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * \brief header file for MockIInferRequest
 * \file mock_iinfer_request.hpp
 */
#pragma once

#include <gmock/gmock.h>

#include <map>
#include <memory>
#include <string>

#include "ie_iinfer_request.hpp"

using namespace InferenceEngine;

IE_SUPPRESS_DEPRECATED_START

class MockIInferRequest : public IInferRequest {
public:
    MOCK_QUALIFIED_METHOD1(StartAsync, noexcept, StatusCode(ResponseDesc*));
    MOCK_QUALIFIED_METHOD2(Wait, noexcept, StatusCode(int64_t millis_timeout, ResponseDesc*));
    MOCK_QUALIFIED_METHOD2(GetUserData, noexcept, StatusCode(void**, ResponseDesc*));
    MOCK_QUALIFIED_METHOD2(SetUserData, noexcept, StatusCode(void*, ResponseDesc*));
    MOCK_QUALIFIED_METHOD1(SetCompletionCallback, noexcept, StatusCode(IInferRequest::CompletionCallback));
    MOCK_QUALIFIED_METHOD1(Infer, noexcept, StatusCode(ResponseDesc*));
    MOCK_QUALIFIED_METHOD2(GetPerformanceCounts, const noexcept,
                           StatusCode(std::map<std::string, InferenceEngineProfileInfo> &perfMap, ResponseDesc*));
    MOCK_QUALIFIED_METHOD3(GetBlob, noexcept, StatusCode(const char*, Blob::Ptr&, ResponseDesc*));
    MOCK_QUALIFIED_METHOD3(GetPreProcess, const noexcept, StatusCode(const char*, const PreProcessInfo**, ResponseDesc*));
    MOCK_QUALIFIED_METHOD3(SetBlob, noexcept, StatusCode(const char*, const Blob::Ptr&, ResponseDesc*));
    MOCK_QUALIFIED_METHOD4(SetBlob, noexcept, StatusCode(const char*, const Blob::Ptr&, const PreProcessInfo&, ResponseDesc*));
    MOCK_QUALIFIED_METHOD2(SetBatch, noexcept, StatusCode(int batch, ResponseDesc*));
    MOCK_QUALIFIED_METHOD1(Cancel, noexcept, InferenceEngine::StatusCode(ResponseDesc*));
};

IE_SUPPRESS_DEPRECATED_END
