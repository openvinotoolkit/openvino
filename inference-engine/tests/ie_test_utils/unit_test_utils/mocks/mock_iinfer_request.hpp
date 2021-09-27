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
    MOCK_METHOD(StatusCode, StartAsync, (ResponseDesc*), (noexcept));
    MOCK_METHOD(StatusCode, Wait, (int64_t, ResponseDesc*), (noexcept));
    MOCK_METHOD(StatusCode, GetUserData, (void**, ResponseDesc*), (noexcept));
    MOCK_METHOD(StatusCode, SetUserData, (void*, ResponseDesc*), (noexcept));
    MOCK_METHOD(StatusCode, SetCompletionCallback, (IInferRequest::CompletionCallback), (noexcept));
    MOCK_METHOD(StatusCode, Infer, (ResponseDesc*), (noexcept));
    MOCK_METHOD(StatusCode, GetPerformanceCounts,
        ((std::map<std::string, InferenceEngineProfileInfo> &), ResponseDesc*), (const, noexcept));
    MOCK_METHOD(StatusCode, GetBlob, (const char*, Blob::Ptr&, ResponseDesc*), (noexcept));
    MOCK_METHOD(StatusCode, GetPreProcess,
        (const char*, const PreProcessInfo**, ResponseDesc*), (const, noexcept));
    MOCK_METHOD(StatusCode, SetBlob, (const char*, const Blob::Ptr&, ResponseDesc*), (noexcept));
    MOCK_METHOD(StatusCode, SetBlob,
        (const char*, const Blob::Ptr&, const PreProcessInfo&, ResponseDesc*), (noexcept));
    MOCK_METHOD(StatusCode, SetBatch, (int batch, ResponseDesc*), (noexcept));
    MOCK_METHOD(StatusCode, Cancel, (ResponseDesc*), (noexcept));
};

IE_SUPPRESS_DEPRECATED_END
