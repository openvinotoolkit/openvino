// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock.h>

#include <map>
#include <string>
#include <vector>

#include <cpp_interfaces/interface/ie_ivariable_state_internal.hpp>
#include <cpp_interfaces/interface/ie_iinfer_request_internal.hpp>

class MockIInferRequestInternal : public InferenceEngine::IInferRequestInternal {
public:
    using InferenceEngine::IInferRequestInternal::IInferRequestInternal;
    MOCK_METHOD(void, StartAsync, ());
    MOCK_METHOD(InferenceEngine::StatusCode, Wait, (int64_t));
    MOCK_METHOD(void, Infer, ());
    MOCK_METHOD((std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>),
        GetPerformanceCounts, (), (const));
    MOCK_METHOD(void, SetBlob, (const std::string&, const InferenceEngine::Blob::Ptr &));
    MOCK_METHOD(InferenceEngine::Blob::Ptr, GetBlob, (const std::string&));
    MOCK_METHOD(void, SetBlob, (const std::string&, const InferenceEngine::Blob::Ptr &, const InferenceEngine::PreProcessInfo&));
    MOCK_METHOD(const InferenceEngine::PreProcessInfo&, GetPreProcess, (const std::string&));
    MOCK_METHOD(void, SetCallback, (std::function<void(std::exception_ptr)>));
    MOCK_METHOD(void, SetBatch, (int));
    MOCK_METHOD(std::vector<InferenceEngine::IVariableStateInternal::Ptr>, QueryState, ());
    MOCK_METHOD(void, Cancel, ());
    MOCK_METHOD(void, StartAsyncImpl, ());
    MOCK_METHOD(void, InferImpl, ());
    MOCK_METHOD(void, checkBlobs, ());
};
