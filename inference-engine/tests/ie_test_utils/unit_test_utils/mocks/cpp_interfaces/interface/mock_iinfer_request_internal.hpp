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
    MOCK_METHOD0(StartAsync, void());
    MOCK_METHOD1(Wait, InferenceEngine::StatusCode(int64_t));
    MOCK_METHOD0(Infer, void());
    MOCK_CONST_METHOD0(GetPerformanceCounts, std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>());
    MOCK_METHOD2(SetBlob, void(const std::string&, const InferenceEngine::Blob::Ptr &));
    MOCK_METHOD1(GetBlob, InferenceEngine::Blob::Ptr(const std::string&));
    MOCK_METHOD3(SetBlob, void(const std::string&, const InferenceEngine::Blob::Ptr &, const InferenceEngine::PreProcessInfo&));
    MOCK_CONST_METHOD1(GetPreProcess, const InferenceEngine::PreProcessInfo&(const std::string&));
    MOCK_METHOD1(SetCallback, void(std::function<void(std::exception_ptr)>));
    MOCK_METHOD1(SetBatch, void(int));
    MOCK_METHOD0(QueryState, std::vector<InferenceEngine::IVariableStateInternal::Ptr>());
    MOCK_METHOD0(Cancel, void());
    MOCK_METHOD0(StartAsyncImpl, void());
    MOCK_METHOD0(InferImpl, void());
    MOCK_METHOD0(checkBlobs, void());
};
