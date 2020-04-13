// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <map>
#include <gmock/gmock.h>
#include <ie_iinfer_request.hpp>

#include <cpp_interfaces/impl/ie_infer_async_request_internal.hpp>

#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_infer_request_internal.hpp"

using namespace InferenceEngine;

class MockAsyncInferRequestInternal : public AsyncInferRequestInternal {
public:
    using AsyncInferRequestInternal::SetBlob;
    MockAsyncInferRequestInternal(InputsDataMap networkInputs, OutputsDataMap networkOutputs)
            : AsyncInferRequestInternal(networkInputs, networkOutputs) {}

    MOCK_METHOD0(StartAsyncImpl, void());
    MOCK_METHOD1(Wait, InferenceEngine::StatusCode(int64_t));
    MOCK_METHOD1(GetUserData, void(void **));
    MOCK_METHOD1(SetUserData, void(void *));
    MOCK_METHOD0(InferImpl, void());
    MOCK_CONST_METHOD1(GetPerformanceCounts, void(std::map<std::string, InferenceEngineProfileInfo> &));
    MOCK_METHOD1(setNetworkInputs, void(InputsDataMap));
    MOCK_METHOD1(setNetworkOutputs, void(OutputsDataMap));
    MOCK_METHOD2(GetBlob, void(const char *name, Blob::Ptr &));
    MOCK_METHOD1(SetCompletionCallback, void(IInferRequest::CompletionCallback));
};
