// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <string>
#include <vector>
#include <memory>
#include <map>

#include "auto_async_infer_request.hpp"

namespace AutoPlugin {
using namespace InferenceEngine;

AutoAsyncInferRequest::AutoAsyncInferRequest(const AutoInferRequest::Ptr&           inferRequest,
                               const AutoExecutableNetwork::Ptr&      autoExecutableNetwork,
                               const InferenceEngine::ITaskExecutor::Ptr&    callbackExecutor)
                               : AsyncInferRequestThreadSafeDefault(inferRequest, autoExecutableNetwork, callbackExecutor) {
    // todo: redefine _pipeline
}
void AutoAsyncInferRequest::Infer_ThreadUnsafe() {
    InferUsingAsync();
}
std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> AutoAsyncInferRequest::GetPerformanceCounts() const {
    CheckState();
    return {};
}
AutoAsyncInferRequest::~AutoAsyncInferRequest() {
    StopAndWait();
}
} // namespace AutoPlugin