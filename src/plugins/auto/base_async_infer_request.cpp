// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <string>
#include <vector>
#include <memory>
#include <map>

#include "base_async_infer_request.hpp"
#include <ie_icore.hpp>
#include <ie_metric_helpers.hpp>
#include <ie_plugin_config.hpp>

namespace MultiDevicePlugin {
BaseAsyncInferRequest::BaseAsyncInferRequest(const Schedule::Ptr& schedule,
    const InferenceEngine::IInferRequestInternal::Ptr& inferRequest,
    const InferenceEngine::ITaskExecutor::Ptr& callbackExecutor):
    AsyncInferRequestThreadSafeDefault(inferRequest, nullptr, callbackExecutor),
    _schedule(schedule),
    _inferRequest(inferRequest) {
    auto pipeline = _schedule->GetPipeline(_inferRequest, &_workerInferRequest);
    if (pipeline.size() > 0) {
        _pipeline = std::move(pipeline);
    }
}

void BaseAsyncInferRequest::Infer_ThreadUnsafe() {
    InferUsingAsync();
}

std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>
BaseAsyncInferRequest::GetPerformanceCounts() const {
    CheckState();
    auto multiDeviceInfer = std::dynamic_pointer_cast<MultiDeviceInferRequest>(_inferRequest);
    return multiDeviceInfer->GetPerformanceCounts();
}

BaseAsyncInferRequest::~BaseAsyncInferRequest() {
    StopAndWait();
}

}  // namespace MultiDevicePlugin
