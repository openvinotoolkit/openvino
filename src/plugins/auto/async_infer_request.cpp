// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "async_infer_request.hpp"

namespace MultiDevicePlugin {
AsyncInferRequest::AsyncInferRequest(const Schedule::Ptr& schedule,
    const IInferPtr& inferRequest,
    const IE::ITaskExecutor::Ptr& callbackExecutor):
    AsyncInferRequestThreadSafeDefault(inferRequest, nullptr, callbackExecutor),
    _schedule(schedule),
    _inferRequest(inferRequest) {
    auto pipeline = _schedule->GetPipeline(_inferRequest, &_workerInferRequest);
    if (pipeline.size() > 0) {
        _pipeline = std::move(pipeline);
    }
}

void AsyncInferRequest::Infer_ThreadUnsafe() {
    InferUsingAsync();
}

std::map<std::string, IE::InferenceEngineProfileInfo>
AsyncInferRequest::GetPerformanceCounts() const {
    CheckState();
    auto multiDeviceInfer = std::dynamic_pointer_cast<MultiDeviceInferRequest>(_inferRequest);
    return multiDeviceInfer->GetPerformanceCounts();
}

AsyncInferRequest::~AsyncInferRequest() {
    StopAndWait();
}

}  // namespace MultiDevicePlugin
