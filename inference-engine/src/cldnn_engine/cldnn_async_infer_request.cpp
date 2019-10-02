// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_async_infer_request.h"
#include <memory>

CLDNNPlugin::CLDNNAsyncInferRequest::CLDNNAsyncInferRequest(const InferenceEngine::InferRequestInternal::Ptr &inferRequest,
                                                               const InferenceEngine::ITaskExecutor::Ptr &taskExecutor,
                                                               const InferenceEngine::TaskSynchronizer::Ptr &taskSynchronizer,
                                                               const InferenceEngine::ITaskExecutor::Ptr &callbackExecutor)
        : InferenceEngine::AsyncInferRequestThreadSafeDefault(inferRequest, taskExecutor, taskSynchronizer, callbackExecutor)
        { }

CLDNNPlugin::CLDNNAsyncInferRequest::~CLDNNAsyncInferRequest() {
    waitAllAsyncTasks();
}

void CLDNNPlugin::CLDNNAsyncInferRequest::Infer() {
    _callbackManager.disableCallback();
    StartAsync();
    Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
    _callbackManager.enableCallback();
}
