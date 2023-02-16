// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_async_infer_request.h"
#include <memory>

CLDNNPlugin::CLDNNAsyncInferRequest::CLDNNAsyncInferRequest(const InferenceEngine::IInferRequestInternal::Ptr &inferRequest,
                                                            const InferenceEngine::ITaskExecutor::Ptr &taskExecutor,
                                                            const InferenceEngine::ITaskExecutor::Ptr &callbackExecutor)
        : InferenceEngine::AsyncInferRequestThreadSafeDefault(inferRequest, taskExecutor, callbackExecutor)
        { }

void CLDNNPlugin::CLDNNAsyncInferRequest::Infer_ThreadUnsafe() {
    InferUsingAsync();
}

CLDNNPlugin::CLDNNAsyncInferRequest::~CLDNNAsyncInferRequest() {
    StopAndWait();
}
