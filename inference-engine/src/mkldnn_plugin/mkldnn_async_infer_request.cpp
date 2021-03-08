// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_async_infer_request.h"
#include <memory>

MKLDNNPlugin::MKLDNNAsyncInferRequest::MKLDNNAsyncInferRequest(const InferenceEngine::InferRequestInternal::Ptr& inferRequest,
                                                               const InferenceEngine::ITaskExecutor::Ptr& taskExecutor,
                                                               const InferenceEngine::ITaskExecutor::Ptr& callbackExecutor)
        : InferenceEngine::AsyncInferRequestThreadSafeDefault(inferRequest, taskExecutor, callbackExecutor) {
    static_cast<MKLDNNInferRequest*>(inferRequest.get())->SetAsyncRequest(this);
}

void MKLDNNPlugin::MKLDNNAsyncInferRequest::Infer_ThreadUnsafe() {
    InferUsingAsync();
}

MKLDNNPlugin::MKLDNNAsyncInferRequest::~MKLDNNAsyncInferRequest() {
    StopAndWait();
}
