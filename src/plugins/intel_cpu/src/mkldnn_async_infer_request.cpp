// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_async_infer_request.h"
#include <memory>

ov::intel_cpu::MKLDNNAsyncInferRequest::MKLDNNAsyncInferRequest(const InferenceEngine::IInferRequestInternal::Ptr& inferRequest,
                                                               const InferenceEngine::ITaskExecutor::Ptr& taskExecutor,
                                                               const InferenceEngine::ITaskExecutor::Ptr& callbackExecutor)
    : InferenceEngine::AsyncInferRequestThreadSafeDefault(inferRequest, taskExecutor, callbackExecutor) {
    static_cast<MKLDNNInferRequestBase*>(inferRequest.get())->SetAsyncRequest(this);
}

ov::intel_cpu::MKLDNNAsyncInferRequest::~MKLDNNAsyncInferRequest() {
    StopAndWait();
}
