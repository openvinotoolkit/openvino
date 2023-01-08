// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "async_infer_request.h"
#include <memory>

ov::intel_cpu::AsyncInferRequest::AsyncInferRequest(const InferenceEngine::IInferRequestInternal::Ptr& inferRequest,
                                                    const InferenceEngine::ITaskExecutor::Ptr& taskExecutor,
                                                    const InferenceEngine::ITaskExecutor::Ptr& callbackExecutor)
    : InferenceEngine::AsyncInferRequestThreadSafeDefault(inferRequest, taskExecutor, callbackExecutor) {
    static_cast<InferRequestBase*>(inferRequest.get())->SetAsyncRequest(this);
}

ov::intel_cpu::AsyncInferRequest::~AsyncInferRequest() {
    StopAndWait();
}
