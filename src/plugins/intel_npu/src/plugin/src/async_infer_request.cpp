// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "async_infer_request.hpp"

#include "intel_npu/common/npu.hpp"

namespace intel_npu {

AsyncInferRequest::AsyncInferRequest(
    const std::shared_ptr<InferRequest>& inferRequest,
    const std::shared_ptr<ov::threading::ITaskExecutor>& requestExecutor,
    const std::shared_ptr<ov::threading::ITaskExecutor>& callbackExecutor,
    const std::shared_ptr<ov::threading::ITaskExecutor>& requestExecutorForSyncRequests)
    : ov::IAsyncInferRequest(inferRequest, requestExecutor, callbackExecutor),
      _inferRequest(inferRequest),
      _requestExecutorForSyncRequests(requestExecutorForSyncRequests) {
    if (_requestExecutorForSyncRequests != nullptr) {
        m_pipeline = {{_requestExecutorForSyncRequests,
                       [this] {
                           _inferRequest->infer_async();
                       }},
                      {requestExecutor, [this] {
                           _inferRequest->get_result();
                       }}};
    }
}

AsyncInferRequest::~AsyncInferRequest() {
    stop_and_wait();
}

}  // namespace intel_npu
