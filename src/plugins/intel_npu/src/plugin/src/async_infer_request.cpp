// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "async_infer_request.hpp"

#include "intel_npu/common/npu.hpp"

namespace intel_npu {

AsyncInferRequest::AsyncInferRequest(const std::shared_ptr<InferRequest>& inferRequest,
                                     const std::shared_ptr<ov::threading::ITaskExecutor>& requestExecutor,
                                     const std::shared_ptr<ov::threading::ITaskExecutor>& callbackExecutor,
                                     const std::shared_ptr<ov::threading::ITaskExecutor>& waitSeqExecutor,
                                     std::function<void()> cleanupExecutors)
    : ov::IAsyncInferRequest(inferRequest, requestExecutor, callbackExecutor),
      _inferRequest(inferRequest),
      _waitSeqExecutor(waitSeqExecutor),
      _cleanupExecutors(cleanupExecutors) {
    if (_waitSeqExecutor) {
        m_pipeline = {{requestExecutor,
                       [this] {
                           _inferRequest->infer_async();
                       }},
                      {_waitSeqExecutor, [this] {
                           _inferRequest->get_result();
                       }}};
    }
}

AsyncInferRequest::~AsyncInferRequest() {
    stop_and_wait();

    if (_cleanupExecutors) {
        _cleanupExecutors();
    }
}

}  // namespace intel_npu
