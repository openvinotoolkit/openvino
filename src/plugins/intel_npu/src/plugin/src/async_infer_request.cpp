// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "async_infer_request.hpp"

#include "intel_npu/common/npu.hpp"

namespace intel_npu {

AsyncInferRequest::AsyncInferRequest(const std::shared_ptr<InferRequest>& syncInferRequest,
                                     const std::shared_ptr<ov::threading::ITaskExecutor>& requestExecutor,
                                     const std::shared_ptr<ov::threading::ITaskExecutor>& callbackExecutor)
    : ov::IAsyncInferRequest(syncInferRequest, requestExecutor, callbackExecutor),
      _syncInferRequest(syncInferRequest) {
    m_pipeline = {{m_request_executor, [this] {
                       _syncInferRequest->get_result();
                   }}};
}

void AsyncInferRequest::start_async_thread_unsafe() {
    _syncInferRequest->infer_async();

    run_first_stage(m_pipeline.begin(), m_pipeline.end(), m_callback_executor);
}

AsyncInferRequest::~AsyncInferRequest() {
    stop_and_wait();
}

}  // namespace intel_npu
