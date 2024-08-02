// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "async_infer_request.hpp"

namespace intel_npu {

// clang-format off
AsyncInferRequest::AsyncInferRequest(const std::shared_ptr<SyncInferRequest>& syncInferRequest,
                                     const std::shared_ptr<ov::threading::ITaskExecutor>& requestExecutor,
                                     const std::shared_ptr<ov::threading::ITaskExecutor>& getResultExecutor,
                                     const std::shared_ptr<ov::threading::ITaskExecutor>& callbackExecutor)
        : ov::IAsyncInferRequest(syncInferRequest, requestExecutor, callbackExecutor),
          _syncInferRequest(syncInferRequest), _getResultExecutor(getResultExecutor) {
    m_pipeline = {
            {requestExecutor,       [this] { _syncInferRequest->infer_async(); }},
            {getResultExecutor,     [this] { _syncInferRequest->get_result(); }}
    };
}
// clang-format on

AsyncInferRequest::~AsyncInferRequest() {
    stop_and_wait();
}

}  // namespace intel_npu
