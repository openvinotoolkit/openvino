// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "async_infer_request.hpp"

namespace intel_npu {

// clang-format off
AsyncInferRequest::AsyncInferRequest(const std::shared_ptr<ov::IInferRequest>& syncInferRequest,
                                     const std::shared_ptr<ov::threading::ITaskExecutor>& requestExecutor,
                                     const std::shared_ptr<ov::threading::ITaskExecutor>& getResultExecutor,
                                     const std::shared_ptr<ov::threading::ITaskExecutor>& callbackExecutor,
                                     const std::function<void(void)>& inferAsyncF,
                                     const std::function<void(void)>& getResultF)
        : ov::IAsyncInferRequest(syncInferRequest, requestExecutor, callbackExecutor),
          _syncInferRequest(syncInferRequest), _getResultExecutor(getResultExecutor) {
    m_pipeline = {
            {requestExecutor,       inferAsyncF},
            {getResultExecutor,     getResultF}
    };
}
// clang-format on

AsyncInferRequest::~AsyncInferRequest() {
    stop_and_wait();
}

}  // namespace intel_npu
