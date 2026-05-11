// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "async_infer_request.hpp"

#include "intel_npu/common/npu.hpp"

namespace intel_npu {

AsyncInferRequest::AsyncInferRequest(const std::shared_ptr<InferRequest>& inferRequest,
                                     const std::shared_ptr<ov::threading::ITaskExecutor>& requestExecutor,
                                     const std::shared_ptr<ov::threading::ITaskExecutor>& resultExecutor,
                                     const std::shared_ptr<ov::threading::ITaskExecutor>& callbackExecutor)
    : ov::IAsyncInferRequest(inferRequest, requestExecutor, callbackExecutor),
      _inferRequest(inferRequest),
      _resultExecutor(resultExecutor) {
    if (_resultExecutor) {
        m_pipeline = {{requestExecutor,
                       [this] {
                           _inferRequest->infer_async();
                       }},
                      {_resultExecutor, [this] {
                           _inferRequest->get_result();
                       }}};
    }
}

AsyncInferRequest::~AsyncInferRequest() {
    stop_and_wait();
}

}  // namespace intel_npu
