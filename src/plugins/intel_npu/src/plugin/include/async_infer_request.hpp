// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/iasync_infer_request.hpp"

namespace intel_npu {

class AsyncInferRequest final : public ov::IAsyncInferRequest {
public:
    explicit AsyncInferRequest(const std::shared_ptr<ov::IInferRequest>& inferRequest,
                               const std::shared_ptr<ov::threading::ITaskExecutor>& requestExecutor,
                               const std::shared_ptr<ov::threading::ITaskExecutor>& getResultExecutor,
                               const std::shared_ptr<ov::threading::ITaskExecutor>& callbackExecutor,
                               const std::function<void(void)>& inferAsyncF,
                               const std::function<void(void)>& getResultF);

    AsyncInferRequest(const AsyncInferRequest&) = delete;

    AsyncInferRequest& operator=(const AsyncInferRequest&) = delete;

    ~AsyncInferRequest();

    std::shared_ptr<ov::IInferRequest> get_sync_infer_request() {
        return _syncInferRequest;
    }

private:
    std::shared_ptr<ov::IInferRequest> _syncInferRequest;
    std::shared_ptr<ov::threading::ITaskExecutor> _getResultExecutor;
};

}  // namespace intel_npu
