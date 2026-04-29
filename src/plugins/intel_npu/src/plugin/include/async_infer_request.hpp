// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/runtime/iasync_infer_request.hpp"

namespace intel_npu {

class InferRequest;

class AsyncInferRequest final : public ov::IAsyncInferRequest {
public:
    explicit AsyncInferRequest(const std::shared_ptr<InferRequest>& inferRequest,
                               const std::shared_ptr<ov::threading::ITaskExecutor>& requestExecutor,
                               const std::shared_ptr<ov::threading::ITaskExecutor>& callbackExecutor,
                               const std::shared_ptr<ov::threading::ITaskExecutor>& waitSeqExecutor,
                               std::function<void()> cleanupExecutors);

    AsyncInferRequest(const AsyncInferRequest&) = delete;
    AsyncInferRequest& operator=(const AsyncInferRequest&) = delete;

    ~AsyncInferRequest();

private:
    std::shared_ptr<InferRequest> _inferRequest;
    std::shared_ptr<ov::threading::ITaskExecutor> _waitSeqExecutor;
    std::function<void()> _cleanupExecutors;
};

}  // namespace intel_npu
