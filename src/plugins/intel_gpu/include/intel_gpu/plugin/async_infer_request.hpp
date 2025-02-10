// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/iasync_infer_request.hpp"
#include "intel_gpu/plugin/sync_infer_request.hpp"

namespace ov::intel_gpu {

class AsyncInferRequest : public ov::IAsyncInferRequest {
public:
    using Parent = ov::IAsyncInferRequest;
    AsyncInferRequest(const std::shared_ptr<SyncInferRequest>& infer_request,
                      const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
                      const std::shared_ptr<ov::threading::ITaskExecutor>& wait_executor,
                      const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor);

    ~AsyncInferRequest() override;

    void start_async() override;

private:
    std::shared_ptr<SyncInferRequest> m_infer_request;
    std::shared_ptr<ov::threading::ITaskExecutor> m_wait_executor;
};

}  // namespace ov::intel_gpu
