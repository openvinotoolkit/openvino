// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/iasync_infer_request.hpp"
#include "intel_gpu/plugin/sync_infer_request.hpp"
#include <string>
#include <map>

namespace ov {
namespace intel_gpu {

class AsyncInferRequest : public ov::IAsyncInferRequest {
public:
    using Parent = ov::IAsyncInferRequest;
    AsyncInferRequest(const std::shared_ptr<SyncInferRequest>& infer_request,
                      const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
                      const std::shared_ptr<ov::threading::ITaskExecutor>& wait_executor,
                      const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor);

    ~AsyncInferRequest() override;
    void setSubInferRequest(const std::vector<std::shared_ptr<IAsyncInferRequest>>& requests);

    std::vector<std::shared_ptr<ov::IAsyncInferRequest>> getSubInferRequest() const {
        return m_sub_infer_requests;
    }

    void setSubInfer(bool has_sub_infer) {
        m_has_sub_infers = has_sub_infer;
    }
    void start_async() override;
    bool m_has_sub_infers = false;

private:
    std::shared_ptr<SyncInferRequest> m_infer_request;
    std::vector<std::shared_ptr<ov::IAsyncInferRequest>> m_sub_infer_requests;
    std::shared_ptr<ov::threading::ITaskExecutor> m_wait_executor;
};

}  // namespace intel_gpu
}  // namespace ov
