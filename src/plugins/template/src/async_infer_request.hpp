// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/iinfer_request.hpp"
#include "sync_infer_request.hpp"

namespace ov {
namespace template_plugin {

// ! [async_infer_request:header]
class AsyncInferRequest : public ov::IAsyncInferRequest {
public:
    AsyncInferRequest(const std::shared_ptr<InferRequest>& request,
                      const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
                      const std::shared_ptr<ov::threading::ITaskExecutor>& wait_executor,
                      const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor);

    ~AsyncInferRequest();
    void cancel() override;

private:
    std::function<void()> m_cancel_callback;
    std::shared_ptr<ov::threading::ITaskExecutor> m_wait_executor;
};
// ! [async_infer_request:header]

}  // namespace template_plugin
}  // namespace ov
