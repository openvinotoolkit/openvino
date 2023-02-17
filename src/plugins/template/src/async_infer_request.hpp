// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "infer_request.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/iinfer_request.hpp"

namespace TemplatePlugin {

// ! [async_infer_request:header]
class AsyncInferRequest : public ov::IAsyncInferRequest {
public:
    AsyncInferRequest(const std::shared_ptr<InferRequest>& request,
                      const InferenceEngine::ITaskExecutor::Ptr& task_executor,
                      const InferenceEngine::ITaskExecutor::Ptr& wait_executor,
                      const InferenceEngine::ITaskExecutor::Ptr& callback_executor);

    ~AsyncInferRequest();

private:
    InferenceEngine::ITaskExecutor::Ptr m_wait_executor;
};
// ! [async_infer_request:header]

}  // namespace TemplatePlugin
