// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpp_interfaces/impl/ie_infer_async_request_thread_safe_default.hpp>

#include "template_infer_request.hpp"

namespace TemplatePlugin {

// ! [async_infer_request:header]
class TemplateAsyncInferRequest : public InferenceEngine::AsyncInferRequestThreadSafeDefault {
public:
    TemplateAsyncInferRequest(const TemplateInferRequest::Ptr& inferRequest, const InferenceEngine::ITaskExecutor::Ptr& taskExecutor,
                              const InferenceEngine::ITaskExecutor::Ptr& waitExecutor, const InferenceEngine::ITaskExecutor::Ptr& callbackExecutor);

    ~TemplateAsyncInferRequest();

private:
    TemplateInferRequest::Ptr _inferRequest;
    InferenceEngine::ITaskExecutor::Ptr _waitExecutor;
};
// ! [async_infer_request:header]

}  // namespace TemplatePlugin
