// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/iasync_infer_request.hpp"
#include "template_infer_request.hpp"

namespace TemplatePlugin {

// ! [async_infer_request:header]
class TemplateAsyncInferRequest : public ov::IAsyncInferRequest {
public:
    TemplateAsyncInferRequest(const TemplateInferRequest::Ptr& inferRequest,
                              const InferenceEngine::ITaskExecutor::Ptr& taskExecutor,
                              const InferenceEngine::ITaskExecutor::Ptr& waitExecutor,
                              const InferenceEngine::ITaskExecutor::Ptr& callbackExecutor);

    ~TemplateAsyncInferRequest();

private:
    TemplateInferRequest::Ptr _inferRequest;
    InferenceEngine::ITaskExecutor::Ptr _waitExecutor;
};
// ! [async_infer_request:header]

}  // namespace TemplatePlugin
