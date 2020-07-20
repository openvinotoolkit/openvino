// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>

#include "template_async_infer_request.hpp"
#include "template_executable_network.hpp"
#include <openvino/itt.hpp>

using namespace TemplatePlugin;
using namespace openvino;

// ! [async_infer_request:ctor]
TemplateAsyncInferRequest::TemplateAsyncInferRequest(
    const TemplateInferRequest::Ptr&           inferRequest,
    const InferenceEngine::ITaskExecutor::Ptr& cpuTaskExecutor,
    const InferenceEngine::ITaskExecutor::Ptr& waitExecutor,
    const InferenceEngine::ITaskExecutor::Ptr& callbackExecutor) :
    AsyncInferRequestThreadSafeDefault(inferRequest, cpuTaskExecutor, callbackExecutor),
    _inferRequest(inferRequest), _waitExecutor(waitExecutor) {
    _pipeline = {
        {cpuTaskExecutor, [this] {
            OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, "PreprocessingAndStartPipeline");
            _inferRequest->inferPreprocess();
            _inferRequest->startPipeline();
        }},
        {_waitExecutor, [this] {
            OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, "WaitPipeline");
            _inferRequest->waitPipeline();
        }},
        {cpuTaskExecutor, [this] {
            OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, "Postprocessing");
            _inferRequest->inferPostprocess();
        }}
    };
}
// ! [async_infer_request:ctor]

// ! [async_infer_request:dtor]
TemplateAsyncInferRequest::~TemplateAsyncInferRequest() {
    InferenceEngine::AsyncInferRequestThreadSafeDefault::StopAndWait();
}
// ! [async_infer_request:dtor]
