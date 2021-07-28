// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "template_async_infer_request.hpp"

#include "template_itt.hpp"

using namespace TemplatePlugin;

// ! [async_infer_request:ctor]
TemplateAsyncInferRequest::TemplateAsyncInferRequest(const TemplateInferRequest::Ptr& inferRequest, const InferenceEngine::ITaskExecutor::Ptr& cpuTaskExecutor,
                                                     const InferenceEngine::ITaskExecutor::Ptr& waitExecutor,
                                                     const InferenceEngine::ITaskExecutor::Ptr& callbackExecutor)
    : AsyncInferRequestThreadSafeDefault(inferRequest, cpuTaskExecutor, callbackExecutor), _inferRequest(inferRequest), _waitExecutor(waitExecutor) {
    // In current implementation we have CPU only tasks and no needs in 2 executors
    // So, by default single stage pipeline is created.
    // This stage executes InferRequest::Infer() using cpuTaskExecutor.
    // But if remote asynchronous device is used the pipeline can by splitted tasks that are executed by cpuTaskExecutor
    // and waiting tasks. Waiting tasks can lock execution thread so they use separate threads from other executor.
    constexpr const auto remoteDevice = false;

    if (remoteDevice) {
        _pipeline = {{cpuTaskExecutor,
                      [this] {
                          OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, "TemplateAsyncInferRequest::PreprocessingAndStartPipeline");
                          _inferRequest->inferPreprocess();
                          _inferRequest->startPipeline();
                      }},
                     {_waitExecutor,
                      [this] {
                          OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, "TemplateAsyncInferRequest::WaitPipeline");
                          _inferRequest->waitPipeline();
                      }},
                     {cpuTaskExecutor, [this] {
                          OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, "TemplateAsyncInferRequest::Postprocessing");
                          _inferRequest->inferPostprocess();
                      }}};
    }
}
// ! [async_infer_request:ctor]

// ! [async_infer_request:dtor]
TemplateAsyncInferRequest::~TemplateAsyncInferRequest() {
    InferenceEngine::AsyncInferRequestThreadSafeDefault::StopAndWait();
}
// ! [async_infer_request:dtor]
