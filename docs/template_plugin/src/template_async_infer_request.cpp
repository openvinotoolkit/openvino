// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>

#include <ie_profiling.hpp>

#include "template_async_infer_request.hpp"
#include "template_executable_network.hpp"

using namespace TemplatePlugin;

// ! [async_infer_request:ctor]
TemplateAsyncInferRequest::TemplateAsyncInferRequest(
    const TemplateInferRequest::Ptr&           inferRequest,
    const InferenceEngine::ITaskExecutor::Ptr& cpuTaskExecutor,
    const InferenceEngine::ITaskExecutor::Ptr& waitExecutor,
    const InferenceEngine::ITaskExecutor::Ptr& callbackExecutor) :
    AsyncInferRequestThreadSafeDefault(inferRequest, cpuTaskExecutor, callbackExecutor),
    _inferRequest(inferRequest), _waitExecutor(waitExecutor) {
    constexpr const auto remoteDevice = false;
    // By default single stage pipeline is created.
    // This stage executes InferRequest::Infer() using cpuTaskExecutor.
    // But if remote asynchronous device is used the pipeline can by splitted tasks that are executed by cpuTaskExecutor
    // and waiting tasks. Waiting tasks can lock execution thread so they use separate threads from other executor.
    if (remoteDevice) {
        _pipeline = {
            {cpuTaskExecutor, [this] {
                IE_PROFILING_AUTO_SCOPE(PreprocessingAndStartPipeline)
                _inferRequest->inferPreprocess();
                _inferRequest->startPipeline();
            }},
            {_waitExecutor, [this] {
                IE_PROFILING_AUTO_SCOPE(WaitPipeline)
                _inferRequest->waitPipeline();
            }},
            {cpuTaskExecutor, [this] {
                IE_PROFILING_AUTO_SCOPE(Postprocessing)
                _inferRequest->inferPostprocess();
            }}
        };
    }
}
// ! [async_infer_request:ctor]

// ! [async_infer_request:dtor]
TemplateAsyncInferRequest::~TemplateAsyncInferRequest() {
    InferenceEngine::AsyncInferRequestThreadSafeDefault::StopAndWait();
}
// ! [async_infer_request:dtor]
