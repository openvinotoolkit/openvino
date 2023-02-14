// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "async_infer_request.hpp"

#include "infer_request.hpp"
#include "openvino/runtime/iinfer_request.hpp"
#include "template_itt.hpp"

// ! [async_infer_request:ctor]
TemplatePlugin::AsyncInferRequest::AsyncInferRequest(const std::shared_ptr<TemplatePlugin::InferRequest>& inferRequest,
                                                     const InferenceEngine::ITaskExecutor::Ptr& cpuTaskExecutor,
                                                     const InferenceEngine::ITaskExecutor::Ptr& waitExecutor,
                                                     const InferenceEngine::ITaskExecutor::Ptr& callbackExecutor)
    : ov::IAsyncInferRequest(inferRequest, cpuTaskExecutor, callbackExecutor),
      m_wait_executor(waitExecutor) {
    // In current implementation we have CPU only tasks and no needs in 2 executors
    // So, by default single stage pipeline is created.
    // This stage executes InferRequest::infer() using cpuTaskExecutor.
    // But if remote asynchronous device is used the pipeline can by splitted tasks that are executed by cpuTaskExecutor
    // and waiting tasks. Waiting tasks can lock execution thread so they use separate threads from other executor.
    constexpr const auto remoteDevice = false;

    if (remoteDevice) {
        m_pipeline = {
            {cpuTaskExecutor,
             [this] {
                 OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin,
                                    "TemplatePlugin::AsyncInferRequest::PreprocessingAndStartPipeline");
                 get_template_infer_request()->infer_preprocess();
                 get_template_infer_request()->start_pipeline();
             }},
            {m_wait_executor,
             [this] {
                 OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, "TemplatePlugin::AsyncInferRequest::WaitPipeline");
                 get_template_infer_request()->wait_pipeline();
             }},
            {cpuTaskExecutor, [this] {
                 OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, "TemplatePlugin::AsyncInferRequest::Postprocessing");
                 get_template_infer_request()->infer_postprocess();
             }}};
    }
}
// ! [async_infer_request:ctor]

// ! [async_infer_request:dtor]
TemplatePlugin::AsyncInferRequest::~AsyncInferRequest() {
    ov::IAsyncInferRequest::stop_and_wait();
}
// ! [async_infer_request:dtor]

std::shared_ptr<TemplatePlugin::InferRequest> TemplatePlugin::AsyncInferRequest::get_template_infer_request() {
    auto infer_request = get_sync_infer_request();

    OPENVINO_ASSERT(infer_request);
    auto template_infer_request = std::static_pointer_cast<TemplatePlugin::InferRequest>(infer_request);
    OPENVINO_ASSERT(template_infer_request);
    return template_infer_request;
}
