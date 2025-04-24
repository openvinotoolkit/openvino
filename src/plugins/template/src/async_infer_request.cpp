// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "async_infer_request.hpp"

#include "itt.hpp"
#include "openvino/runtime/iinfer_request.hpp"
#include "sync_infer_request.hpp"

// ! [async_infer_request:ctor]
ov::template_plugin::AsyncInferRequest::AsyncInferRequest(
    const std::shared_ptr<ov::template_plugin::InferRequest>& request,
    const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
    const std::shared_ptr<ov::threading::ITaskExecutor>& wait_executor,
    const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor)
    : ov::IAsyncInferRequest(request, task_executor, callback_executor),
      m_wait_executor(wait_executor) {
    // In current implementation we have CPU only tasks and no needs in 2 executors
    // So, by default single stage pipeline is created.
    // This stage executes InferRequest::infer() using cpuTaskExecutor.
    // But if remote asynchronous device is used the pipeline can by splitted tasks that are executed by cpuTaskExecutor
    // and waiting tasks. Waiting tasks can lock execution thread so they use separate threads from other executor.
    constexpr const auto remoteDevice = false;

    m_cancel_callback = [request] {
        request->cancel();
    };
    if (remoteDevice) {
        m_pipeline = {{task_executor,
                       [this, request] {
                           OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin,
                                              "TemplatePlugin::AsyncInferRequest::infer_preprocess_and_start_pipeline");
                           request->infer_preprocess();
                           request->start_pipeline();
                       }},
                      {m_wait_executor,
                       [this, request] {
                           OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin,
                                              "TemplatePlugin::AsyncInferRequest::wait_pipeline");
                           request->wait_pipeline();
                       }},
                      {task_executor, [this, request] {
                           OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin,
                                              "TemplatePlugin::AsyncInferRequest::infer_postprocess");
                           request->infer_postprocess();
                       }}};
    }
}
// ! [async_infer_request:ctor]

// ! [async_infer_request:dtor]
ov::template_plugin::AsyncInferRequest::~AsyncInferRequest() {
    ov::IAsyncInferRequest::stop_and_wait();
}
// ! [async_infer_request:dtor]

// ! [async_infer_request:cancel]
void ov::template_plugin::AsyncInferRequest::cancel() {
    ov::IAsyncInferRequest::cancel();
    m_cancel_callback();
}
// ! [async_infer_request:cancel]
