// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "async_infer_request.hpp"

// #include "itt.hpp"
#include "openvino/runtime/iinfer_request.hpp"
#include "sync_infer_request.hpp"


struct RequestExecutor : ov::threading::ITaskExecutor {
    explicit RequestExecutor(ov::SoPtr<ov::IAsyncInferRequest>& inferRequest) : _inferRequest(inferRequest) {
        _inferRequest->set_callback([this](std::exception_ptr exceptionPtr) mutable {
            _exceptionPtr = exceptionPtr;
            auto capturedTask = std::move(_task);
            capturedTask();
        });
    }
    void run(ov::threading::Task task) override {
        _task = std::move(task);
        _inferRequest->start_async();
    };
    ov::SoPtr<ov::IAsyncInferRequest>& _inferRequest;
    std::exception_ptr _exceptionPtr;
    ov::threading::Task _task;
};

ov::hetero::AsyncInferRequest::AsyncInferRequest(
    const std::shared_ptr<ov::hetero::InferRequest>& request,
    const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
    const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor)
    : ov::IAsyncInferRequest(request, task_executor, callback_executor),
      _heteroInferRequest(std::static_pointer_cast<ov::hetero::InferRequest>(request)) {


    // In current implementation we have CPU only tasks and no needs in 2 executors
    // So, by default single stage pipeline is created.
    // This stage executes InferRequest::infer() using cpuTaskExecutor.
    // But if remote asynchronous device is used the pipeline can by splitted tasks that are executed by cpuTaskExecutor
    // and waiting tasks. Waiting tasks can lock execution thread so they use separate threads from other executor.
    

    m_cancel_callback = [request] {
        request->cancel();
    };

    m_pipeline.clear();
    for (std::size_t requestId = 0; requestId < _heteroInferRequest->m_infer_requests.size(); ++requestId) {
        auto requestExecutor =
            std::make_shared<RequestExecutor>(_heteroInferRequest->m_infer_requests[requestId]._request);
        m_pipeline.emplace_back(requestExecutor, [requestExecutor] {
            if (nullptr != requestExecutor->_exceptionPtr) {
                std::rethrow_exception(requestExecutor->_exceptionPtr);
            }
        });
    }


    // constexpr const auto remoteDevice = false;
    // if (remoteDevice) {
    //     m_pipeline = {{task_executor,
    //                    [this, request] {
    //                        OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin,
    //                                           "TemplatePlugin::AsyncInferRequest::infer_preprocess_and_start_pipeline");
    //                        request->infer_preprocess();
    //                        request->start_pipeline();
    //                    }},
    //                   {m_wait_executor,
    //                    [this, request] {
    //                        OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin,
    //                                           "TemplatePlugin::AsyncInferRequest::wait_pipeline");
    //                        request->wait_pipeline();
    //                    }},
    //                   {task_executor, [this, request] {
    //                        OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin,
    //                                           "TemplatePlugin::AsyncInferRequest::infer_postprocess");
    //                        request->infer_postprocess();
    //                    }}};
    // }
}

ov::hetero::AsyncInferRequest::~AsyncInferRequest() {
    ov::IAsyncInferRequest::stop_and_wait();
}

void ov::hetero::AsyncInferRequest::cancel() {
    ov::IAsyncInferRequest::cancel();
    m_cancel_callback();
}
