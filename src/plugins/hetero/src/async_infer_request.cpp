// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "async_infer_request.hpp"

struct RequestExecutor : ov::threading::ITaskExecutor {
    explicit RequestExecutor(ov::SoPtr<ov::IAsyncInferRequest>& request) : m_request(request) {
        m_request->set_callback([this](std::exception_ptr exception_ptr) mutable {
            m_exception_ptr = std::move(exception_ptr);
            auto task = std::move(m_task);
            task();
        });
    }
    void run(ov::threading::Task task) override {
        m_task = std::move(task);
        m_request->start_async();
    };
    ov::SoPtr<ov::IAsyncInferRequest>& m_request;
    std::exception_ptr m_exception_ptr;
    ov::threading::Task m_task;
};

ov::hetero::AsyncInferRequest::AsyncInferRequest(const std::shared_ptr<ov::hetero::InferRequest>& request,
                                                 const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
                                                 const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor)
    : ov::IAsyncInferRequest(request, task_executor, callback_executor),
      m_infer_request(std::static_pointer_cast<ov::hetero::InferRequest>(request)) {
    m_pipeline.clear();
    for (auto&& request : m_infer_request->m_subrequests) {
        auto request_executor = std::make_shared<RequestExecutor>(request);
        m_pipeline.emplace_back(request_executor, [request_executor] {
            if (nullptr != request_executor->m_exception_ptr) {
                std::rethrow_exception(request_executor->m_exception_ptr);
            }
        });
    }
}

ov::hetero::AsyncInferRequest::~AsyncInferRequest() {
    ov::IAsyncInferRequest::stop_and_wait();
}

void ov::hetero::AsyncInferRequest::cancel() {
    ov::IAsyncInferRequest::cancel();
    for (auto&& request : m_infer_request->m_subrequests) {
        request->cancel();
    }
}