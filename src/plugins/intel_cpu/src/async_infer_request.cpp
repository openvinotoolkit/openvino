// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "async_infer_request.h"

#include "openvino/runtime/threading/cpu_message.hpp"

ov::intel_cpu::AsyncInferRequest::AsyncInferRequest(
    const std::shared_ptr<IInferRequest>& request,
    const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
    const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor)
    : ov::IAsyncInferRequest(request, task_executor, callback_executor) {
    static_cast<SyncInferRequest*>(request.get())->set_async_request(this);
    m_internal_request = request;
}

ov::intel_cpu::AsyncInferRequest::~AsyncInferRequest() {
    if (m_has_sub_infers) {
        m_sub_infer_requests.clear();
    }
    stop_and_wait();
}

void ov::intel_cpu::AsyncInferRequest::throw_if_canceled() const {
    check_cancelled_state();
}

void ov::intel_cpu::AsyncInferRequest::setSubInferRequest(
    const std::vector<std::shared_ptr<IAsyncInferRequest>>& requests) {
    m_sub_infer_requests = requests;
}

void ov::intel_cpu::AsyncInferRequest::infer() {
    if (m_is_single_thread) {
        m_internal_request->infer();
    } else {
        ov::IAsyncInferRequest::infer();
    }
}
