// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "async_infer_request.h"

ov::intel_cpu::AsyncInferRequest::AsyncInferRequest(
    const std::shared_ptr<IInferRequest>& request,
    const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
    const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor)
    : ov::IAsyncInferRequest(request, task_executor, callback_executor) {
    static_cast<SyncInferRequest*>(request.get())->set_async_request(this);
}

ov::intel_cpu::AsyncInferRequest::~AsyncInferRequest() {
    stop_and_wait();
}

void ov::intel_cpu::AsyncInferRequest::throw_if_canceled() const {
    check_cancelled_state();
}
