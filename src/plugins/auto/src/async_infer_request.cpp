// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "async_infer_request.hpp"
ov::auto_plugin::AsyncInferRequest::AsyncInferRequest(const Schedule::Ptr& schedule,
                                                      const std::shared_ptr<ov::auto_plugin::InferRequest>& request,
                                                      const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor) :
                                                      IAsyncInferRequest(request, nullptr, callback_executor),
                                                      m_schedule(schedule),
                                                      m_inferrequest(request) {
    auto pipeline = m_schedule->get_async_pipeline(m_inferrequest, &m_worker_inferrequest);
    if (pipeline.size() > 0) {
        m_pipeline = std::move(pipeline);
    }
}

std::vector<ov::ProfilingInfo> ov::auto_plugin::AsyncInferRequest::get_profiling_info() const {
    check_state();
    auto scheduled_request = std::dynamic_pointer_cast<InferRequest>(m_inferrequest);
    return scheduled_request->get_profiling_info();
}

void ov::auto_plugin::AsyncInferRequest::infer_thread_unsafe() {
    start_async_thread_unsafe();
}

ov::auto_plugin::AsyncInferRequest::~AsyncInferRequest() {
    stop_and_wait();
}