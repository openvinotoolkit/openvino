// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/async_infer_request.hpp"
#include "intel_gpu/runtime/itt.hpp"

#include "openvino/runtime/threading/cpu_message.hpp"

#include <memory>

namespace ov {
namespace intel_gpu {

AsyncInferRequest::AsyncInferRequest(const std::shared_ptr<SyncInferRequest>& infer_request,
                                     const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
                                     const std::shared_ptr<ov::threading::ITaskExecutor>& wait_executor,
                                     const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor)
    : ov::IAsyncInferRequest(infer_request, task_executor, callback_executor)
    , m_infer_request(infer_request)
    , m_wait_executor(wait_executor) {
    m_infer_request->set_task_executor(task_executor);
    if (infer_request->use_external_queue()) {
        m_pipeline.clear();
        m_pipeline.emplace_back(wait_executor,
                        [this] {
                            OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "AsyncInferRequest::WaitPipeline");
                            m_infer_request->wait_notify();
                        });
    }
    // static_cast<SyncInferRequest*>(infer_request.get())->set_async_request(this);
    m_infer_request->set_async_request(this);
}
void AsyncInferRequest::start_async() {
    if (m_infer_request->use_external_queue()) {
        m_infer_request->setup_stream_graph();
        m_infer_request->enqueue_notify();
    }
    Parent::start_async();
}

void AsyncInferRequest::setSubInferRequest(
    const std::vector<std::shared_ptr<IAsyncInferRequest>>& requests) {
    m_sub_infer_requests = requests;
}

AsyncInferRequest::~AsyncInferRequest() {
    if (m_has_sub_infers) {
        auto message = ov::threading::message_manager();
        message->stop_server_thread();
        m_sub_infer_requests.clear();
    }
    stop_and_wait();
}

}  // namespace intel_gpu
}  // namespace ov
