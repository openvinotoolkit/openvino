// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/async_infer_request.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include <memory>

namespace ov::intel_gpu {

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
}
void AsyncInferRequest::start_async() {
    if (m_infer_request->use_external_queue()) {
        m_infer_request->setup_stream_graph();
        m_infer_request->enqueue_notify();
    }
    Parent::start_async();
}

AsyncInferRequest::~AsyncInferRequest() {
    stop_and_wait();
}

}  // namespace ov::intel_gpu
