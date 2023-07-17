// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "async_infer_request.hpp"

namespace ov {
namespace autobatch_plugin {

AsyncInferRequest::AsyncInferRequest(const std::shared_ptr<SyncInferRequest>& request,
                                     const ov::SoPtr<ov::IAsyncInferRequest>& request_without_batch,
                                     const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor)
    : ov::IAsyncInferRequest(request, nullptr, callback_executor),
      m_sync_request(request),
      m_request_without_batch(request_without_batch) {
    // this executor starts the inference while  the task (checking the result) is passed to the next stage
    struct ThisRequestExecutor : public ov::threading::ITaskExecutor {
        explicit ThisRequestExecutor(AsyncInferRequest* _this_) : _this{_this_} {}
        void run(ov::threading::Task task) override {
            auto workerInferRequest = _this->m_sync_request->m_batched_request_wrapper;
            std::pair<AsyncInferRequest*, ov::threading::Task> t;
            t.first = _this;
            t.second = std::move(task);
            workerInferRequest->_tasks.push(t);
            // it is ok to call size() here as the queue only grows (and the bulk removal happens under the mutex)
            const int sz = static_cast<int>(workerInferRequest->_tasks.size());
            if (sz == workerInferRequest->_batch_size) {
                workerInferRequest->_cond.notify_one();
            }
        };
        AsyncInferRequest* _this = nullptr;
    };
    m_pipeline = {{/*TaskExecutor*/ std::make_shared<ThisRequestExecutor>(this), /*task*/ [this] {
                       if (this->m_sync_request->m_exception_ptr)  // if the exception happened in the batch1 fallback
                           std::rethrow_exception(this->m_sync_request->m_exception_ptr);
                       auto batchReq = this->m_sync_request->m_batched_request_wrapper;
                       if (batchReq->_exception_ptr)  // when the batchN execution failed
                           std::rethrow_exception(batchReq->_exception_ptr);
                       // in the case of non-batched execution the tensors were set explicitly
                       if (SyncInferRequest::eExecutionFlavor::BATCH_EXECUTED ==
                           this->m_sync_request->m_batched_request_status) {
                           this->m_sync_request->copy_outputs_if_needed();
                       }
                   }}};
}

std::vector<ov::ProfilingInfo> AsyncInferRequest::get_profiling_info() const {
    check_state();
    if (SyncInferRequest::eExecutionFlavor::BATCH_EXECUTED == m_sync_request->m_batched_request_status)
        return m_sync_request->get_profiling_info();
    else
        return m_request_without_batch->get_profiling_info();
}

void AsyncInferRequest::infer_thread_unsafe() {
    start_async_thread_unsafe();
}

AsyncInferRequest::~AsyncInferRequest() {
    stop_and_wait();
}
}  // namespace autobatch_plugin
}  // namespace ov