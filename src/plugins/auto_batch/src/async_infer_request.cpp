// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "async_infer_request.hpp"

namespace ov {
namespace autobatch_plugin {

AsyncInferRequest::AsyncInferRequest(const SyncInferRequest::Ptr& inferRequest,
                                     InferenceEngine::SoIInferRequestInternal& inferRequestWithoutBatch,
                                     const InferenceEngine::ITaskExecutor::Ptr& callbackExecutor)
    : AsyncInferRequestThreadSafeDefault(inferRequest, nullptr, callbackExecutor),
      m_infer_request_without_batch(inferRequestWithoutBatch),
      m_sync_infer_request{inferRequest} {
    // this executor starts the inference while  the task (checking the result) is passed to the next stage
    struct ThisRequestExecutor : public InferenceEngine::ITaskExecutor {
        explicit ThisRequestExecutor(AsyncInferRequest* _this_) : _this{_this_} {}
        void run(InferenceEngine::Task task) override {
            auto& workerInferRequest = _this->m_sync_infer_request->m_batched_request_wrapper;
            std::pair<AsyncInferRequest*, InferenceEngine::Task> t;
            t.first = _this;
            t.second = std::move(task);
            workerInferRequest._tasks.push(t);
            // it is ok to call size() here as the queue only grows (and the bulk removal happens under the mutex)
            const int sz = static_cast<int>(workerInferRequest._tasks.size());
            if (sz == workerInferRequest._batchSize) {
                workerInferRequest._cond.notify_one();
            }
        };
        AsyncInferRequest* _this = nullptr;
    };
    _pipeline = {
        {/*TaskExecutor*/ std::make_shared<ThisRequestExecutor>(this), /*task*/ [this] {
             if (this->m_sync_infer_request->m_exceptionPtr)  // if the exception happened in the batch1 fallback
                 std::rethrow_exception(this->m_sync_infer_request->m_exceptionPtr);
             auto& batchReq = this->m_sync_infer_request->m_batched_request_wrapper;
             if (batchReq.m_exceptionPtr)  // when the batchN execution failed
                 std::rethrow_exception(batchReq.m_exceptionPtr);
             // in the case of non-batched execution the blobs were set explicitly
             if (SyncInferRequest::eExecutionFlavor::BATCH_EXECUTED ==
                 this->m_sync_infer_request->m_batched_request_status)
                 this->m_sync_infer_request->CopyOutputsIfNeeded();
         }}};
}

std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> AsyncInferRequest::GetPerformanceCounts() const {
    CheckState();
    if (SyncInferRequest::eExecutionFlavor::BATCH_EXECUTED == m_sync_infer_request->m_batched_request_status)
        return m_sync_infer_request->m_batched_request_wrapper._inferRequestBatched->GetPerformanceCounts();
    else
        return m_infer_request_without_batch->GetPerformanceCounts();
}

void AsyncInferRequest::Infer_ThreadUnsafe() {
    InferUsingAsync();
}

AsyncInferRequest::~AsyncInferRequest() {
    StopAndWait();
}
}  // namespace autobatch_plugin
}  // namespace ov