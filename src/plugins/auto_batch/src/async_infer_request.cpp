// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "async_infer_request.hpp"

namespace ov {
namespace autobatch_plugin {

using namespace InferenceEngine;

AutoBatchAsyncInferRequest::AutoBatchAsyncInferRequest(
    const SyncInferRequest::Ptr& inferRequest,
    InferenceEngine::SoIInferRequestInternal& inferRequestWithoutBatch,
    const ITaskExecutor::Ptr& callbackExecutor)
    : AsyncInferRequestThreadSafeDefault(inferRequest, nullptr, callbackExecutor),
      _inferRequestWithoutBatch(inferRequestWithoutBatch),
      _inferRequest{inferRequest} {
    // this executor starts the inference while  the task (checking the result) is passed to the next stage
    struct ThisRequestExecutor : public ITaskExecutor {
        explicit ThisRequestExecutor(AutoBatchAsyncInferRequest* _this_) : _this{_this_} {}
        void run(Task task) override {
            auto& workerInferRequest = _this->_inferRequest->m_batched_request_wrapper;
            std::pair<AutoBatchAsyncInferRequest*, InferenceEngine::Task> t;
            t.first = _this;
            t.second = std::move(task);
            workerInferRequest._tasks.push(t);
            // it is ok to call size() here as the queue only grows (and the bulk removal happens under the mutex)
            const int sz = static_cast<int>(workerInferRequest._tasks.size());
            if (sz == workerInferRequest._batchSize) {
                workerInferRequest._cond.notify_one();
            }
        };
        AutoBatchAsyncInferRequest* _this = nullptr;
    };
    _pipeline = {{/*TaskExecutor*/ std::make_shared<ThisRequestExecutor>(this), /*task*/ [this] {
                      if (this->_inferRequest->m_exceptionPtr)  // if the exception happened in the batch1 fallback
                          std::rethrow_exception(this->_inferRequest->m_exceptionPtr);
                      auto& batchReq = this->_inferRequest->m_batched_request_wrapper;
                      if (batchReq.m_exceptionPtr)  // when the batchN execution failed
                          std::rethrow_exception(batchReq.m_exceptionPtr);
                      // in the case of non-batched execution the blobs were set explicitly
                      if (SyncInferRequest::eExecutionFlavor::BATCH_EXECUTED ==
                          this->_inferRequest->m_batched_request_status)
                          this->_inferRequest->CopyOutputsIfNeeded();
                  }}};
}

std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> AutoBatchAsyncInferRequest::GetPerformanceCounts()
    const {
    CheckState();
    if (SyncInferRequest::eExecutionFlavor::BATCH_EXECUTED == _inferRequest->m_batched_request_status)
        return _inferRequest->m_batched_request_wrapper._inferRequestBatched->GetPerformanceCounts();
    else
        return _inferRequestWithoutBatch->GetPerformanceCounts();
}

void AutoBatchAsyncInferRequest::Infer_ThreadUnsafe() {
    InferUsingAsync();
}

AutoBatchAsyncInferRequest::~AutoBatchAsyncInferRequest() {
    StopAndWait();
}
}  // namespace autobatch_plugin
}  // namespace ov