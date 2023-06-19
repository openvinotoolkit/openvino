// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "async_infer_request.hpp"

namespace AutoBatchPlugin {

using namespace InferenceEngine;

AutoBatchAsyncInferRequest::AutoBatchAsyncInferRequest(
    const AutoBatchInferRequest::Ptr& inferRequest,
    InferenceEngine::SoIInferRequestInternal& inferRequestWithoutBatch,
    const ITaskExecutor::Ptr& callbackExecutor)
    : AsyncInferRequestThreadSafeDefault(inferRequest, nullptr, callbackExecutor),
      _inferRequestWithoutBatch(inferRequestWithoutBatch),
      _inferRequest{inferRequest} {
    // this executor starts the inference while  the task (checking the result) is passed to the next stage
    struct ThisRequestExecutor : public ITaskExecutor {
        explicit ThisRequestExecutor(AutoBatchAsyncInferRequest* _this_) : _this{_this_} {}
        void run(Task task) override {
            auto& workerInferRequest = _this->_inferRequest->_myBatchedRequestWrapper;
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
                      if (this->_inferRequest->_exceptionPtr)  // if the exception happened in the batch1 fallback
                          std::rethrow_exception(this->_inferRequest->_exceptionPtr);
                      auto& batchReq = this->_inferRequest->_myBatchedRequestWrapper;
                      if (batchReq._exceptionPtr)  // when the batchN execution failed
                          std::rethrow_exception(batchReq._exceptionPtr);
                      // in the case of non-batched execution the blobs were set explicitly
                      if (AutoBatchInferRequest::eExecutionFlavor::BATCH_EXECUTED ==
                          this->_inferRequest->_wasBatchedRequestUsed)
                          this->_inferRequest->CopyOutputsIfNeeded();
                  }}};
}

std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> AutoBatchAsyncInferRequest::GetPerformanceCounts()
    const {
    CheckState();
    if (AutoBatchInferRequest::eExecutionFlavor::BATCH_EXECUTED == _inferRequest->_wasBatchedRequestUsed)
        return _inferRequest->_myBatchedRequestWrapper._inferRequestBatched->GetPerformanceCounts();
    else
        return _inferRequestWithoutBatch->GetPerformanceCounts();
}

void AutoBatchAsyncInferRequest::Infer_ThreadUnsafe() {
    InferUsingAsync();
}

AutoBatchAsyncInferRequest::~AutoBatchAsyncInferRequest() {
    StopAndWait();
}
}  // namespace AutoBatchPlugin