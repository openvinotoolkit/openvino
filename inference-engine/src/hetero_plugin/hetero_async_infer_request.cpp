// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>
#include <memory>
#include "hetero_async_infer_request.hpp"

using namespace HeteroPlugin;
using namespace InferenceEngine;

HeteroAsyncInferRequest::HeteroAsyncInferRequest(const IInferRequestInternal::Ptr&  request,
                                                 const ITaskExecutor::Ptr&          taskExecutor,
                                                 const ITaskExecutor::Ptr&          callbackExecutor) :
    AsyncInferRequestThreadSafeDefault(request, taskExecutor, callbackExecutor),
    _heteroInferRequest(std::static_pointer_cast<HeteroInferRequest>(request)) {
    _pipeline.clear();
    for (std::size_t requestId = 0; requestId < _heteroInferRequest->_inferRequests.size(); ++requestId) {
        struct RequestExecutor : ITaskExecutor {
            explicit RequestExecutor(SoIInferRequestInternal & inferRequest) : _inferRequest(inferRequest) {
                _inferRequest->SetCallback(
                [this] (std::exception_ptr exceptionPtr) mutable {
                    _exceptionPtr = exceptionPtr;
                    auto capturedTask = std::move(_task);
                    capturedTask();
                });
            }
            void run(Task task) override {
                _task = std::move(task);
                _inferRequest->StartAsync();
            };
            SoIInferRequestInternal &  _inferRequest;
            std::exception_ptr         _exceptionPtr;
            Task                       _task;
        };

        auto requestExecutor = std::make_shared<RequestExecutor>(_heteroInferRequest->_inferRequests[requestId]._request);
        _pipeline.emplace_back(requestExecutor, [requestExecutor] {
            if (nullptr != requestExecutor->_exceptionPtr) {
                std::rethrow_exception(requestExecutor->_exceptionPtr);
            }
        });
    }
}

void HeteroAsyncInferRequest::StartAsync_ThreadUnsafe() {
    _heteroInferRequest->updateInOutIfNeeded();
    RunFirstStage(_pipeline.begin(), _pipeline.end());
}

StatusCode HeteroAsyncInferRequest::Wait(int64_t millis_timeout) {
    auto waitStatus = StatusCode::OK;
    try {
        waitStatus = AsyncInferRequestThreadSafeDefault::Wait(millis_timeout);
    } catch(...) {
        for (auto&& requestDesc : _heteroInferRequest->_inferRequests) {
            requestDesc._request->Wait(InferRequest::RESULT_READY);
        }
        throw;
    }
    return waitStatus;
}

HeteroAsyncInferRequest::~HeteroAsyncInferRequest() {
    StopAndWait();
}
