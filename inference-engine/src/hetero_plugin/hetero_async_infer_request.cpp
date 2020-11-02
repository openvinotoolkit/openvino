// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>
#include <memory>
#include "hetero_async_infer_request.hpp"

using namespace HeteroPlugin;
using namespace InferenceEngine;

HeteroAsyncInferRequest::HeteroAsyncInferRequest(const InferRequestInternal::Ptr& request,
                                                 const ITaskExecutor::Ptr&        taskExecutor,
                                                 const ITaskExecutor::Ptr&        callbackExecutor) :
    AsyncInferRequestThreadSafeDefault(request, taskExecutor, callbackExecutor),
    _heteroInferRequest(std::static_pointer_cast<HeteroInferRequest>(request)),
    _statusCodes{_heteroInferRequest->_inferRequests.size(), StatusCode::OK} {
    _pipeline.clear();
    for (std::size_t requestId = 0; requestId < _heteroInferRequest->_inferRequests.size(); ++requestId) {
        struct RequestExecutor : ITaskExecutor {
            explicit RequestExecutor(InferRequest* inferRequest) : _inferRequest{inferRequest} {
                _inferRequest->SetCompletionCallback<std::function<void(InferRequest, StatusCode)>>(
                [this] (InferRequest, StatusCode sts) mutable {
                    _status = sts;
                    auto capturedTask = std::move(_task);
                    capturedTask();
                });
            }
            void run(Task task) override {
                _task = std::move(task);
                _inferRequest->StartAsync();
            };
            InferRequest*   _inferRequest = nullptr;
            StatusCode      _status = StatusCode::OK;
            Task            _task;
        };

        auto reuestExecutor = std::make_shared<RequestExecutor>(_heteroInferRequest->_inferRequests[requestId]._request.get());
        _pipeline.emplace_back(reuestExecutor, [reuestExecutor] {
            if (StatusCode::OK != reuestExecutor->_status) {
                THROW_IE_EXCEPTION << InferenceEngine::details::as_status << reuestExecutor->_status;
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
            requestDesc._request->Wait(IInferRequest::RESULT_READY);
        }
        throw;
    }
    return waitStatus;
}

HeteroAsyncInferRequest::~HeteroAsyncInferRequest() {
    StopAndWait();
}
