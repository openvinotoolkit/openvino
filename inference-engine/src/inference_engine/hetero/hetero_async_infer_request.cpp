// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cassert>
#include "hetero_async_infer_request.hpp"
#include <ie_util_internal.hpp>
#include <ie_profiling.hpp>

using namespace HeteroPlugin;
using namespace InferenceEngine;

HeteroAsyncInferRequest::HeteroAsyncInferRequest(HeteroInferRequest::Ptr request,
                                                 const ITaskExecutor::Ptr &taskExecutor,
                                                 const TaskSynchronizer::Ptr &taskSynchronizer,
                                                 const ITaskExecutor::Ptr &callbackExecutor)
        : AsyncInferRequestThreadSafeDefault(request, taskExecutor, taskSynchronizer, callbackExecutor),
          _heteroInferRequest(request) {
    _heteroInferRequest->setCallbackSequence();

    std::function<void(InferRequest, StatusCode)> f =
        [&](InferRequest /*request*/, StatusCode /*sts*/) {
            setIsRequestBusy(false);
        };

    _heteroInferRequest->setCallbackForLastRequest(f);
}

void HeteroAsyncInferRequest::StartAsync() {
    IE_PROFILING_AUTO_SCOPE(Hetero_Async)
    if (isRequestBusy())
        THROW_IE_EXCEPTION << InferenceEngine::details::as_status << StatusCode::REQUEST_BUSY << REQUEST_BUSY_str;
    setIsRequestBusy(true);
    _heteroInferRequest->updateInOutIfNeeded();
    _heteroInferRequest->startFirstAsyncRequest();
}

InferenceEngine::StatusCode HeteroAsyncInferRequest::Wait(int64_t millis_timeout) {
    auto sts = _heteroInferRequest->waitAllRequests(millis_timeout);
    if (sts != StatusCode::RESULT_NOT_READY && sts != StatusCode::REQUEST_BUSY) {
        setIsRequestBusy(false);
    }
    return sts;
}

void HeteroAsyncInferRequest::SetCompletionCallback(IInferRequest::CompletionCallback callback) {
    AsyncInferRequestThreadSafeDefault::SetCompletionCallback(callback);

    std::function<void(InferRequest, StatusCode)> f =
            [&](InferRequest /*request*/, StatusCode sts) {
                setIsRequestBusy(false);
                _callbackManager.set_requestStatus(sts);
                _callbackManager.runCallback();
            };

    _heteroInferRequest->setCallbackForLastRequest(f);
}
