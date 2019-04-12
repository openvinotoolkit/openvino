//
// Copyright 2017-2018 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "hetero_async_infer_request.h"
#include <assert.h>
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
    if (isRequestBusy()) THROW_IE_EXCEPTION << REQUEST_BUSY_str;
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