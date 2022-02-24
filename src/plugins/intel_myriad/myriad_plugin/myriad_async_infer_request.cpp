// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include "myriad_async_infer_request.h"
#include <vpu/utils/profiling.hpp>

using namespace vpu::MyriadPlugin;
using namespace InferenceEngine;

MyriadAsyncInferRequest::MyriadAsyncInferRequest(MyriadInferRequest::Ptr request,
                                                 const InferenceEngine::ITaskExecutor::Ptr &taskExecutorStart,
                                                 const InferenceEngine::ITaskExecutor::Ptr &callbackExecutor,
                                                 const InferenceEngine::ITaskExecutor::Ptr &taskExecutorGetResult)
: InferenceEngine::AsyncInferRequestThreadSafeDefault(request, taskExecutorStart, callbackExecutor),
    _request(request), _taskExecutorGetResult(taskExecutorGetResult) {
        _pipeline = {
            {_requestExecutor, [this] {
                _request->InferAsync();
            }},
            {_taskExecutorGetResult, [this] {
                _request->GetResult();
            }}
        };
    }

MyriadAsyncInferRequest::~MyriadAsyncInferRequest() {
    StopAndWait();
}
