// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include "myriad_async_infer_request.h"
#include <vpu/utils/profiling.hpp>

using namespace vpu::MyriadPlugin;
using namespace InferenceEngine;

MyriadAsyncInferRequest::MyriadAsyncInferRequest(MyriadInferRequest::Ptr request,
                                                 const InferenceEngine::ITaskExecutor::Ptr &taskExecutorStart,
                                                 const InferenceEngine::ITaskExecutor::Ptr &taskExecutorGetResult,
                                                 const InferenceEngine::TaskSynchronizer::Ptr &taskSynchronizer,
                                                 const InferenceEngine::ITaskExecutor::Ptr &callbackExecutor)
        : InferenceEngine::AsyncInferRequestThreadSafeDefault(request,
                                                              taskExecutorStart,
                                                              taskSynchronizer,
                                                              callbackExecutor),
          _request(request), _taskExecutorGetResult(taskExecutorGetResult) {}


InferenceEngine::StagedTask::Ptr MyriadAsyncInferRequest::createAsyncRequestTask() {
    VPU_PROFILE(createAsyncRequestTask);
    return std::make_shared<StagedTask>([this]() {
        auto asyncTaskCopy = _asyncTask;
        try {
            switch (asyncTaskCopy->getStage()) {
                case 3: {
                    _request->InferAsync();
                    asyncTaskCopy->stageDone();
                    _taskExecutorGetResult->startTask(asyncTaskCopy);
                }
                    break;
                case 2: {
                    _request->GetResult();
                    asyncTaskCopy->stageDone();
                    if (_callbackManager.isCallbackEnabled()) {
                        _callbackManager.startTask(asyncTaskCopy);
                    } else {
                        asyncTaskCopy->stageDone();
                    }
                }
                    break;
                case 1: {
                    setIsRequestBusy(false);
                    asyncTaskCopy->stageDone();
                    _callbackManager.runCallback();
                }
                    break;
                default:
                    break;
            }
        } catch (...) {
            processAsyncTaskFailure(asyncTaskCopy);
        }
    }, 3);
}

MyriadAsyncInferRequest::~MyriadAsyncInferRequest() {
    waitAllAsyncTasks();
}

