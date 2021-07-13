// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <string>
#include <vector>
#include <memory>
#include <map>

#include "auto_async_infer_request.hpp"

namespace AutoPlugin {
using namespace InferenceEngine;

AutoAsyncInferRequest::AutoAsyncInferRequest(const AutoInferRequest::Ptr&           inferRequest,
                               const AutoExecutableNetwork::Ptr&      autoExecutableNetwork,
                               const InferenceEngine::ITaskExecutor::Ptr&    callbackExecutor)
                               : AsyncInferRequestThreadSafeDefault(inferRequest, nullptr, callbackExecutor)
                               , _inferRequest(inferRequest) {
    // this executor starts the inference while  the task (checking the result) is passed to the next stage
    struct ThisRequestExecutor : public ITaskExecutor {
        explicit ThisRequestExecutor(AutoAsyncInferRequest* asyncInferRequest) : _asyncInferRequest{asyncInferRequest} {}
        void run(Task task) override {
            auto workerInferRequest = _asyncInferRequest->_workerInferRequest;
            workerInferRequest->_task = std::move(task);
            workerInferRequest->_inferRequest->StartAsync();
        };
        AutoAsyncInferRequest* _asyncInferRequest = nullptr;
    };

    // todo: redefine _pipeline
    _pipeline = {
        // schedule a worker for current infer request, then we need sets the device-agnostic blobs to the actual (scheduled device-specific) request
        { autoExecutableNetwork, [this](){
            _workerInferRequest = AutoExecutableNetwork::_thisWorkerInferRequest;
            _inferRequest->SetBlobsToAnotherRequest(_workerInferRequest->_inferRequest);
        }},
        // final task in the pipeline:
        { /*TaskExecutor*/std::make_shared<ThisRequestExecutor>(this), /*task*/ [this] {
            if (nullptr != _workerInferRequest->_exceptionPtr) {
                std::rethrow_exception(_workerInferRequest->_exceptionPtr);
            }
            // fixme: this causes a exception for both master branch and current. both MULTI:GPU and AUTO:GPU
//            if (true)
//                _perfMap = _workerInferRequest->_inferRequest->GetPerformanceCounts();
        }}
    };
}
void AutoAsyncInferRequest::Infer_ThreadUnsafe() {
    InferUsingAsync();
}
std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> AutoAsyncInferRequest::GetPerformanceCounts() const {
    CheckState();
    return _perfMap;
}

AutoAsyncInferRequest::~AutoAsyncInferRequest() {
    StopAndWait();
}
} // namespace AutoPlugin