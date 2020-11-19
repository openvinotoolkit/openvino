// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <string>
#include <vector>
#include <memory>
#include <map>

#include "multi_device_async_infer_request.hpp"

namespace MultiDevicePlugin {
    using namespace InferenceEngine;

MultiDeviceAsyncInferRequest::MultiDeviceAsyncInferRequest(
    const MultiDeviceInferRequest::Ptr&         inferRequest,
    const bool                                  needPerfCounters,
    const MultiDeviceExecutableNetwork::Ptr&    multiDeviceExecutableNetwork,
    const ITaskExecutor::Ptr&                   callbackExecutor) :
    AsyncInferRequestThreadSafeDefault(inferRequest, nullptr, callbackExecutor),
    _multiDeviceExecutableNetwork{multiDeviceExecutableNetwork},
    _inferRequest{inferRequest},
    _needPerfCounters{needPerfCounters} {
    struct ThisRequestExecutor : public ITaskExecutor {
        explicit ThisRequestExecutor(MultiDeviceAsyncInferRequest* _this_) : _this{_this_} {}
        void run(Task task) override {
            auto workerInferRequest = _this->_workerInferRequest;
            workerInferRequest->_task = std::move(task);
            workerInferRequest->_inferRequest.StartAsync();
        };
        MultiDeviceAsyncInferRequest* _this = nullptr;
    };
    _pipeline = {
        {_multiDeviceExecutableNetwork, [this] {
            _workerInferRequest = MultiDeviceExecutableNetwork::_thisWorkerInferRequest;
            _inferRequest->SetBlobsToAnotherRequest(_workerInferRequest->_inferRequest);
        }},
        {std::make_shared<ThisRequestExecutor>(this), [this] {
            auto status = _workerInferRequest->_status;
            if (InferenceEngine::StatusCode::OK != status) {
                if (nullptr != InferenceEngine::CurrentException()) {
                    std::rethrow_exception(InferenceEngine::CurrentException());
                } else {
                    THROW_IE_EXCEPTION << InferenceEngine::details::as_status << status;
                }
            }
            if (_needPerfCounters) {
                _perfMap = _workerInferRequest->_inferRequest.GetPerformanceCounts();
            }
        }}
    };
}

void MultiDeviceAsyncInferRequest::Infer_ThreadUnsafe() {
    InferUsingAsync();
}

void MultiDeviceAsyncInferRequest::GetPerformanceCounts_ThreadUnsafe(std::map<std::string, InferenceEngineProfileInfo> &perfMap) const {
    perfMap = std::move(_perfMap);
}

MultiDeviceAsyncInferRequest::~MultiDeviceAsyncInferRequest() {
    StopAndWait();
}

}  // namespace MultiDevicePlugin
