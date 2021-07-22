// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include <memory>
#include <map>

#include "plugin_async_infer_request.hpp"

namespace PluginHelper {
using namespace InferenceEngine;
PluginAsyncInferRequest::PluginAsyncInferRequest(const PluginInferRequest::Ptr&               inferRequest,
                                                 const PluginExecHelper::Ptr&                 autoExecutableNetwork,
                                                 const InferenceEngine::ITaskExecutor::Ptr&   callbackExecutor,
                                                 bool                                         enablePerfCount)
    : AsyncInferRequestThreadSafeDefault(inferRequest, nullptr, callbackExecutor)
    , _inferRequest(inferRequest)
    , _enablePerfCount(enablePerfCount) {
    // this executor starts the inference while  the task (checking the result) is passed to the next stage
    struct ThisRequestExecutor : public InferenceEngine::ITaskExecutor {
        explicit ThisRequestExecutor(PluginAsyncInferRequest* asyncInferRequest) : _asyncInferRequest{asyncInferRequest} {}
        void run(InferenceEngine::Task task) override {
            auto workerInferRequest = _asyncInferRequest->_workerInferRequest;
            // this replace the actual infer request callback task in the worker created by auto_exec_network.cpp
            // and note that this task is next stage task which here means PerfCount part task, so it will be executed after callback.
            workerInferRequest->_task = std::move(task);
            workerInferRequest->_inferRequest->StartAsync();
        };
        PluginAsyncInferRequest* _asyncInferRequest = nullptr;
    };

    _pipeline = {
        // if the request is coming with device-specific remote blobs make sure it is scheduled to the specific device only:
        {   /*TaskExecutor*/ std::make_shared<ImmediateExecutor>(), /*task*/ [this, autoExecutableNetwork] {
            // by default, no preferred device:
            autoExecutableNetwork->_thisPreferredDeviceName = "";
            // if any input is remote (e.g. was set with SetBlob), let' use the corresponding device
            for (const auto &it : autoExecutableNetwork->GetInputsInfo()) {
                auto b = _inferRequest->GetBlob(it.first);
                auto r = b->as<RemoteBlob>();
                if (r) {
                    const auto name = r->getDeviceName();
                    const auto res = std::find_if(
                        autoExecutableNetwork->_devicePrioritiesInitial.cbegin(),
                        autoExecutableNetwork->_devicePrioritiesInitial.cend(),
                        [&name](const PluginHelper::DeviceInformation& d){ return d.deviceName == name; });
                    if (autoExecutableNetwork->_devicePrioritiesInitial.cend() == res) {
                        IE_THROW() << "None of the devices (for which current plugin configuration was "
                                      "initialized) supports a remote blob created on the device named " << name;
                    } else {
                        // it is ok to take the c_str() here (as pointed in the multi_device_exec_network.hpp we need to use const char*)
                        // as the original strings are from the "persistent" vector (with the right lifetime)
                        autoExecutableNetwork->_thisPreferredDeviceName = res->deviceName.c_str();
                        break;
                    }
                }
            }
        }},
        // schedule a worker for current infer request, then we need sets the device-agnostic blobs to the actual (scheduled device-specific) request
        {   autoExecutableNetwork, [this](){
            _workerInferRequest = PluginExecHelper::_thisWorkerInferRequest;
            _inferRequest->SetBlobsToAnotherRequest(_workerInferRequest->_inferRequest);
        }},
        // final task in the pipeline:
        {   /*TaskExecutor*/std::make_shared<ThisRequestExecutor>(this), /*task*/ [this] {
            if (nullptr != _workerInferRequest->_exceptionPtr) {
                std::rethrow_exception(_workerInferRequest->_exceptionPtr);
            }
            if (_enablePerfCount) {
                // fixme: this causes a exception for both master branch and current. both MULTI:GPU and AUTO:GPU
                // Ticket: 59892
                _perfMap = _workerInferRequest->_inferRequest->GetPerformanceCounts();
            }
        }}
    };
}

void PluginAsyncInferRequest::Infer_ThreadUnsafe() {
    InferUsingAsync();
}

std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> PluginAsyncInferRequest::GetPerformanceCounts() const {
    CheckState();
    return _perfMap;
}

PluginAsyncInferRequest::~PluginAsyncInferRequest() {
    StopAndWait();
}

} // namespace AutoPlugin