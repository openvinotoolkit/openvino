// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <map>
#include <vector>
#include <utility>
#include <memory>
#include <string>

#include <cpp_interfaces/impl/ie_infer_async_request_thread_safe_default.hpp>
#include "multi_device_infer_request.hpp"
#include "multi_device_exec_network.hpp"

namespace MultiDevicePlugin {

class MultiDeviceAsyncInferRequest : public InferenceEngine::AsyncInferRequestThreadSafeDefault {
public:
    using Ptr = std::shared_ptr<MultiDeviceAsyncInferRequest>;

    explicit MultiDeviceAsyncInferRequest(const MultiDeviceInferRequest::Ptr&           inferRequest,
                                          const bool                                    needPerfCounters,
                                          const MultiDeviceExecutableNetwork::Ptr&      multiDeviceExecutableNetwork,
                                          const InferenceEngine::ITaskExecutor::Ptr&    callbackExecutor);
    void Infer_ThreadUnsafe() override;
    void GetPerformanceCounts_ThreadUnsafe(std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> &_perfMap) const override;
    ~MultiDeviceAsyncInferRequest() override;

protected:
    MultiDeviceExecutableNetwork::Ptr                                   _multiDeviceExecutableNetwork;
    MultiDeviceInferRequest::Ptr                                        _inferRequest;
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>  _perfMap;
    bool                                                                _needPerfCounters = false;
    MultiDeviceExecutableNetwork::WorkerInferRequest*                   _workerInferRequest = nullptr;
};

}  // namespace MultiDevicePlugin
