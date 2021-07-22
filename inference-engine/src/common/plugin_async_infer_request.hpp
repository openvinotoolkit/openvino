// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <map>
#include <vector>
#include <utility>
#include <memory>
#include <string>

#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>

#include "plugin_infer_request.hpp"
#include "plugin_helper.hpp"
#include "plugin_exec_network.hpp"

namespace PluginHelper {

class PluginAsyncInferRequest : public InferenceEngine::AsyncInferRequestThreadSafeDefault {
public:
    using Ptr = std::shared_ptr<PluginAsyncInferRequest>;

    explicit PluginAsyncInferRequest(const PluginInferRequest::Ptr&              inferRequest,
                                     const PluginExecHelper::Ptr&                executableNetwork,
                                     const InferenceEngine::ITaskExecutor::Ptr&  callbackExecutor,
                                     bool                                        enablePerfCount);
    void Infer_ThreadUnsafe() override;
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> GetPerformanceCounts() const override;
    ~PluginAsyncInferRequest();

private:
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>  _perfMap;
    PluginInferRequest::Ptr                                             _inferRequest;
    PluginHelper::WorkerInferRequest*                                   _workerInferRequest = nullptr;
    bool                                                                _enablePerfCount;
};



}  // namespace PluginHelper
