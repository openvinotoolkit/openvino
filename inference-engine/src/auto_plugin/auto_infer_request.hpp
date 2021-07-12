// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <cpp_interfaces/interface/ie_iinfer_request_internal.hpp>
#include <ie_blob.h>
#include <ie_common.h>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "auto_exec_network.hpp"

namespace AutoPlugin {

class AutoInferRequest : public InferenceEngine::IInferRequestInternal {
public:
    using Ptr = std::shared_ptr<AutoInferRequest>;
    explicit AutoInferRequest(const InferenceEngine::InputsDataMap&             networkInputs,
                              const InferenceEngine::OutputsDataMap&            networkOutputs,
                              const InferenceEngine::SoIInferRequestInternal&   inferRequest,
                              const InferenceEngine::IExecutableNetworkInternal::Ptr executeNetwork,
                              bool alreadyActualNetwork,
                              bool enablePerfCount);
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> GetPerformanceCounts() const override;
    void InferImpl() override;
    void SetBlob(const std::string& name, const InferenceEngine::Blob::Ptr& data) override;
    InferenceEngine::Blob::Ptr GetBlob(const std::string& name) override;
    void Cancel() override;
    //async impl
    void StartAsync() override;
    InferenceEngine::StatusCode Wait(int64_t millis_timeout) override;
    void SetCallback(Callback callback) override;
    void SetBlobsToAnotherRequest(const InferenceEngine::SoIInferRequestInternal& req);

private:
    void HotSwapRequests();
    void SetBlobsToDeviceRequest();

private:
    InferenceEngine::SoIInferRequestInternal _inferRequest;
    AutoPlugin::AutoExecutableNetwork::Ptr _autoExecutableNetwork;
    Callback _callback; // need to save the callback for hot-swap of the requests
    bool _alreadyActualNetwork{ false };
    bool _enablePerfCount { false };
};

}  // namespace AutoPlugin
