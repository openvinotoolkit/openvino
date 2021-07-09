// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <map>
#include <vector>
#include <string>

#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>
#include <threading/ie_itask_executor.hpp>

namespace AutoPlugin {

class AutoInferencePlugin;

using DeviceName = std::string;
using ConfigType = std::map<std::string, std::string>;
using NetworkFuture = std::future<InferenceEngine::SoExecutableNetworkInternal>;

class AutoExecutableNetwork : public InferenceEngine::IExecutableNetworkInternal {
public:
    using Ptr = std::shared_ptr<AutoExecutableNetwork>;

    AutoExecutableNetwork(const std::string&                 modelPath,
                          const InferenceEngine::CNNNetwork& network,
                          const ConfigType&                  config,
                          AutoInferencePlugin*               plugin);

    void Export(std::ostream& networkModel) override;
    InferenceEngine::RemoteContext::Ptr GetContext() const override;
    InferenceEngine::CNNNetwork GetExecGraphInfo() override;
    InferenceEngine::Parameter GetMetric(const std::string &name) const override;
    void SetConfig(const std::map<std::string, InferenceEngine::Parameter>& config) override;
    InferenceEngine::Parameter GetConfig(const std::string& name) const override;
    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                                                       InferenceEngine::OutputsDataMap networkOutputs) override;
    bool TryGetActualNetwork(InferenceEngine::SoExecutableNetworkInternal& soExecNetwork);

    ~AutoExecutableNetwork();

private:
    void WaitForActualDevice() const;

private:
    InferenceEngine::SoExecutableNetworkInternal _networkFirstReady;
    mutable InferenceEngine::SoExecutableNetworkInternal _networkActualNeeded;
    NetworkFuture _cpuFuture;
    mutable NetworkFuture _acceleratorFuture;
    bool _enablePerfCount;
    mutable std::atomic<bool> _alreadyActualNetwork = {false};
    std::map<std::string, InferenceEngine::Parameter> _cacheConfig;
    AutoInferencePlugin* _autoPlugin;
};

}  // namespace AutoPlugin
