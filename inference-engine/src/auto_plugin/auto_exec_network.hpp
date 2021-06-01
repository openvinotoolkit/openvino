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

using DeviceName = std::string;

struct DeviceInformation {
    DeviceName deviceName;
    std::map<std::string, std::string> config;
};

class AutoExecutableNetwork : public InferenceEngine::IExecutableNetworkInternal {
public:
    using Ptr = std::shared_ptr<AutoExecutableNetwork>;

    explicit AutoExecutableNetwork(const InferenceEngine::SoExecutableNetworkInternal& network);

    void Export(std::ostream& networkModel) override;
    InferenceEngine::RemoteContext::Ptr GetContext() const override;
    InferenceEngine::CNNNetwork GetExecGraphInfo() override;
    InferenceEngine::Parameter GetMetric(const std::string &name) const override;
    void SetConfig(const std::map<std::string, InferenceEngine::Parameter>& config) override;
    InferenceEngine::Parameter GetConfig(const std::string& name) const override;
    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                                                       InferenceEngine::OutputsDataMap networkOutputs) override;

    ~AutoExecutableNetwork();

private:
    InferenceEngine::SoExecutableNetworkInternal _network;
};

}  // namespace AutoPlugin
