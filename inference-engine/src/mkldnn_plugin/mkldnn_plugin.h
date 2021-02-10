// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpp_interfaces/impl/ie_plugin_internal.hpp>
#include "mkldnn_exec_network.h"

#include <string>
#include <map>
#include <unordered_map>
#include <memory>
#include <functional>
#include <vector>

namespace MKLDNNPlugin {

class Engine : public InferenceEngine::InferencePluginInternal {
public:
    Engine();
    ~Engine() override;

    InferenceEngine::ExecutableNetworkInternal::Ptr
    LoadExeNetworkImpl(const InferenceEngine::CNNNetwork &network,
                       const std::map<std::string, std::string> &config) override;

    void AddExtension(InferenceEngine::IExtensionPtr extension) override;

    void SetConfig(const std::map<std::string, std::string> &config) override;

    InferenceEngine::Parameter GetConfig(const std::string& name, const std::map<std::string, InferenceEngine::Parameter>& options) const override;

    InferenceEngine::Parameter GetMetric(const std::string& name, const std::map<std::string, InferenceEngine::Parameter>& options) const override;

    InferenceEngine::QueryNetworkResult QueryNetwork(const InferenceEngine::CNNNetwork& network,
                                                     const std::map<std::string, std::string>& config) const override;

private:
    Config engConfig;
    NumaNodesWeights weightsSharing;
    MKLDNNExtensionManager::Ptr extensionManager = std::make_shared<MKLDNNExtensionManager>();
};

}  // namespace MKLDNNPlugin
