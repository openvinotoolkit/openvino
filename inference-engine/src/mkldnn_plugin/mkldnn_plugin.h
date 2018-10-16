// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mkldnn_graph.h"
#include <string>
#include <map>
#include <memory>
#include <cpp_interfaces/impl/ie_plugin_internal.hpp>

namespace MKLDNNPlugin {

class Engine : public InferenceEngine::InferencePluginInternal {
public:
    Engine() = default;
    ~Engine() override = default;

    InferenceEngine::ExecutableNetworkInternal::Ptr
    LoadExeNetworkImpl(InferenceEngine::ICNNNetwork &network,
                       const std::map<std::string, std::string> &config) override;

    void AddExtension(InferenceEngine::IExtensionPtr extension) override;
    /**
     * @deprecated
     * @param config
     */
    void SetConfig(const std::map<std::string, std::string> &config) override;

    /**
     * @depricated Use the version with config parameter
     */
    void QueryNetwork(const InferenceEngine::ICNNNetwork& network, InferenceEngine::QueryNetworkResult& res) const override;
    void QueryNetwork(const InferenceEngine::ICNNNetwork& network,
                      const std::map<std::string, std::string>& config, InferenceEngine::QueryNetworkResult& res) const override;


private:
    Config engConfig;
    MKLDNNExtensionManager::Ptr extensionManager = std::make_shared<MKLDNNExtensionManager>();
};

}  // namespace MKLDNNPlugin
