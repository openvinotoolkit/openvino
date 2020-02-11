// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <map>
#include <cpp_interfaces/impl/ie_plugin_internal.hpp>
#include <cpp_interfaces/impl/ie_executable_network_internal.hpp>
#include "gna_executable_network.hpp"

namespace GNAPluginNS {

class GNAPluginInternal  : public InferenceEngine::InferencePluginInternal {
 public:
    InferenceEngine::ExecutableNetworkInternal::Ptr LoadExeNetworkImpl(const InferenceEngine::ICore * core,
                                                InferenceEngine::ICNNNetwork &network,
                                                const std::map<std::string, std::string> &config) override {
        return std::make_shared<GNAExecutableNetwork>(network, config);
    }
    void SetConfig(const std::map<std::string, std::string> &config) override {
        auto plg = std::make_shared<GNAPlugin>();
        plg->SetConfig(config);
    }
    InferenceEngine::IExecutableNetwork::Ptr  ImportNetwork(
                                                const std::string &modelFileName,
                                                const std::map<std::string, std::string> &config) override {
        return make_executable_network(std::make_shared<GNAExecutableNetwork>(modelFileName, config));
    }
    InferenceEngine::ExecutableNetwork ImportNetwork(
                                        std::istream &,
                                        const std::map<std::string, std::string> &) override {
        THROW_GNA_EXCEPTION << "Not implemented";
    }

    std::string GetName() const noexcept override {
        auto plg = std::make_shared<GNAPlugin>();
        return plg->GetName();
    }

    InferenceEngine::ICNNNetwork&  RemoveConstLayers(InferenceEngine::ICNNNetwork &network) override {
        return network;
    }

    void QueryNetwork(const InferenceEngine::ICNNNetwork& network,
                      const std::map<std::string, std::string>& config,
                      InferenceEngine::QueryNetworkResult& res) const override {
        auto plg = std::make_shared<GNAPlugin>();
        try {
            plg->SetConfig(config);
        } catch (InferenceEngine::details::InferenceEngineException) {}
        plg->QueryNetwork(network, config, res);
    }

    InferenceEngine::Parameter GetMetric(const std::string& name,
                                         const std::map<std::string, InferenceEngine::Parameter> & options) const override {
        GNAPlugin statelessPlugin;
        return statelessPlugin.GetMetric(name, options);
    }

    InferenceEngine::Parameter GetConfig(const std::string& name, const std::map<std::string, InferenceEngine::Parameter> & options) const override {
        GNAPlugin statelessPlugin;
        return statelessPlugin.GetConfig(name, options);
    }
};

}  // namespace GNAPluginNS
