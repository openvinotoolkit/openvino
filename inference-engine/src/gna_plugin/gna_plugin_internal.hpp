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
#include "gna_plugin_config.hpp"
#include <legacy/ie_util_internal.hpp>

namespace GNAPluginNS {

class GNAPluginInternal  : public InferenceEngine::InferencePluginInternal {
private:
    Config defaultConfig;
    std::weak_ptr <GNAPlugin> plgPtr;
    std::shared_ptr<GNAPlugin> GetCurrentPlugin() const {
        auto ptr = plgPtr.lock();
        if (ptr == nullptr) {
            return std::make_shared<GNAPlugin>();
        } else {
            return ptr;
        }
    }

public:
    InferenceEngine::ExecutableNetworkInternal::Ptr LoadExeNetworkImpl(
                                                const InferenceEngine::CNNNetwork &network,
                                                const std::map<std::string, std::string> &config) override {
        Config updated_config(defaultConfig);
        updated_config.UpdateFromMap(config);
        auto plg = std::make_shared<GNAPlugin>(updated_config.key_config_map);
        plgPtr = plg;
        InferenceEngine::CNNNetwork clonedNetwork(InferenceEngine::cloneNetwork(network));
        return std::make_shared<GNAExecutableNetwork>(clonedNetwork, plg);
    }

    void SetConfig(const std::map<std::string, std::string> &config) override {
        defaultConfig.UpdateFromMap(config);
    }

    InferenceEngine::ExecutableNetwork ImportNetwork(
                                                const std::string &modelFileName,
                                                const std::map<std::string, std::string> &config) override {
        Config updated_config(defaultConfig);
        updated_config.UpdateFromMap(config);
        auto plg = std::make_shared<GNAPlugin>(updated_config.key_config_map);
        plgPtr = plg;

        return make_executable_network(std::make_shared<GNAExecutableNetwork>(modelFileName, plg));
    }

    InferenceEngine::ExecutableNetwork ImportNetwork(std::istream& networkModel,
                                                     const std::map<std::string, std::string>& config) override {
        Config updated_config(defaultConfig);
        updated_config.UpdateFromMap(config);
        auto plg = std::make_shared<GNAPlugin>(updated_config.key_config_map);
        plgPtr = plg;
        return make_executable_network(std::make_shared<GNAExecutableNetwork>(networkModel, plg));
    }

    using InferenceEngine::InferencePluginInternal::ImportNetwork;

    std::string GetName() const noexcept override {
        return GetCurrentPlugin()->GetName();
    }

    InferenceEngine::QueryNetworkResult QueryNetwork(const InferenceEngine::CNNNetwork& network,
                                                     const std::map<std::string, std::string>& config) const override {
        auto plg = GetCurrentPlugin();
        try {
            plg->SetConfig(config);
        } catch (InferenceEngine::details::InferenceEngineException) {}
        return plg->QueryNetwork(network, config);
    }

    InferenceEngine::Parameter GetMetric(const std::string& name,
                                         const std::map<std::string, InferenceEngine::Parameter> & options) const override {
        return GetCurrentPlugin()->GetMetric(name, options);
    }

    InferenceEngine::Parameter GetConfig(const std::string& name, const std::map<std::string, InferenceEngine::Parameter> & options) const override {
        return defaultConfig.GetParameter(name);
    }
};

}  // namespace GNAPluginNS
