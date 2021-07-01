// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <map>
#include <cpp_interfaces/interface/ie_iplugin_internal.hpp>
#include <cpp_interfaces/interface/ie_iexecutable_network_internal.hpp>
#include "gna_executable_network.hpp"
#include "gna_plugin_config.hpp"
#include <legacy/ie_util_internal.hpp>

namespace GNAPluginNS {

class GNAPluginInternal  : public InferenceEngine::IInferencePlugin {
private:
    std::mutex syncCallsToLoadExeNetworkImpl;
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
    InferenceEngine::IExecutableNetworkInternal::Ptr LoadExeNetworkImpl(
                                                const InferenceEngine::CNNNetwork &network,
                                                const std::map<std::string, std::string> &config) override {
        std::lock_guard<std::mutex> lock{ syncCallsToLoadExeNetworkImpl };
        Config updated_config(defaultConfig);
        updated_config.UpdateFromMap(config);
        auto plg = std::make_shared<GNAPlugin>(updated_config.keyConfigMap);
        plgPtr = plg;
        InferenceEngine::CNNNetwork clonedNetwork(InferenceEngine::cloneNetwork(network));
        return std::make_shared<GNAExecutableNetwork>(clonedNetwork, plg);
    }

    void SetConfig(const std::map<std::string, std::string> &config) override {
        defaultConfig.UpdateFromMap(config);
    }

    InferenceEngine::IExecutableNetworkInternal::Ptr ImportNetwork(
                                                const std::string &modelFileName,
                                                const std::map<std::string, std::string> &config) override {
        Config updated_config(defaultConfig);
        updated_config.UpdateFromMap(config);
        auto plg = std::make_shared<GNAPlugin>(updated_config.keyConfigMap);
        plgPtr = plg;

        return std::make_shared<GNAExecutableNetwork>(modelFileName, plg);
    }

    InferenceEngine::IExecutableNetworkInternal::Ptr ImportNetwork(std::istream& networkModel,
                                                     const std::map<std::string, std::string>& config) override {
        Config updated_config(defaultConfig);
        updated_config.UpdateFromMap(config);
        auto plg = std::make_shared<GNAPlugin>(updated_config.keyConfigMap);
        plgPtr = plg;
        return std::make_shared<GNAExecutableNetwork>(networkModel, plg);
    }

    std::string GetName() const noexcept override {
        return GetCurrentPlugin()->GetName();
    }

    InferenceEngine::QueryNetworkResult QueryNetwork(const InferenceEngine::CNNNetwork& network,
                                                     const std::map<std::string, std::string>& config) const override {
        auto plg = GetCurrentPlugin();
        try {
            plg->SetConfig(config);
        } catch (InferenceEngine::Exception&) {}
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
