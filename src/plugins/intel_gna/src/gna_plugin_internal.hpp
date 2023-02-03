// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpp_interfaces/interface/ie_iexecutable_network_internal.hpp>
#include <cpp_interfaces/interface/ie_iplugin_internal.hpp>
#include <legacy/ie_util_internal.hpp>
#include <map>
#include <memory>
#include <string>

#include "gna_executable_network.hpp"
#include "gna_plugin_config.hpp"

namespace ov {
namespace intel_gna {

class GNAPluginInternal : public InferenceEngine::IInferencePlugin {
private:
    std::mutex syncCallsToLoadExeNetworkImpl;
    Config defaultConfig;
    std::weak_ptr<GNAPlugin> plgPtr;
    std::shared_ptr<GNAPlugin> GetCurrentPlugin() const {
        auto ptr = plgPtr.lock();
        if (ptr == nullptr) {
            return std::make_shared<GNAPlugin>();
        } else {
            return ptr;
        }
    }
    void remove_core_property(std::map<std::string, std::string>& configMap, const std::set<std::string>& coreConfig) {
        for (auto it : coreConfig) {
            auto item = configMap.find(it);
            if (item != configMap.end())
                configMap.erase(item);
        }
    }

protected:
    std::string _pluginInternalName = "GNA";

public:
    InferenceEngine::IExecutableNetworkInternal::Ptr LoadExeNetworkImpl(
        const InferenceEngine::CNNNetwork& network,
        const std::map<std::string, std::string>& config) override {
        std::lock_guard<std::mutex> lock{syncCallsToLoadExeNetworkImpl};
        Config updated_config(defaultConfig);
        auto core_config = GetCore() ? GetCore()->QueryCoreSupportedConfig() : std::set<std::string>();
        updated_config.UpdateFromMap(config, core_config);

        auto _config = updated_config.keyConfigMap;
        remove_core_property(_config, core_config);
        auto plg = std::make_shared<GNAPlugin>(_config);
        plgPtr = plg;
        InferenceEngine::CNNNetwork clonedNetwork(InferenceEngine::cloneNetwork(network));
        return std::make_shared<GNAExecutableNetwork>(clonedNetwork, plg);
    }

    void SetConfig(const std::map<std::string, std::string>& config) override {
        defaultConfig.UpdateFromMap(config);
    }

    InferenceEngine::IExecutableNetworkInternal::Ptr ImportNetwork(
        const std::string& modelFileName,
        const std::map<std::string, std::string>& config) override {
        Config updated_config(defaultConfig);
        auto core_config = GetCore() ? GetCore()->QueryCoreSupportedConfig() : std::set<std::string>();
        updated_config.UpdateFromMap(config, core_config);

        auto _config = updated_config.keyConfigMap;
        remove_core_property(_config, core_config);
        auto plg = std::make_shared<GNAPlugin>(_config);
        plgPtr = plg;
        auto network_impl = std::make_shared<GNAExecutableNetwork>(modelFileName, plg);
        // set pointer for IInferencePlugin interface
        network_impl->SetPointerToPlugin(shared_from_this());

        return network_impl;
    }

    InferenceEngine::IExecutableNetworkInternal::Ptr ImportNetwork(
        std::istream& networkModel,
        const std::map<std::string, std::string>& config) override {
        Config updated_config(defaultConfig);
        auto core_config = GetCore() ? GetCore()->QueryCoreSupportedConfig() : std::set<std::string>();
        updated_config.UpdateFromMap(config, core_config);

        auto _config = updated_config.keyConfigMap;
        remove_core_property(_config, core_config);
        auto plg = std::make_shared<GNAPlugin>(_config);
        plgPtr = plg;
        auto network_impl = std::make_shared<GNAExecutableNetwork>(networkModel, plg);
        // set pointer for IInferencePlugin interface
        network_impl->SetPointerToPlugin(shared_from_this());

        return network_impl;
    }

    std::string GetName() const noexcept override {
        auto ptr = plgPtr.lock();
        if (ptr == nullptr) {
            return _pluginInternalName;
        } else {
            return ptr->GetName();
        }
    }

    InferenceEngine::QueryNetworkResult QueryNetwork(const InferenceEngine::CNNNetwork& network,
                                                     const std::map<std::string, std::string>& config) const override {
        auto plg = GetCurrentPlugin();
        try {
            plg->SetConfig(config);
        } catch (InferenceEngine::Exception&) {
        }
        return plg->QueryNetwork(network, config);
    }

    InferenceEngine::Parameter GetMetric(
        const std::string& name,
        const std::map<std::string, InferenceEngine::Parameter>& options) const override {
        return GetCurrentPlugin()->GetMetric(name, options);
    }

    InferenceEngine::Parameter GetConfig(
        const std::string& name,
        const std::map<std::string, InferenceEngine::Parameter>& options) const override {
        auto core_config = GetCore() ? GetCore()->QueryCoreSupportedConfig() : std::set<std::string>();
        if (core_config.count(name)) {
            THROW_GNA_EXCEPTION << "GetConfig: Unsupported GNA config key: " << name.c_str();
        }
        return defaultConfig.GetParameter(name);
    }
};

}  // namespace intel_gna
}  // namespace ov
