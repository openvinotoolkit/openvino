// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <vector>
#include <string>
#include <unordered_set>

#include <cpp_interfaces/impl/ie_plugin_internal.hpp>
#include <cpp_interfaces/interface/ie_internal_plugin_config.hpp>
#include "auto_exec_network.hpp"

namespace AutoPlugin {
namespace IE = InferenceEngine;
using ConfigType = std::map<std::string, std::string>;

class AutoInferencePlugin : public IE::InferencePluginInternal {
public:
    AutoInferencePlugin();
    ~AutoInferencePlugin() = default;
    IE::ExecutableNetworkInternal::Ptr LoadExeNetworkImpl(const IE::CNNNetwork& network, const ConfigType& config) override;
    IE::IExecutableNetworkInternal::Ptr LoadNetwork(const std::string& fileName, const ConfigType& config) override;
    IE::QueryNetworkResult QueryNetwork(const IE::CNNNetwork& network, const ConfigType& config) const override;
    IE::Parameter GetMetric(const std::string& name, const std::map<std::string, IE::Parameter>& options) const override;
    IE::Parameter GetConfig(const std::string& name, const std::map<std::string, IE::Parameter> & options) const override;
    void SetConfig(const ConfigType& config) override;

private:
    std::vector<AutoPlugin::DeviceInformation> GetDeviceChoice(const ConfigType&  config) const;

protected:
    ConfigType GetSupportedConfig(const ConfigType& config, const AutoPlugin::DeviceName & deviceName) const;
};

}  // namespace AutoPlugin
