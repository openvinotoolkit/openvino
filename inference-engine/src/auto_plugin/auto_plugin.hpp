// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <vector>
#include <string>
#include <unordered_set>
#include <type_traits>

#include <cpp_interfaces/interface/ie_internal_plugin_config.hpp>
#include <cpp_interfaces/interface/ie_iplugin_internal.hpp>
#include <threading/ie_executor_manager.hpp>

#include "auto_exec_network.hpp"

namespace AutoPlugin {
namespace IE = InferenceEngine;
using ConfigType = std::map<std::string, std::string>;

class AutoInferencePlugin : public IE::IInferencePlugin {
public:
    AutoInferencePlugin();
    ~AutoInferencePlugin() = default;
    IE::IExecutableNetworkInternal::Ptr LoadExeNetworkImpl(const IE::CNNNetwork& network, const ConfigType& config) override;
    IE::IExecutableNetworkInternal::Ptr LoadNetwork(const std::string& fileName, const ConfigType& config) override;
    IE::QueryNetworkResult QueryNetwork(const IE::CNNNetwork& network, const ConfigType& config) const override;
    IE::Parameter GetMetric(const std::string& name, const std::map<std::string, IE::Parameter>& options) const override;
    IE::Parameter GetConfig(const std::string& name, const std::map<std::string, IE::Parameter> & options) const override;
    void SetConfig(const ConfigType& config) override;
    std::vector<DeviceName> GetDeviceList(const ConfigType&  config) const;
    DeviceName SelectDevice(const std::vector<DeviceName>& metaDevices, const std::string& networkPrecision = METRIC_VALUE(FP32));

private:
    std::vector<std::string> GetOptimizationCapabilities(const std::map<std::string, IE::Parameter>& options) const;
    void CheckConfig(const ConfigType& config);
    static ConfigType mergeConfigs(ConfigType config, const ConfigType& local);
};

}  // namespace AutoPlugin
