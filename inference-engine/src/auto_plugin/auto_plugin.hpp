// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <vector>
#include <string>
#include <unordered_set>
#include <type_traits>

#include <cpp_interfaces/interface/ie_iplugin_internal.hpp>
#include <cpp_interfaces/interface/ie_internal_plugin_config.hpp>
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

private:
    std::vector<DeviceName> GetDeviceList(const ConfigType&  config) const;
    std::vector<std::string> GetOptimizationCapabilities(const std::map<std::string, IE::Parameter>& options) const;
    DeviceName SelectDevice(const std::vector<DeviceName>& metaDevices, const std::string& networkPrecision = METRIC_VALUE(FP32));
    ConfigType GetSupportedConfig(const ConfigType& config, const DeviceName & deviceName) const;
    void CheckConfig(const ConfigType& config);
    static ConfigType mergeConfigs(ConfigType config, const ConfigType& local);

    template <typename T>
    std::shared_ptr<AutoExecutableNetwork> LoadNetworkImpl(const T &param, const ConfigType &config, const std::string &networkPrecision = METRIC_VALUE(FP32)) {
        if (GetCore() == nullptr) {
            IE_THROW() << "Please, work with AUTO device via InferencEngine::Core object";
        }

        CheckConfig(config);

        auto fullConfig = mergeConfigs(_config, config);
        auto metaDevices = GetDeviceList(fullConfig);
        DeviceName selectedDevice;
        IE::SoExecutableNetworkInternal executableNetwork;
        while (!metaDevices.empty()) {
            selectedDevice = SelectDevice(metaDevices, networkPrecision);
            try {
                executableNetwork = GetCore()->LoadNetwork(param, selectedDevice, {});
                break;
            } catch (...) {
                auto eraseDevice = std::find_if(metaDevices.begin(), metaDevices.end(),
                    [=](const DeviceName& d)->bool{return d == selectedDevice;});
                if (eraseDevice == metaDevices.end()) {
                    IE_THROW() << "Didn't find the selected device name";
                }
                metaDevices.erase(eraseDevice);
                executableNetwork = {};
            }
        }
        if (!executableNetwork) {
            IE_THROW() << "Failed to load network by AUTO plugin";
        }

        bool enablePerfCount = fullConfig.find(IE::PluginConfigParams::KEY_PERF_COUNT) != fullConfig.end();

        auto impl = std::make_shared<AutoExecutableNetwork>(executableNetwork, enablePerfCount);

        if (std::is_same<std::string, T>::value) {
            SetExeNetworkInfo(impl, executableNetwork->GetInputsInfo(),
                                    executableNetwork->GetOutputsInfo());
        }

        return impl;
    }
};

}  // namespace AutoPlugin
