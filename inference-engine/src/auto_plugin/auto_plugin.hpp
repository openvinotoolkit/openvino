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
    std::vector<AutoPlugin::DeviceInformation> GetDeviceChoice(const ConfigType&  config) const;
    std::vector<std::string> GetOptimizationCapabilities(const std::map<std::string, IE::Parameter>& options) const;
    DeviceInformation SelectDevice(const std::vector<DeviceInformation>& metaDevices, const std::string& networkPrecision = METRIC_VALUE(FP32));
    ConfigType GetSupportedConfig(const ConfigType& config, const AutoPlugin::DeviceName & deviceName) const;
    static ConfigType mergeConfigs(ConfigType config, const ConfigType& local);

    template <typename T>
    std::shared_ptr<AutoExecutableNetwork> LoadNetworkImpl(const T &param, const ConfigType &config, const std::string &networkPrecision = METRIC_VALUE(FP32)) {
        if (GetCore() == nullptr) {
            IE_THROW() << "Please, work with AUTO device via InferencEngine::Core object";
        }
        auto fullConfig = mergeConfigs(_config, config);
        auto metaDevices = GetDeviceChoice(fullConfig);
        DeviceInformation selectedDevice;
        IE::SoExecutableNetworkInternal executableNetwork;
        while (!metaDevices.empty()) {
            selectedDevice = SelectDevice(metaDevices, networkPrecision);
            try {
                executableNetwork = GetCore()->LoadNetwork(param, selectedDevice.deviceName, selectedDevice.config);
                break;
            } catch (...) {
                auto eraseDevice = std::find_if(metaDevices.begin(), metaDevices.end(),
                    [=](const DeviceInformation& d)->bool{return d.deviceName == selectedDevice.deviceName;});
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
        auto impl = std::make_shared<AutoExecutableNetwork>(executableNetwork);

        if (std::is_same<std::string, T>::value) {
            SetExeNetworkInfo(impl, executableNetwork->GetInputsInfo(),
                                    executableNetwork->GetOutputsInfo());
        }

        return impl;
    }
};

}  // namespace AutoPlugin
