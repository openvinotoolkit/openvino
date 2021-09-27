// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <map>
#include <vector>
#include <string>

#include <cpp_interfaces/interface/ie_iplugin_internal.hpp>
#include <cpp_interfaces/interface/ie_internal_plugin_config.hpp>
#include "multi_device_exec_network.hpp"

namespace MultiDevicePlugin {

class MultiDeviceInferencePlugin : public InferenceEngine::IInferencePlugin {
public:
    MultiDeviceInferencePlugin();
    ~MultiDeviceInferencePlugin() = default;

    InferenceEngine::IExecutableNetworkInternal::Ptr LoadExeNetworkImpl(const InferenceEngine::CNNNetwork&        network,
                                                                       const std::map<std::string, std::string>& config) override;

    InferenceEngine::IExecutableNetworkInternal::Ptr LoadNetwork(const std::string& modelPath,
                                                                 const std::map<std::string, std::string>& config) override;

    void SetConfig(const std::map<std::string, std::string>& config) override;
    InferenceEngine::Parameter GetConfig(const std::string& name, const std::map<std::string, InferenceEngine::Parameter> & options) const override;
    InferenceEngine::QueryNetworkResult QueryNetwork(const InferenceEngine::CNNNetwork&        network,
                                                     const std::map<std::string, std::string>& config) const override;
    InferenceEngine::Parameter GetMetric(const std::string& name,
                                         const std::map<std::string, InferenceEngine::Parameter>& options) const override;

    std::vector<MultiDevicePlugin::DeviceInformation> ParseMetaDevices(const std::string & devicesRequestsCfg,
                                                                       const std::map<std::string, std::string> & config) const;

    std::string GetDeviceList(const std::map<std::string, std::string>& config) const;
    DeviceInformation SelectDevice(const std::vector<DeviceInformation>& metaDevices, const std::string& networkPrecision = METRIC_VALUE(FP32));

protected:
    std::map<std::string, std::string> GetSupportedConfig(const std::map<std::string, std::string>& config,
                                                          const MultiDevicePlugin::DeviceName & deviceName) const;

private:
    InferenceEngine::IExecutableNetworkInternal::Ptr LoadNetworkImpl(const std::string& modelPath,
                                                                       InferenceEngine::CNNNetwork network,
                                                                       const std::map<std::string, std::string>& config,
                                                                       const std::string &networkPrecision = METRIC_VALUE(FP32));
    static void CheckConfig(const std::map<std::string, std::string>& config, bool& needPerfCounters,
                            std::map<std::string, std::string>& filterConfig);
    std::vector<DeviceInformation> FilterDevice(const std::vector<DeviceInformation>& metaDevices,
                                                const std::map<std::string, std::string>& config);
};

}  // namespace MultiDevicePlugin
