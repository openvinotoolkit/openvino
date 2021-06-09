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

protected:
    std::map<std::string, std::string> GetSupportedConfig(const std::map<std::string, std::string>& config,
                                                          const MultiDevicePlugin::DeviceName & deviceName) const;

private:
    InferenceEngine::IExecutableNetworkInternal::Ptr LoadExeNetworkImpl(const std::string& modelPath,
                                                                       InferenceEngine::CNNNetwork network,
                                                                       const std::map<std::string, std::string>& config);
};

}  // namespace MultiDevicePlugin
