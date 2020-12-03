// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <map>
#include <vector>
#include <string>

#include <cpp_interfaces/impl/ie_plugin_internal.hpp>
#include <cpp_interfaces/interface/ie_internal_plugin_config.hpp>
#include "multi_device_exec_network.hpp"

namespace MultiDevicePlugin {

class MultiDeviceInferencePlugin : public InferenceEngine::InferencePluginInternal {
public:
    MultiDeviceInferencePlugin();
    ~MultiDeviceInferencePlugin() override = default;

    InferenceEngine::ExecutableNetworkInternal::Ptr LoadExeNetworkImpl(const InferenceEngine::ICNNNetwork& network,
                                                                       const std::map<std::string, std::string>& config) override;

    void SetConfig(const std::map<std::string, std::string>& config) override;
    Parameter GetConfig(const std::string& name,
                        const std::map<std::string, Parameter> & options) const override;
    InferenceEngine::QueryNetworkResult QueryNetwork(const InferenceEngine::ICNNNetwork&       network,
                                                     const std::map<std::string, std::string>& config) const override;
    InferenceEngine::Parameter GetMetric(const std::string& name,
                                         const std::map<std::string, InferenceEngine::Parameter>& options) const override;

    std::vector<MultiDevicePlugin::DeviceInformation> ParseMetaDevices(const std::string & devicesRequestsCfg,
                                                  const std::map<std::string, std::string> & config) const;

protected:
    std::map<std::string, std::string> GetSupportedConfig(const std::map<std::string, std::string>& config,
                                                          const MultiDevicePlugin::DeviceName & deviceName) const;
};

}  // namespace MultiDevicePlugin
