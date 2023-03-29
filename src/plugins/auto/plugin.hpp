// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <map>
#include <vector>
#include <string>
#include <list>

#include <cpp_interfaces/interface/ie_iplugin_internal.hpp>
#include <cpp_interfaces/interface/ie_internal_plugin_config.hpp>
#include "utils/log_util.hpp"
#include "common.hpp"
#include "utils/plugin_config.hpp"

#ifdef  MULTIUNITTEST
#define MOCKTESTMACRO virtual
#define MultiDevicePlugin MockMultiDevicePlugin
#else
#define MOCKTESTMACRO
#endif

namespace MultiDevicePlugin {

class MultiDeviceInferencePlugin : public InferenceEngine::IInferencePlugin {
public:
    MultiDeviceInferencePlugin();
    ~MultiDeviceInferencePlugin() = default;

    InferenceEngine::IExecutableNetworkInternal::Ptr LoadExeNetworkImpl(const InferenceEngine::CNNNetwork&        network,
                                                                       const std::map<std::string, std::string>& config) override;

    ov::SoPtr<InferenceEngine::IExecutableNetworkInternal> LoadNetwork(const std::string& modelPath,
                                                                 const std::map<std::string, std::string>& config) override;

    void SetConfig(const std::map<std::string, std::string>& config) override;
    InferenceEngine::Parameter GetConfig(const std::string& name, const std::map<std::string, InferenceEngine::Parameter> & options) const override;
    InferenceEngine::QueryNetworkResult QueryNetwork(const InferenceEngine::CNNNetwork&        network,
                                                     const std::map<std::string, std::string>& config) const override;
    InferenceEngine::Parameter GetMetric(const std::string& name,
                                         const std::map<std::string, InferenceEngine::Parameter>& options) const override;

    MOCKTESTMACRO std::vector<MultiDevicePlugin::DeviceInformation> ParseMetaDevices(const std::string & devicesRequestsCfg,
                                                                       const std::map<std::string, std::string> & config) const;

    MOCKTESTMACRO std::string GetDeviceList(const std::map<std::string, std::string>& config) const;

    MOCKTESTMACRO std::list<DeviceInformation> GetValidDevice(const std::vector<DeviceInformation>& metaDevices,
                                                   const std::string& networkPrecision = METRIC_VALUE(FP32));

    MOCKTESTMACRO DeviceInformation SelectDevice(const std::vector<DeviceInformation>& metaDevices,
                                                 const std::string& networkPrecision = METRIC_VALUE(FP32),
                                                 unsigned int priority = 0);
    void UnregisterPriority(const unsigned int& priority, const std::string& deviceName);
    void RegisterPriority(const unsigned int& priority, const std::string& deviceName);

protected:
    std::map<std::string, std::string> GetSupportedConfig(const std::map<std::string, std::string>& config,
                                                          const MultiDevicePlugin::DeviceName & deviceName) const;

    ov::AnyMap PreProcessConfig(const std::map<std::string, std::string>& orig_config) const;

private:
    InferenceEngine::IExecutableNetworkInternal::Ptr LoadNetworkImpl(const std::string& modelPath,
                                                                       InferenceEngine::CNNNetwork network,
                                                                       const std::map<std::string, std::string>& config,
                                                                       const std::string &networkPrecision = METRIC_VALUE(FP32));
    std::vector<DeviceInformation> FilterDevice(const std::vector<DeviceInformation>& metaDevices,
                                                const std::map<std::string, std::string>& config);
    std::vector<DeviceInformation> FilterDeviceByNetwork(const std::vector<DeviceInformation>& metaDevices,
                                                         InferenceEngine::CNNNetwork network);
    std::string GetLogTag() const noexcept;
    static std::mutex _mtx;
    static std::map<unsigned int, std::list<std::string>> _priorityMap;
    std::string _LogTag;
    PluginConfig _pluginConfig;
};

}  // namespace MultiDevicePlugin
