// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <map>

#include "cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp"
#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"

namespace ov {
namespace autobatch_plugin {

struct DeviceInformation {
    std::string device_name;
    std::map<std::string, std::string> config;
    int batch_for_device;
};

class Plugin : public InferenceEngine::IInferencePlugin {
public:
    Plugin();

    virtual ~Plugin() = default;

    InferenceEngine::IExecutableNetworkInternal::Ptr LoadExeNetworkImpl(
        const InferenceEngine::CNNNetwork& network,
        const std::map<std::string, std::string>& config) override;

    InferenceEngine::IExecutableNetworkInternal::Ptr LoadExeNetworkImpl(
        const InferenceEngine::CNNNetwork& network,
        const std::shared_ptr<InferenceEngine::RemoteContext>& context,
        const std::map<std::string, std::string>& config) override;

    void SetConfig(const std::map<std::string, std::string>& config) override;

    void CheckConfig(const std::map<std::string, std::string>& config);

    InferenceEngine::Parameter GetConfig(
        const std::string& name,
        const std::map<std::string, InferenceEngine::Parameter>& options) const override;

    InferenceEngine::QueryNetworkResult QueryNetwork(const InferenceEngine::CNNNetwork& network,
                                                     const std::map<std::string, std::string>& config) const override;
    InferenceEngine::Parameter GetMetric(
        const std::string& name,
        const std::map<std::string, InferenceEngine::Parameter>& options) const override;

    InferenceEngine::RemoteContext::Ptr CreateContext(const InferenceEngine::ParamMap&) override;

#ifdef AUTOBATCH_UNITTEST

public:
#else

protected:
#endif
    DeviceInformation ParseMetaDevice(const std::string& devicesBatchCfg,
                                      const std::map<std::string, std::string>& config) const;

    static DeviceInformation ParseBatchDevice(const std::string& deviceWithBatch);

    InferenceEngine::IExecutableNetworkInternal::Ptr LoadNetworkImpl(
        const InferenceEngine::CNNNetwork& network,
        const std::shared_ptr<InferenceEngine::RemoteContext> context,
        const std::map<std::string, std::string>& config);
};
}  // namespace autobatch_plugin
}  // namespace ov