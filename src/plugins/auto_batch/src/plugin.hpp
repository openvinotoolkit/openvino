// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <map>

#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"
#include "ie_metric_helpers.hpp"

#ifdef AUTOBATCH_UNITTEST
#    define AutoBatchPlugin MockAutoBatchPlugin
#endif

namespace AutoBatchPlugin {

using DeviceName = std::string;

struct DeviceInformation {
    DeviceName deviceName;
    std::map<std::string, std::string> config;
    int batchForDevice;
};

class AutoBatchInferencePlugin : public InferenceEngine::IInferencePlugin {
public:
    AutoBatchInferencePlugin();
    virtual ~AutoBatchInferencePlugin() = default;
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
}  // namespace AutoBatchPlugin