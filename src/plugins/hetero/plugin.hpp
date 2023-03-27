// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"
#include "description_buffer.hpp"
#include "ie_icore.hpp"

namespace HeteroPlugin {

using Configs = std::map<std::string, std::string>;

template <typename T>
struct ParsedConfig {
    Configs hetero_config;
    T device_config;
};

class Engine : public InferenceEngine::IInferencePlugin {
public:
    using DeviceMetaInformationMap = std::unordered_map<std::string, Configs>;

    Engine();

    InferenceEngine::IExecutableNetworkInternal::Ptr LoadExeNetworkImpl(const InferenceEngine::CNNNetwork& network,
                                                                        const Configs& config) override;

    void SetConfig(const Configs& config) override;

    InferenceEngine::QueryNetworkResult QueryNetwork(const InferenceEngine::CNNNetwork& network,
                                                     const Configs& config) const override;

    InferenceEngine::Parameter GetMetric(const std::string& name, const ov::AnyMap& options) const override;

    InferenceEngine::Parameter GetConfig(const std::string& name, const ov::AnyMap& options) const override;

    InferenceEngine::IExecutableNetworkInternal::Ptr ImportNetwork(std::istream& heteroModel,
                                                                   const Configs& config) override;

    DeviceMetaInformationMap GetDevicePlugins(const std::string& targetFallback, const Configs& localConfig) const;

    std::string GetTargetFallback(const Configs& config, bool raise_exception = true) const;
    std::string GetTargetFallback(const ov::AnyMap& config, bool raise_exception = true) const;

    ParsedConfig<Configs> MergeConfigs(const Configs& user_config) const;
    ParsedConfig<ov::AnyMap> MergeConfigs(const ov::AnyMap& user_config) const;

private:
    std::string DeviceCachingProperties(const std::string& targetFallback) const;

    Configs _device_config;
};

}  // namespace HeteroPlugin
