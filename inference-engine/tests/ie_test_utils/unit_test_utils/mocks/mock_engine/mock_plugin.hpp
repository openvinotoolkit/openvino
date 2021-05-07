// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>

#include <cpp_interfaces/impl/ie_plugin_internal.hpp>
#include <ie_icnn_network.hpp>

class MockPlugin : public InferenceEngine::InferencePluginInternal {
    InferenceEngine::IInferencePlugin * _target = nullptr;

public:
    explicit MockPlugin(InferenceEngine::IInferencePlugin*target);

    void SetConfig(const std::map<std::string, std::string>& config) override;

    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal>
    LoadNetwork(const InferenceEngine::CNNNetwork &network,
                const std::map<std::string, std::string> &config) override;

    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal>
    LoadNetwork(const InferenceEngine::CNNNetwork& network,
                const std::map<std::string, std::string>& config,
                InferenceEngine::RemoteContext::Ptr context) override;

    std::shared_ptr<InferenceEngine::ExecutableNetworkInternal>
    LoadExeNetworkImpl(const InferenceEngine::CNNNetwork& network,
                       const std::map<std::string, std::string>& config) override;

    std::shared_ptr<InferenceEngine::ExecutableNetworkInternal>
    ImportNetworkImpl(std::istream& networkModel,
        const std::map<std::string, std::string>& config) override;

    std::shared_ptr<InferenceEngine::ExecutableNetworkInternal>
    ImportNetworkImpl(std::istream& networkModel,
        const InferenceEngine::RemoteContext::Ptr& context,
        const std::map<std::string, std::string>& config) override;

    InferenceEngine::Parameter GetMetric(const std::string& name,
                        const std::map<std::string, InferenceEngine::Parameter>& options) const override;

    InferenceEngine::RemoteContext::Ptr GetDefaultContext(const InferenceEngine::ParamMap& params) override;

    InferenceEngine::QueryNetworkResult QueryNetwork(const InferenceEngine::CNNNetwork& network,
                                                     const std::map<std::string, std::string>& config) const override;

    std::map<std::string, std::string> config;
};
