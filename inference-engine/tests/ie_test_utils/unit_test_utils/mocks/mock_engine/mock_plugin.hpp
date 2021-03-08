// Copyright (C) 2018-2020 Intel Corporation
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
    InferenceEngine::ExecutableNetwork
    LoadNetwork(const InferenceEngine::CNNNetwork &network,
                const std::map<std::string, std::string> &config) override;
    InferenceEngine::ExecutableNetworkInternal::Ptr
    LoadExeNetworkImpl(const InferenceEngine::CNNNetwork& network,
                       const std::map<std::string, std::string>& config) override;

    std::map<std::string, std::string> config;
};
