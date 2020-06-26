// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>

#include <cpp_interfaces/interface/ie_plugin.hpp>
#include <ie_icnn_network.hpp>

class MockPlugin : public InferenceEngine::IInferencePlugin {
    InferenceEngine::IInferencePlugin * _target = nullptr;
    InferenceEngine::Version version;

public:
    explicit MockPlugin(InferenceEngine::IInferencePlugin*target);

    void GetVersion(const InferenceEngine::Version *& versionInfo) noexcept override;

    InferenceEngine::StatusCode AddExtension(InferenceEngine::IExtensionPtr extension, InferenceEngine::ResponseDesc *resp) noexcept override;

    InferenceEngine::StatusCode SetConfig(const std::map<std::string, std::string>& config,
                                          InferenceEngine::ResponseDesc* resp) noexcept override;


    InferenceEngine::StatusCode
    LoadNetwork(InferenceEngine::IExecutableNetwork::Ptr &ret, const InferenceEngine::ICNNNetwork &network,
                const std::map<std::string, std::string> &config, InferenceEngine::ResponseDesc *resp) noexcept override;

    InferenceEngine::StatusCode
    ImportNetwork(InferenceEngine::IExecutableNetwork::Ptr &ret, const std::string &modelFileName,
                  const std::map<std::string, std::string> &config, InferenceEngine::ResponseDesc *resp) noexcept override;

    void Release() noexcept override;

    std::map<std::string, std::string> config;
};
