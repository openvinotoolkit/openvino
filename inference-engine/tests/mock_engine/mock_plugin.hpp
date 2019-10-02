// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <inference_engine.hpp>
#include <ie_plugin_ptr.hpp>
#include <ie_icnn_network.hpp>

class MockPlugin : public InferenceEngine::IInferencePlugin {
    InferenceEngine::IInferencePlugin * _target = nullptr;
public:
    MockPlugin(InferenceEngine::IInferencePlugin*target);
    void GetVersion(const InferenceEngine::Version *& versionInfo) noexcept override;
    void Release() noexcept override;
    void SetLogCallback(InferenceEngine::IErrorListener & listener) noexcept override;

    InferenceEngine::StatusCode AddExtension(InferenceEngine::IExtensionPtr extension, InferenceEngine::ResponseDesc *resp) noexcept override;

    InferenceEngine::StatusCode SetConfig(const std::map<std::string, std::string>& config,
                                          InferenceEngine::ResponseDesc* resp) noexcept override;

    InferenceEngine::StatusCode
    LoadNetwork(InferenceEngine::IExecutableNetwork::Ptr &ret, InferenceEngine::ICNNNetwork &network,
                const std::map<std::string, std::string> &config, InferenceEngine::ResponseDesc *resp) noexcept override;

    InferenceEngine::StatusCode
    ImportNetwork(InferenceEngine::IExecutableNetwork::Ptr &ret, const std::string &modelFileName,
                  const std::map<std::string, std::string> &config, InferenceEngine::ResponseDesc *resp) noexcept override;

    std::map<std::string, std::string> config;
};
