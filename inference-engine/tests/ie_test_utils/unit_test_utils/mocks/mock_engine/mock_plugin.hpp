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

    void SetName(const std::string& pluginName) noexcept override;
    std::string GetName() const noexcept override;
    void SetCore(InferenceEngine::ICore* core) noexcept override;
    const InferenceEngine::ICore& GetCore() const override;
    InferenceEngine::Parameter
    GetConfig(const std::string& name, const std::map<std::string, InferenceEngine::Parameter>& options) const override;
    InferenceEngine::Parameter
    GetMetric(const std::string& name, const std::map<std::string, InferenceEngine::Parameter>& options) const override;
    InferenceEngine::RemoteContext::Ptr
    CreateContext(const InferenceEngine::ParamMap& params) override;
    InferenceEngine::RemoteContext::Ptr GetDefaultContext() override;
    InferenceEngine::ExecutableNetwork
    LoadNetwork(const InferenceEngine::ICNNNetwork& network, const std::map<std::string, std::string>& config,
                InferenceEngine::RemoteContext::Ptr context) override;
    InferenceEngine::ExecutableNetwork
    ImportNetwork(std::istream& networkModel, const std::map<std::string, std::string>& config) override;
    InferenceEngine::ExecutableNetwork
    ImportNetwork(std::istream& networkModel, const InferenceEngine::RemoteContext::Ptr& context,
                  const std::map<std::string, std::string>& config) override;

    std::map<std::string, std::string> config;
};
