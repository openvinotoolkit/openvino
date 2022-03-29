// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>

#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"
#include "ie_icore.hpp"
#include "openvino/core/extension.hpp"

// Plugin doesn't support Subtract operation
class MockPluginAbc : public InferenceEngine::IInferencePlugin {
public:
    explicit MockPluginAbc();

    void SetConfig(const std::map<std::string, std::string>& config) override;

    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> LoadNetwork(
        const InferenceEngine::CNNNetwork& network,
        const std::map<std::string, std::string>& config) override;

    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> LoadNetwork(
        const InferenceEngine::CNNNetwork& network,
        const std::map<std::string, std::string>& config,
        const std::shared_ptr<InferenceEngine::RemoteContext>& context) override;

    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> LoadExeNetworkImpl(
        const InferenceEngine::CNNNetwork& network,
        const std::map<std::string, std::string>& config) override;

    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> LoadNetwork(
        const std::string& modelPath,
        const std::map<std::string, std::string>& config) override;

    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> ImportNetwork(
        std::istream& networkModel,
        const std::map<std::string, std::string>& config) override;

    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> ImportNetwork(
        std::istream& networkModel,
        const std::shared_ptr<InferenceEngine::RemoteContext>& context,
        const std::map<std::string, std::string>& config) override;

    InferenceEngine::Parameter GetConfig(
        const std::string& name,
        const std::map<std::string, InferenceEngine::Parameter>& options) const override;

    InferenceEngine::Parameter GetMetric(
        const std::string& name,
        const std::map<std::string, InferenceEngine::Parameter>& options) const override;

    std::shared_ptr<InferenceEngine::RemoteContext> GetDefaultContext(const InferenceEngine::ParamMap& params) override;

    InferenceEngine::QueryNetworkResult QueryNetwork(const InferenceEngine::CNNNetwork& network,
                                                     const std::map<std::string, std::string>& config) const override;

    void SetCore(std::weak_ptr<InferenceEngine::ICore> core) noexcept override;

    void SetName(const std::string& name) noexcept override;

    std::string GetName() const noexcept override;

    void AddExtension(const ov::Extension::Ptr& extension) override;

private:
    std::map<std::string, std::string> m_config;
    std::vector<ov::Extension::Ptr> m_extensions;
};
