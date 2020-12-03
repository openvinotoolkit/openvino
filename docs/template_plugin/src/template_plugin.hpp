// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "template_config.hpp"
#include "template_executable_network.hpp"
#include <cpp_interfaces/impl/ie_plugin_internal.hpp>

#include "backend.hpp"

//! [plugin:header]
namespace TemplatePlugin {

class Plugin : public InferenceEngine::InferencePluginInternal {
public:
    using Ptr = std::shared_ptr<Plugin>;

    Plugin();
    ~Plugin() override;

    void SetConfig(const std::map<std::string, std::string> &config) override;
    InferenceEngine::QueryNetworkResult
    QueryNetwork(const InferenceEngine::CNNNetwork &network,
                 const std::map<std::string, std::string>& config) const override;
    InferenceEngine::ExecutableNetworkInternal::Ptr
    LoadExeNetworkImpl(const InferenceEngine::CNNNetwork &network,
                       const std::map<std::string, std::string> &config) override;
    void AddExtension(InferenceEngine::IExtensionPtr extension) override;
    InferenceEngine::Parameter GetConfig(const std::string& name, const std::map<std::string, InferenceEngine::Parameter> & options) const override;
    InferenceEngine::Parameter GetMetric(const std::string& name, const std::map<std::string, InferenceEngine::Parameter> & options) const override;
    InferenceEngine::ExecutableNetwork ImportNetworkImpl(std::istream& model, const std::map<std::string, std::string>& config) override;

private:
    friend class ExecutableNetwork;
    friend class TemplateInferRequest;

    std::shared_ptr<ngraph::runtime::Backend>   _backend;
    Configuration                               _cfg;
    InferenceEngine::ITaskExecutor::Ptr         _waitExecutor;
};

}  // namespace TemplatePlugin
//! [plugin:header]