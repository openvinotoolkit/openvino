// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <inference_engine.hpp>
#include <description_buffer.hpp>
#include <cpp_interfaces/impl/ie_plugin_internal.hpp>

#include <memory>
#include <string>
#include <map>
#include <unordered_map>
#include <vector>

#include "template_executable_network.hpp"
#include "template_config.hpp"

//! [plugin:header]
namespace TemplatePlugin {

class Plugin : public InferenceEngine::InferencePluginInternal {
public:
    using Ptr = std::shared_ptr<Plugin>;

    Plugin();
    ~Plugin() override = default;

    void SetConfig(const std::map<std::string, std::string> &config) override;
    void QueryNetwork(const InferenceEngine::ICNNNetwork &network,
                      const std::map<std::string, std::string>& config,
                      InferenceEngine::QueryNetworkResult &res) const override;
    InferenceEngine::ExecutableNetworkInternal::Ptr
    LoadExeNetworkImpl(const InferenceEngine::ICNNNetwork &network,
                       const std::map<std::string, std::string> &config) override;
    void AddExtension(InferenceEngine::IExtensionPtr extension) override;
    InferenceEngine::Parameter GetConfig(const std::string& name, const std::map<std::string, InferenceEngine::Parameter> & options) const override;
    InferenceEngine::Parameter GetMetric(const std::string& name, const std::map<std::string, InferenceEngine::Parameter> & options) const override;
    InferenceEngine::ExecutableNetwork ImportNetworkImpl(std::istream& model, const std::map<std::string, std::string>& config) override;

private:
    Configuration                    _cfg;
};

}  // namespace TemplatePlugin
//! [plugin:header]