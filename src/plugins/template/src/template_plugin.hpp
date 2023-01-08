// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpp_interfaces/interface/ie_iplugin_internal.hpp>

#include "backend.hpp"
#include "template_config.hpp"
#include "template_executable_network.hpp"

//! [plugin:header]
namespace TemplatePlugin {

class Plugin : public InferenceEngine::IInferencePlugin {
public:
    using Ptr = std::shared_ptr<Plugin>;

    Plugin();
    ~Plugin();

    void SetConfig(const std::map<std::string, std::string>& config) override;
    InferenceEngine::QueryNetworkResult QueryNetwork(const InferenceEngine::CNNNetwork& network,
                                                     const std::map<std::string, std::string>& config) const override;
    InferenceEngine::IExecutableNetworkInternal::Ptr LoadExeNetworkImpl(
        const InferenceEngine::CNNNetwork& network,
        const std::map<std::string, std::string>& config) override;
    void AddExtension(const std::shared_ptr<InferenceEngine::IExtension>& extension) override;
    InferenceEngine::Parameter GetConfig(
        const std::string& name,
        const std::map<std::string, InferenceEngine::Parameter>& options) const override;
    InferenceEngine::Parameter GetMetric(
        const std::string& name,
        const std::map<std::string, InferenceEngine::Parameter>& options) const override;
    InferenceEngine::IExecutableNetworkInternal::Ptr ImportNetwork(
        std::istream& model,
        const std::map<std::string, std::string>& config) override;

private:
    friend class ExecutableNetwork;
    friend class TemplateInferRequest;

    std::shared_ptr<ngraph::runtime::Backend> _backend;
    Configuration _cfg;
    InferenceEngine::ITaskExecutor::Ptr _waitExecutor;
};

}  // namespace TemplatePlugin
   //! [plugin:header]
