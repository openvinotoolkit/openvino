// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpp_interfaces/interface/ie_iplugin_internal.hpp>

#include "backend.hpp"
#include "template_rw_properties.hpp"
#include "template_executable_network.hpp"

//! [plugin:header]
namespace TemplatePlugin {

class Plugin : public InferenceEngine::IInferencePlugin {
public:
    using Ptr = std::shared_ptr<Plugin>;

    Plugin();
    ~Plugin();

    InferenceEngine::QueryNetworkResult QueryNetwork(const InferenceEngine::CNNNetwork& network,
                                                     const std::map<std::string, std::string>& config) const override;
    InferenceEngine::IExecutableNetworkInternal::Ptr LoadExeNetworkImpl(
        const InferenceEngine::CNNNetwork& network,
        const std::map<std::string, std::string>& config) override;
    void AddExtension(const std::shared_ptr<InferenceEngine::IExtension>& extension) override;
    InferenceEngine::IExecutableNetworkInternal::Ptr ImportNetwork(
        std::istream& model,
        const std::map<std::string, std::string>& config) override;

private:
    friend class ExecutableNetwork;
    friend class TemplateInferRequest;

    std::shared_ptr<ngraph::runtime::Backend> _backend;
    RwProperties _cfg;
    InferenceEngine::ITaskExecutor::Ptr _waitExecutor;
};

}  // namespace TemplatePlugin
   //! [plugin:header]
