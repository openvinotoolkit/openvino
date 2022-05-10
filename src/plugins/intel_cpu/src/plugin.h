// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpp_interfaces/interface/ie_iplugin_internal.hpp>
#include "exec_network.h"

#include <string>
#include <map>
#include <unordered_map>
#include <memory>
#include <functional>
#include <vector>
#include <cfloat>

namespace ov {
namespace intel_cpu {

class Engine : public InferenceEngine::IInferencePlugin {
public:
    Engine();
    ~Engine();

    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal>
    LoadExeNetworkImpl(const InferenceEngine::CNNNetwork &network,
                       const std::map<std::string, std::string> &config) override;

    void AddExtension(const InferenceEngine::IExtensionPtr& extension) override;

    void SetConfig(const std::map<std::string, std::string> &config) override;

    InferenceEngine::Parameter GetConfig(const std::string& name, const std::map<std::string, InferenceEngine::Parameter>& options) const override;

    InferenceEngine::Parameter GetMetric(const std::string& name, const std::map<std::string, InferenceEngine::Parameter>& options) const override;

    InferenceEngine::QueryNetworkResult QueryNetwork(const InferenceEngine::CNNNetwork& network,
                                                     const std::map<std::string, std::string>& config) const override;

    InferenceEngine::IExecutableNetworkInternal::Ptr ImportNetwork(std::istream& networkModel,
                                                     const std::map<std::string, std::string>& config) override;

private:
    bool isLegacyAPI() const;

    InferenceEngine::Parameter GetMetricLegacy(const std::string& name, const std::map<std::string, InferenceEngine::Parameter>& options) const;

    InferenceEngine::Parameter GetConfigLegacy(const std::string& name, const std::map<std::string, InferenceEngine::Parameter>& options) const;

    void ApplyPerformanceHints(std::map<std::string, std::string> &config, const std::shared_ptr<ngraph::Function>& ngraphFunc) const;

    Config engConfig;
    ExtensionManager::Ptr extensionManager = std::make_shared<ExtensionManager>();
    /* Explicily configured streams have higher priority even than performance hints.
       So track if streams is set explicitly (not auto-configured) */
    bool streamsExplicitlySetForEngine = false;
    const std::string deviceFullName;
};

}   // namespace intel_cpu
}   // namespace ov
