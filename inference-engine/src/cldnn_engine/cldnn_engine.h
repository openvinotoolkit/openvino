// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "inference_engine.hpp"
#include <map>
#include <string>
#include <memory>
#include <api/engine.hpp>
#include <cpp_interfaces/impl/ie_plugin_internal.hpp>

namespace CLDNNPlugin {

using CLDNNCustomLayerPtr = std::shared_ptr<class CLDNNCustomLayer>;

class clDNNEngine : public InferenceEngine::InferencePluginInternal {
    struct impl;
    std::shared_ptr<impl> _impl;

    cldnn::engine_info engine_info;
public:
    clDNNEngine();

    InferenceEngine::ExecutableNetworkInternal::Ptr LoadExeNetworkImpl(const InferenceEngine::ICore * core, InferenceEngine::ICNNNetwork &network,
                                                                       const std::map<std::string, std::string> &config) override;

    void SetConfig(const std::map<std::string, std::string> &config) override;
    InferenceEngine::Parameter GetConfig(const std::string& name, const std::map<std::string, InferenceEngine::Parameter>& options) const override;
    InferenceEngine::Parameter GetMetric(const std::string& name, const std::map<std::string, InferenceEngine::Parameter>& options) const override;
    /**
     * @deprecated Use the version with config parameter
     */
    void QueryNetwork(const InferenceEngine::ICNNNetwork& network, InferenceEngine::QueryNetworkResult& res) const override;
    void QueryNetwork(const InferenceEngine::ICNNNetwork& network,
                      const std::map<std::string, std::string>& config, InferenceEngine::QueryNetworkResult& res) const override;
};

};  // namespace CLDNNPlugin
