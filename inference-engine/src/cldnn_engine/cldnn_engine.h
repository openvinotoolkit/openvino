// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "inference_engine.hpp"
#include <map>
#include <string>
#include <memory>
#include <cpp_interfaces/impl/ie_plugin_internal.hpp>

namespace CLDNNPlugin {

using CLDNNCustomLayerPtr = std::shared_ptr<class CLDNNCustomLayer>;

class clDNNEngine : public InferenceEngine::InferencePluginInternal {
    struct impl;
    impl *_impl;

public:
    clDNNEngine();

    virtual ~clDNNEngine();

    InferenceEngine::ExecutableNetworkInternal::Ptr LoadExeNetworkImpl(InferenceEngine::ICNNNetwork &network,
                                                                       const std::map<std::string, std::string> &config) override;

    void SetConfig(const std::map<std::string, std::string> &config) override;
    /**
     * @depricated Use the version with config parameter
     */
    void QueryNetwork(const InferenceEngine::ICNNNetwork& network, InferenceEngine::QueryNetworkResult& res) const override;
    void QueryNetwork(const InferenceEngine::ICNNNetwork& network,
                      const std::map<std::string, std::string>& config, InferenceEngine::QueryNetworkResult& res) const override;
};

};  // namespace CLDNNPlugin
