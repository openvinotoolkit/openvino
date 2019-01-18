// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <map>
#include <cpp_interfaces/impl/ie_plugin_internal.hpp>
#include <cpp_interfaces/impl/ie_executable_network_internal.hpp>
#include "gna_executable_network.hpp"

namespace GNAPluginNS {

class GNAPluginInternal  : public InferenceEngine::InferencePluginInternal {
 public:
    InferenceEngine::ExecutableNetworkInternal::Ptr LoadExeNetworkImpl(InferenceEngine::ICNNNetwork &network,
                                                                       const std::map<std::string, std::string> &config) override {
        return std::make_shared<GNAExecutableNetwork>(network, config);
    }
    void SetConfig(const std::map<std::string, std::string> &config) override {}
    InferenceEngine::IExecutableNetwork::Ptr  ImportNetwork(const std::string &modelFileName,
                                                            const std::map<std::string, std::string> &config) override {
        return make_executable_network(std::make_shared<GNAExecutableNetwork>(modelFileName, config));
    }
};

}  // namespace GNAPluginNS
