// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <string>
#include <map>
#include <ie_icnn_network.hpp>
#include <ie_ihetero_plugin.hpp>
#include <utility>
#include <vector>

namespace InferenceEngine {

class FallbackPolicy {
public:
    FallbackPolicy(InferenceEngine::MapDeviceLoaders& deviceLoaders, bool dumpDotFile);

    void init(const std::string &config, const std::map<std::string, std::string> &allConfigs,
              const std::vector<InferenceEngine::IExtensionPtr> &extensions);

    void setAffinity(const std::map<std::string, std::string>& config, ICNNNetwork& pNetwork);

private:
    InferenceEngine::MapDeviceLoaders &_deviceLoaders;
    std::vector<std::string> _fallbackDevices;
    bool _dumpDotFile;
};

}  // namespace InferenceEngine
