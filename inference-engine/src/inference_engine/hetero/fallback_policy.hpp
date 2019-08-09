// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <string>
#include <map>
#include <ie_icnn_network.hpp>
#include <ie_icore.hpp>
#include <ie_ihetero_plugin.hpp>
#include <utility>
#include <vector>

namespace InferenceEngine {

class FallbackPolicy {
public:
    FallbackPolicy(InferenceEngine::MapDeviceLoaders& deviceLoaders, bool dumpDotFile,
                   const InferenceEngine::ICore * core);

    void init(const std::string &config, const std::map<std::string, std::string> &allConfigs,
              const std::vector<InferenceEngine::IExtensionPtr> &extensions);

    QueryNetworkResult getAffinities(const std::map<std::string, std::string>& config, const ICNNNetwork& pNetwork) const;
    void setAffinity(const QueryNetworkResult & res, ICNNNetwork& pNetwork) const;

private:
    InferenceEngine::MapDeviceLoaders &_deviceLoaders;
    std::vector<std::string> _fallbackDevices;
    bool _dumpDotFile;
    const InferenceEngine::ICore * _core;
};

}  // namespace InferenceEngine
