//
// Copyright (C) 2018-2019 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
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
