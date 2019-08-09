// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_ihetero_plugin.hpp"
#include <string>
#include <map>
#include <vector>
#include <cpp/ie_plugin_cpp.hpp>
#include <ie_icore.hpp>

namespace InferenceEngine {

IE_SUPPRESS_DEPRECATED_START
class HeteroDeviceLoader: public IHeteroDeviceLoader {
public:
    HeteroDeviceLoader(const std::string& deviceId, const ICore * core);

    ~HeteroDeviceLoader() override;

    StatusCode LoadNetwork(
        const std::string& device,
        IExecutableNetwork::Ptr &ret,
        ICNNNetwork &network,
        const std::map<std::string, std::string> &config,
        ResponseDesc *resp)noexcept override;

    void QueryNetwork(const std::string &device,
                      const ICNNNetwork &network,
                      const std::map<std::string, std::string>& config,
                      QueryNetworkResult &res)noexcept override;

    void initConfigs(const std::map<std::string, std::string> &config,
                     const std::vector<InferenceEngine::IExtensionPtr> &extensions);

    void SetLogCallback(IErrorListener &listener) override;

protected:
    const ICore * _core;
    std::string _deviceId;
    std::string _deviceName;
    InferenceEngine::InferenceEnginePluginPtr _plugin;
};
IE_SUPPRESS_DEPRECATED_END

}  // namespace InferenceEngine
