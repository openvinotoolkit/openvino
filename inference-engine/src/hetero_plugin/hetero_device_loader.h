// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_ihetero_plugin.hpp"
#include <string>
#include <map>
#include <vector>
#include <cpp/ie_plugin_cpp.hpp>

namespace InferenceEngine {

class HeteroDeviceLoader: public IHeteroDeviceLoader {
public:
    explicit HeteroDeviceLoader(const std::string& deviceId);
    StatusCode LoadNetwork(
        const std::string& device,
        IExecutableNetwork::Ptr &ret,
        ICNNNetwork &network,
        const std::map<std::string, std::string> &config,
        ResponseDesc *resp)noexcept override;

    /**
     * @depricated Use the version with config parameter
     */
    void QueryNetwork(const std::string &device,
                      const ICNNNetwork &network,
                      QueryNetworkResult &res)noexcept override;

    void QueryNetwork(const std::string &device,
                      const ICNNNetwork &network,
                      const std::map<std::string, std::string>& config,
                      QueryNetworkResult &res)noexcept override;

    void initConfigs(const std::map<std::string, std::string> &config,
                     const std::vector<InferenceEngine::IExtensionPtr> &extensions);

    void SetLogCallback(IErrorListener &listener) override;

protected:
    std::string _deviceId;
    InferenceEngine::InferenceEnginePluginPtr _plugin;
};

}  // namespace InferenceEngine
