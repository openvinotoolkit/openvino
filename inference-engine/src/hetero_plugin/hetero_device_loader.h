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
