// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>

#include <ie_common.h>
#include <cpp/ie_plugin_cpp.hpp>
#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>

#include "hetero_infer_request.h"
#include "cnn_network_impl.hpp"
#include "hetero_async_infer_request.h"

namespace HeteroPlugin {

/**
 * @class ExecutableNetwork
 * @brief Interface of executable network
 */
class HeteroExecutableNetwork : public InferenceEngine::ExecutableNetworkThreadSafeDefault {
public:
    typedef std::shared_ptr<HeteroExecutableNetwork> Ptr;

    /**
    * @brief constructor
    */
    HeteroExecutableNetwork(InferenceEngine::ICNNNetwork &network,
                            const std::map<std::string, std::string> &config,
                            const std::vector<InferenceEngine::IExtensionPtr> &extensions,
                            InferenceEngine::MapDeviceLoaders &deviceLoaders,
                            InferenceEngine::IErrorListener *listener);

    virtual ~HeteroExecutableNetwork() = default;

    /**
     * this functino implements the loading of hetero network,
     * performs split to subgraphs and prepare intermediate blobs
     *
     * @param network
     * @param extensions
     */
    void load(InferenceEngine::ICNNNetwork &network,
              const std::map<std::string, std::string> &config,
              const std::vector<InferenceEngine::IExtensionPtr> &extensions,
              InferenceEngine::IErrorListener *listener);

    InferenceEngine::InferRequestInternal::Ptr CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                                                      InferenceEngine::OutputsDataMap networkOutputs) override;

    void CreateInferRequest(InferenceEngine::IInferRequest::Ptr &asyncRequest) override;

private:
    struct NetworkDesc {
        std::string _device;
        InferenceEngine::details::CNNNetworkImplPtr _clonedNetwork;
        InferenceEngine::IHeteroDeviceLoader::Ptr _deviceLoader;
        InferenceEngine::ExecutableNetwork::Ptr network;
        std::unordered_set<std::string> _oNames;
        std::unordered_set<std::string> _iNames;
    };
    std::vector<NetworkDesc> networks;

    InferenceEngine::MapDeviceLoaders &_deviceLoaders;
};

}  // namespace HeteroPlugin
