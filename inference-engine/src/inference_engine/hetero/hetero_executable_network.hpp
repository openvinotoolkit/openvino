// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header file for ExecutableNetwork
 * @file dlia_executable_network.hpp
 */
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

#include "hetero_infer_request.hpp"
#include "ie_icore.hpp"
#include "cnn_network_impl.hpp"
#include "hetero_async_infer_request.hpp"

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
                            const InferenceEngine::ICore * core,
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
              const InferenceEngine::ICore * core,
              const std::map<std::string, std::string> &config,
              const std::vector<InferenceEngine::IExtensionPtr> &extensions,
              InferenceEngine::IErrorListener *listener);

    InferenceEngine::InferRequestInternal::Ptr CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                                                      InferenceEngine::OutputsDataMap networkOutputs) override;

    void CreateInferRequest(InferenceEngine::IInferRequest::Ptr &asyncRequest) override;

    void GetConfig(const std::string &name, InferenceEngine::Parameter &result, InferenceEngine::ResponseDesc *resp) const override;

    void GetMetric(const std::string &name, InferenceEngine::Parameter &result, InferenceEngine::ResponseDesc *resp) const override;

private:
    struct NetworkDesc {
        std::string _device;
        InferenceEngine::details::CNNNetworkImplPtr _clonedNetwork;
        IE_SUPPRESS_DEPRECATED_START
        InferenceEngine::IHeteroDeviceLoader::Ptr _deviceLoader;
        IE_SUPPRESS_DEPRECATED_END
        InferenceEngine::ExecutableNetwork::Ptr network;
        std::unordered_set<std::string> _oNames;
        std::unordered_set<std::string> _iNames;
    };
    std::vector<NetworkDesc> networks;

    InferenceEngine::MapDeviceLoaders &_deviceLoaders;
    std::string _name;
    std::vector<std::string> _affinities;
    std::map<std::string, std::string> _config;
};

}  // namespace HeteroPlugin
