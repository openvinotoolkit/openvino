// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header file for ExecutableNetwork
 * @file hetero_executable_network.hpp
 */
#pragma once

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>

#include <ie_common.h>
#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>

#include "hetero_infer_request.hpp"
#include "ie_icore.hpp"
#include "hetero_async_infer_request.hpp"

namespace HeteroPlugin {

class Engine;

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
    HeteroExecutableNetwork(const InferenceEngine::CNNNetwork&          network,
                            const std::map<std::string, std::string>&   config,
                            Engine*                                     plugin);
    /**
    * @brief Import from opened file constructor
    */
    HeteroExecutableNetwork(std::istream&                               heteroModel,
                            const std::map<std::string, std::string>&   config,
                            Engine*                                     plugin);

    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                                                       InferenceEngine::OutputsDataMap networkOutputs) override;

    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequest() override;

    InferenceEngine::Parameter GetConfig(const std::string &name) const override;

    InferenceEngine::Parameter GetMetric(const std::string &name) const override;

    void Export(std::ostream& modelFile) override;

private:
    void InitCNNImpl(const InferenceEngine::CNNNetwork&    network);
    void InitNgraph(const InferenceEngine::CNNNetwork&     network);

    struct NetworkDesc {
        std::string                                   _device;
        InferenceEngine::CNNNetwork                   _clonedNetwork;
        InferenceEngine::SoExecutableNetworkInternal  _network;
    };

    std::vector<NetworkDesc>                     _networks;
    Engine*                                      _heteroPlugin;
    std::string                                  _name;
    std::map<std::string, std::string>           _config;
    std::unordered_map<std::string, std::string> _blobNameMap;
};

}  // namespace HeteroPlugin
