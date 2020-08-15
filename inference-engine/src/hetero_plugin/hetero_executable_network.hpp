// Copyright (C) 2018-2020 Intel Corporation
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
#include <legacy/cnn_network_impl.hpp>
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
    HeteroExecutableNetwork(const InferenceEngine::ICNNNetwork&         network,
                            const std::map<std::string, std::string>&   config,
                            Engine*                                     plugin);
    /**
    * @brief Import from opened file constructor
    */
    HeteroExecutableNetwork(std::istream&                               heteroModel,
                            const std::map<std::string, std::string>&   config,
                            Engine*                                     plugin);

    ~HeteroExecutableNetwork() override = default;

    InferenceEngine::InferRequestInternal::Ptr CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                                                      InferenceEngine::OutputsDataMap networkOutputs) override;

    void CreateInferRequest(InferenceEngine::IInferRequest::Ptr &asyncRequest) override;

    void GetConfig(const std::string &name, InferenceEngine::Parameter &result, InferenceEngine::ResponseDesc *resp) const override;

    void GetMetric(const std::string &name, InferenceEngine::Parameter &result, InferenceEngine::ResponseDesc *resp) const override;

    void ExportImpl(std::ostream& modelFile) override;

private:
    void InitCNNImpl(const InferenceEngine::ICNNNetwork&    network);

    void InitNgraph(const InferenceEngine::ICNNNetwork&     network);

    struct NetworkDesc {
        std::string                                 _device;
        InferenceEngine::CNNNetwork                 _clonedNetwork;
        InferenceEngine::ExecutableNetwork          _network;
    };
    std::vector<NetworkDesc> networks;

    Engine*                             _heteroPlugin;
    std::string                         _name;
    std::map<std::string, std::string>  _config;
    std::unordered_map<std::string, std::string> _blobNameMap;
};

}  // namespace HeteroPlugin
