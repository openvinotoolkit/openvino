// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>

#include "mkldnn_graph.h"
#include "mkldnn_extension_mngr.h"
#include <threading/ie_thread_local.hpp>

#include <vector>
#include <memory>
#include <map>
#include <string>
#include <legacy/cnn_network_impl.hpp>
#include <unordered_map>

namespace MKLDNNPlugin {

class MKLDNNExecNetwork: public InferenceEngine::ExecutableNetworkThreadSafeDefault {
public:
    typedef std::shared_ptr<MKLDNNExecNetwork> Ptr;

    InferenceEngine::InferRequestInternal::Ptr
    CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
              InferenceEngine::OutputsDataMap networkOutputs) override;

    void CreateInferRequest(InferenceEngine::IInferRequest::Ptr &asyncRequest) override;

    MKLDNNExecNetwork(const InferenceEngine::ICNNNetwork &network, const Config &cfg,
                      const MKLDNNExtensionManager::Ptr &extMgr, NumaNodesWeights &weightsSharing);

    ~MKLDNNExecNetwork() override = default;

    void setProperty(const std::map<std::string, std::string> &properties);

    void GetConfig(const std::string &name, InferenceEngine::Parameter &result, InferenceEngine::ResponseDesc *resp) const override;

    void GetMetric(const std::string &name, InferenceEngine::Parameter &result, InferenceEngine::ResponseDesc *resp) const override;

    void GetExecGraphInfo(InferenceEngine::ICNNNetwork::Ptr &graphPtr) override;

    std::vector<InferenceEngine::IMemoryStateInternal::Ptr> QueryState() override;

    InferenceEngine::ThreadLocal<MKLDNNGraph::Ptr>  _graphs;

protected:
    friend class MKLDNNInferRequest;
    MKLDNNExtensionManager::Ptr extensionManager;
    std::vector<InferenceEngine::IMemoryStateInternal::Ptr> memoryStates;
    InferenceEngine::details::CNNNetworkImplPtr _clonedNetwork;
    std::mutex                                  _cfgMutex;
    Config                                      _cfg;
    std::atomic_int                             _numRequests = {0};
    std::string                                 _name;


    bool CanProcessDynBatch(const InferenceEngine::ICNNNetwork &network) const;
};

}  // namespace MKLDNNPlugin
