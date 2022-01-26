// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>
#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>

#include "mkldnn_graph.h"
#include "mkldnn_extension_mngr.h"
#include <threading/ie_thread_local.hpp>

#include <vector>
#include <memory>
#include <map>
#include <string>
#include <unordered_map>

namespace MKLDNNPlugin {

class MKLDNNExecNetwork: public InferenceEngine::ExecutableNetworkThreadSafeDefault {
public:
    typedef std::shared_ptr<MKLDNNExecNetwork> Ptr;

    std::shared_ptr<InferenceEngine::IInferRequestInternal>
    CreateInferRequestImpl(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                           const std::vector<std::shared_ptr<const ov::Node>>& outputs) override;

    std::shared_ptr<InferenceEngine::IInferRequestInternal>
    CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                           InferenceEngine::OutputsDataMap networkOutputs) override;

    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequest() override;

    MKLDNNExecNetwork(const InferenceEngine::CNNNetwork &network, const Config &cfg,
                      const MKLDNNExtensionManager::Ptr &extMgr, NumaNodesWeights &weightsSharing);

    void setProperty(const std::map<std::string, std::string> &properties);

    InferenceEngine::Parameter GetConfig(const std::string &name) const override;

    InferenceEngine::Parameter GetMetric(const std::string &name) const override;

    std::shared_ptr<ngraph::Function> GetExecGraphInfo() override;

    void Export(std::ostream& modelStream) override;

protected:
    friend class MKLDNNInferRequestBase;
    MKLDNNExtensionManager::Ptr extensionManager;
    std::vector<InferenceEngine::IVariableStateInternal::Ptr> memoryStates;
    const InferenceEngine::CNNNetwork           _network;
    mutable std::mutex                          _cfgMutex;
    Config                                      _cfg;
    std::atomic_int                             _numRequests = {0};
    std::string                                 _name;
    struct Graph : public MKLDNNGraph {
        std::mutex  _mutex;
        struct Lock : public std::unique_lock<std::mutex> {
            explicit Lock(Graph& graph) : std::unique_lock<std::mutex>(graph._mutex), _graph(graph) {}
            Graph&                          _graph;
        };
    };

    // WARNING: Do not use _graphs directly.
    mutable std::deque<Graph>                   _graphs;
    NumaNodesWeights&                           _numaNodesWeights;

    /* WARNING: Use GetGraph() function to get access to graph in current stream.
     * NOTE: Main thread is interpreted as master thread of external stream so use this function to get access to graphs
     *       even from main thread
     */
    Graph::Lock GetGraph() const;


    bool CanProcessDynBatch(const InferenceEngine::CNNNetwork &network) const;
};

}  // namespace MKLDNNPlugin
