// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>
#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>

#include "graph.h"
#include "extension_mngr.h"
#include <threading/ie_thread_local.hpp>

#include <vector>
#include <memory>
#include <map>
#include <string>
#include <unordered_map>

namespace ov {
namespace intel_cpu {

class ExecNetwork: public InferenceEngine::ExecutableNetworkThreadSafeDefault {
public:
    typedef std::shared_ptr<ExecNetwork> Ptr;

    std::shared_ptr<InferenceEngine::IInferRequestInternal>
    CreateInferRequestImpl(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                           const std::vector<std::shared_ptr<const ov::Node>>& outputs) override;

    std::shared_ptr<InferenceEngine::IInferRequestInternal>
    CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                           InferenceEngine::OutputsDataMap networkOutputs) override;

    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequest() override;

    ExecNetwork(const InferenceEngine::CNNNetwork &network, const Config &cfg,
                const ExtensionManager::Ptr &extMgr,
                const std::shared_ptr<InferenceEngine::IInferencePlugin>& plugin);

    void setProperty(const std::map<std::string, std::string> &properties);

    InferenceEngine::Parameter GetConfig(const std::string &name) const override;

    InferenceEngine::Parameter GetMetric(const std::string &name) const override;

    std::shared_ptr<ngraph::Function> GetExecGraphInfo() override;

    void Export(std::ostream& modelStream) override;

protected:
    friend class InferRequestBase;
    ExtensionManager::Ptr extensionManager;
    std::vector<InferenceEngine::IVariableStateInternal::Ptr> memoryStates;
    const InferenceEngine::CNNNetwork           _network;
    // Generic synchronization primitive on ExecNetwork level.
    // Usage example: helps to avoid data races during CPU Graph initialization in multi-streams scenario
    mutable std::shared_ptr<std::mutex>         _mutex;
    Config                                      _cfg;
    std::atomic_int                             _numRequests = {0};
    std::string                                 _name;
    struct GraphGuard : public Graph {
        std::mutex  _mutex;
        struct Lock : public std::unique_lock<std::mutex> {
            explicit Lock(GraphGuard& graph) : std::unique_lock<std::mutex>(graph._mutex), _graph(graph) {}
            GraphGuard& _graph;
        };
    };

    // WARNING: Do not use _graphs directly.
    mutable std::deque<GraphGuard>              _graphs;
    mutable NumaNodesWeights                    _numaNodesWeights;

    /* WARNING: Use GetGraph() function to get access to graph in current stream.
     * NOTE: Main thread is interpreted as master thread of external stream so use this function to get access to graphs
     *       even from main thread
     */
    GraphGuard::Lock GetGraph() const;

    bool canBeExecViaLegacyDynBatch(std::shared_ptr<const ov::Model> function, int64_t& maxBatchSize) const;
    bool CanProcessDynBatch(const InferenceEngine::CNNNetwork &network) const;

    bool isLegacyAPI() const;

    InferenceEngine::Parameter GetConfigLegacy(const std::string &name) const;

    InferenceEngine::Parameter GetMetricLegacy(const std::string &name, const GraphGuard& graph) const;
};

}   // namespace intel_cpu
}   // namespace ov

