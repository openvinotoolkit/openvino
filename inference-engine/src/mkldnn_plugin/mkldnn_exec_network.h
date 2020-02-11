// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>

#include "mkldnn_graph.h"
#include "mkldnn_extension_mngr.h"

#include <vector>
#include <memory>
#include <map>
#include <string>

namespace MKLDNNPlugin {

class MKLDNNExecNetwork: public InferenceEngine::ExecutableNetworkThreadSafeDefault {
public:
    typedef std::shared_ptr<MKLDNNExecNetwork> Ptr;

    InferenceEngine::InferRequestInternal::Ptr
    CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
              InferenceEngine::OutputsDataMap networkOutputs) override;

    void CreateInferRequest(InferenceEngine::IInferRequest::Ptr &asyncRequest) override;

    MKLDNNExecNetwork(const InferenceEngine::ICNNNetwork &network, const Config &cfg,
                      const MKLDNNExtensionManager::Ptr& extMgr);

    virtual ~MKLDNNExecNetwork() {
        graphs.clear();
        extensionManager.reset();
    }

    void setProperty(const std::map<std::string, std::string> &properties);

    void GetConfig(const std::string &name, Parameter &result, ResponseDesc *resp) const override;

    void GetMetric(const std::string &name, Parameter &result, ResponseDesc *resp) const override;

    void GetExecGraphInfo(InferenceEngine::ICNNNetwork::Ptr &graphPtr) override;

    std::vector<IMemoryStateInternal::Ptr> QueryState() override;

protected:
    MKLDNNExtensionManager::Ptr extensionManager;
    std::vector<MKLDNNGraph::Ptr> graphs;
    std::vector<IMemoryStateInternal::Ptr> memoryStates;

    bool CanProcessDynBatch(const InferenceEngine::ICNNNetwork &network) const;
};

}  // namespace MKLDNNPlugin