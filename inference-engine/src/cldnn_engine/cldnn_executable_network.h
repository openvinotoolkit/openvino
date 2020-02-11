// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <map>
#include <set>
#include <memory>
#include <string>
#include <utility>
#include "ie_blob.h"
#include "ie_plugin.hpp"
#include "cpp/ie_cnn_network.h"
#include "debug_options.h"
#include "inference_engine.hpp"
#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>
#include "cldnn_graph.h"
#include "cldnn_config.h"
#include "cldnn_remote_context.h"

namespace CLDNNPlugin {

class CLDNNExecNetwork : public InferenceEngine::ExecutableNetworkThreadSafeDefault {
public:
    typedef std::shared_ptr<CLDNNExecNetwork> Ptr;

    explicit CLDNNExecNetwork(InferenceEngine::ICNNNetwork &network, RemoteContext::Ptr context, Config config);

    void GetExecGraphInfo(InferenceEngine::ICNNNetwork::Ptr &graphPtr) override;
    void CreateInferRequest(InferenceEngine::IInferRequest::Ptr &asyncRequest) override;
    InferenceEngine::InferRequestInternal::Ptr CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                                                      InferenceEngine::OutputsDataMap networkOutputs) override;

    static unsigned int GetWaitingCounter();
    static unsigned int GetRunningCounter();
    void GetMetric(const std::string &name, InferenceEngine::Parameter &result, InferenceEngine::ResponseDesc *resp) const override;
    void GetConfig(const std::string &name, InferenceEngine::Parameter &result, InferenceEngine::ResponseDesc *resp) const override;
    void GetContext(RemoteContext::Ptr &pContext, ResponseDesc *resp) const override;

    std::vector<std::shared_ptr<CLDNNGraph>> m_graphs;
    gpu::ClContext::Ptr m_context;
    Config m_config;
};

};  // namespace CLDNNPlugin
