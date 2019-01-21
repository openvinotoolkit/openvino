// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <memory>
#include <string>
#include <map>
#include <vector>

#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>
#include "gna_infer_request.hpp"
#include "gna_plugin.hpp"
#include <cpp_interfaces/ie_executor_manager.hpp>
#include <cpp_interfaces/impl/ie_executable_network_thread_safe_async_only.hpp>

namespace GNAPluginNS {

class GNAExecutableNetwork : public InferenceEngine::ExecutableNetworkThreadSafeAsyncOnly {
    std::shared_ptr<GNAPlugin> plg;

 public:
    GNAExecutableNetwork(const std::string &aotFileName, const std::map<std::string, std::string> &config) :
        plg(std::make_shared<GNAPlugin>(config)) {
        plg->ImportNetwork(aotFileName);
        _networkInputs  = plg->GetInputs();
        _networkOutputs = plg->GetOutputs();
    }

    GNAExecutableNetwork(InferenceEngine::ICNNNetwork &network, const std::map<std::string, std::string> &config)
        : plg(std::make_shared<GNAPlugin>(config)) {
        plg->LoadNetwork(network);
    }

    InferenceEngine::AsyncInferRequestInternal::Ptr
        CreateAsyncInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                    InferenceEngine::OutputsDataMap networkOutputs) override {
        return std::make_shared<GNAInferRequest>(plg, networkInputs, networkOutputs);
    }



    std::vector<InferenceEngine::IMemoryStateInternal::Ptr>  QueryState() override {
        auto pluginStates = plg->QueryState();
        std::vector<InferenceEngine::IMemoryStateInternal::Ptr> state(pluginStates.begin(), pluginStates.end());
        return plg->QueryState();
    }

    void Export(const std::string &modelFileName) override {
        plg->Export(modelFileName);
    }
};
}  // namespace GNAPluginNS
