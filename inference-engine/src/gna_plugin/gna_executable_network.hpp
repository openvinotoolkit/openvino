// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <memory>
#include <string>
#include <map>
#include <vector>

#include <net_pass.h>
#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>
#include "gna_infer_request.hpp"
#include "gna_plugin.hpp"
#include <threading/ie_executor_manager.hpp>
#include <cpp_interfaces/impl/ie_executable_network_thread_safe_async_only.hpp>

namespace GNAPluginNS {

class GNAExecutableNetwork : public InferenceEngine::ExecutableNetworkThreadSafeAsyncOnly {
    std::shared_ptr<GNAPlugin> plg;

 public:
    GNAExecutableNetwork(const std::string &aotFileName, std::shared_ptr<GNAPlugin> plg)
        : plg(plg) {
        plg->ImportNetwork(aotFileName);
        _networkInputs  = plg->GetInputs();
        _networkOutputs = plg->GetOutputs();
    }

    GNAExecutableNetwork(InferenceEngine::ICNNNetwork &network, std::shared_ptr<GNAPlugin> plg)
        : plg(plg) {
        InferenceEngine::NetPass::ConvertPrecision(network, InferenceEngine::Precision::I64, InferenceEngine::Precision::I32);
        InferenceEngine::NetPass::ConvertPrecision(network, InferenceEngine::Precision::U64, InferenceEngine::Precision::I32);
        plg->LoadNetwork(network);
    }

    GNAExecutableNetwork(const std::string &aotFileName, const std::map<std::string, std::string> &config)
        : GNAExecutableNetwork(aotFileName, std::make_shared<GNAPlugin>(config)) {
    }

    GNAExecutableNetwork(InferenceEngine::ICNNNetwork &network, const std::map<std::string, std::string> &config)
        : GNAExecutableNetwork(network, std::make_shared<GNAPlugin>(config)) {
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

    using ExecutableNetworkInternal::Export;

    void ExportImpl(std::ostream&) override {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
    }

    void GetConfig(const std::string &name,
                   InferenceEngine::Parameter &result,
                   InferenceEngine::ResponseDesc* /*resp*/) const override {
        result = plg->GetConfig(name, {});
    }

    void GetMetric(const std::string& name,
                   InferenceEngine::Parameter& result,
                   InferenceEngine::ResponseDesc* /* resp */) const override {
        result = plg->GetMetric(name, {});
    }
};

}  // namespace GNAPluginNS
