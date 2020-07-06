// Copyright (C) 2018-2020 Intel Corporation
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
#include <gna/gna_config.hpp>
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
        if (network.getFunction()) {
            auto convertedNetwork = std::make_shared<InferenceEngine::details::CNNNetworkImpl>(network);
            plg->LoadNetwork(*convertedNetwork);
        } else {
            plg->LoadNetwork(network);
        }
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

    void SetConfig(const std::map<std::string, InferenceEngine::Parameter>& config,
                   InferenceEngine::ResponseDesc* /* resp */) override {
        using namespace InferenceEngine::GNAConfigParams;
        if (config.empty()) {
            THROW_IE_EXCEPTION << "The list of configuration values is empty";
        }
        for (auto&& item : config) {
            if (item.first != KEY_GNA_DEVICE_MODE) {
                THROW_IE_EXCEPTION << "The following config value cannot be changed dynamically for ExecutableNetwork in the GNA plugin: "
                                   << item.first << ". Only " << KEY_GNA_DEVICE_MODE << " is supported.";
            }
        }

        InferenceEngine::Parameter old_mode_parameter;
        GetConfig(KEY_GNA_DEVICE_MODE, old_mode_parameter, {});
        auto old_mode = old_mode_parameter.as<std::string>();
        if (old_mode == InferenceEngine::GNAConfigParams::GNA_SW_FP32) {
            THROW_IE_EXCEPTION << "Dynamic switching from GNA_SW_FP32 mode is not supported for ExecutableNetwork.";
        }

        auto new_mode = config.begin()->second.as<std::string>();
        if (new_mode == InferenceEngine::GNAConfigParams::GNA_SW_FP32) {
            THROW_IE_EXCEPTION << "Dynamic switching to GNA_SW_FP32 mode is not supported for ExecutableNetwork.";
        }

        std::map<std::string, std::string> configForPlugin;
        configForPlugin[KEY_GNA_DEVICE_MODE] = new_mode;
        plg->SetConfig(configForPlugin);
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
