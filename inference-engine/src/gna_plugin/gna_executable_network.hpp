// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <memory>
#include <string>
#include <map>
#include <vector>

#include "gna_infer_request.hpp"
#include "gna_plugin.hpp"
#include <gna/gna_config.hpp>
#include <threading/ie_executor_manager.hpp>
#include <cpp_interfaces/interface/ie_iexecutable_network_internal.hpp>

namespace GNAPluginNS {

class GNAExecutableNetwork : public InferenceEngine::IExecutableNetworkInternal {
    std::shared_ptr<GNAPlugin> plg;

 public:
     GNAExecutableNetwork(const std::string& aotFileName, std::shared_ptr<GNAPlugin> plg)
         : plg(plg) {
         std::fstream inputStream(aotFileName, std::ios_base::in | std::ios_base::binary);
         if (inputStream.fail()) {
             THROW_GNA_EXCEPTION << "Cannot open file to import model: " << aotFileName;
         }

         plg->ImportNetwork(inputStream);
         _networkInputs = plg->GetInputs();
         _networkOutputs = plg->GetOutputs();
     }

    GNAExecutableNetwork(std::istream& networkModel, std::shared_ptr<GNAPlugin> plg)
        : plg(plg) {
        plg->ImportNetwork(networkModel);
        _networkInputs = plg->GetInputs();
        _networkOutputs = plg->GetOutputs();
    }

    GNAExecutableNetwork(InferenceEngine::CNNNetwork &network, std::shared_ptr<GNAPlugin> plg)
        : plg(plg) {
        plg->LoadNetwork(network);
    }

    GNAExecutableNetwork(const std::string& aotFileName, const std::map<std::string, std::string>& config)
        : GNAExecutableNetwork(aotFileName, std::make_shared<GNAPlugin>(config)) {
    }

    GNAExecutableNetwork(InferenceEngine::CNNNetwork &network, const std::map<std::string, std::string> &config)
        : GNAExecutableNetwork(network, std::make_shared<GNAPlugin>(config)) {
    }

    InferenceEngine::IInferRequestInternal::Ptr
        CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                               InferenceEngine::OutputsDataMap networkOutputs) override {
        return std::make_shared<GNAInferRequest>(plg, networkInputs, networkOutputs);
    }

    INFERENCE_ENGINE_DEPRECATED("Use InferRequest::QueryState instead")
    std::vector<InferenceEngine::IVariableStateInternal::Ptr>  QueryState() override {
        IE_SUPPRESS_DEPRECATED_START
        return plg->QueryState();
        IE_SUPPRESS_DEPRECATED_END
    }

    void Export(const std::string &modelFileName) override {
        plg->Export(modelFileName);
    }

    void Export(std::ostream& modelStream) override {
        plg->Export(modelStream);
    }

    void SetConfig(const std::map<std::string, InferenceEngine::Parameter>& config) override {
        using namespace InferenceEngine::GNAConfigParams;
        if (config.empty()) {
            IE_THROW() << "The list of configuration values is empty";
        }
        for (auto&& item : config) {
            if (item.first != KEY_GNA_DEVICE_MODE) {
                IE_THROW() << "The following config value cannot be changed dynamically for ExecutableNetwork in the GNA plugin: "
                                   << item.first << ". Only " << KEY_GNA_DEVICE_MODE << " is supported.";
            }
        }

        InferenceEngine::Parameter old_mode_parameter = GetConfig(KEY_GNA_DEVICE_MODE);
        auto old_mode = old_mode_parameter.as<std::string>();
        if (old_mode == InferenceEngine::GNAConfigParams::GNA_SW_FP32) {
            IE_THROW() << "Dynamic switching from GNA_SW_FP32 mode is not supported for ExecutableNetwork.";
        }

        auto new_mode = config.begin()->second.as<std::string>();
        if (new_mode == InferenceEngine::GNAConfigParams::GNA_SW_FP32) {
            IE_THROW() << "Dynamic switching to GNA_SW_FP32 mode is not supported for ExecutableNetwork.";
        }

        std::map<std::string, std::string> configForPlugin;
        configForPlugin[KEY_GNA_DEVICE_MODE] = new_mode;
        plg->SetConfig(configForPlugin);
    }

    InferenceEngine::Parameter GetConfig(const std::string &name) const override {
        return plg->GetConfig(name, {});
    }

    InferenceEngine::Parameter GetMetric(const std::string& name) const override {
        return plg->GetMetric(name, {});
    }
};

}  // namespace GNAPluginNS
