// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <cpp_interfaces/interface/ie_iexecutable_network_internal.hpp>
#include <gna/gna_config.hpp>
#include <ie_icore.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "gna_infer_request.hpp"
#include "gna_plugin.hpp"

namespace ov {
namespace intel_gna {

class GNAExecutableNetwork : public InferenceEngine::IExecutableNetworkInternal {
    std::shared_ptr<GNAPlugin> plg;

public:
    GNAExecutableNetwork(const std::string& aotFileName, std::shared_ptr<GNAPlugin> plg) : plg(plg) {
        std::fstream inputStream(aotFileName, std::ios_base::in | std::ios_base::binary);
        if (inputStream.fail()) {
            THROW_GNA_EXCEPTION << "Cannot open file to import model: " << aotFileName;
        }
        plg->ImportNetwork(inputStream);
        // old API
        setNetworkInputs(plg->GetNetworkInputs());
        setNetworkOutputs(plg->GetNetworkOutputs());
        // new API
        setInputs(plg->GetInputs());
        setOutputs(plg->GetOutputs());
    }

    GNAExecutableNetwork(std::istream& networkModel, std::shared_ptr<GNAPlugin> plg) : plg(plg) {
        plg->ImportNetwork(networkModel);
        // old API
        setNetworkInputs(plg->GetNetworkInputs());
        setNetworkOutputs(plg->GetNetworkOutputs());
        // new API
        setInputs(plg->GetInputs());
        setOutputs(plg->GetOutputs());
    }

    GNAExecutableNetwork(const InferenceEngine::CNNNetwork& network, std::shared_ptr<GNAPlugin> plg) : plg(plg) {
        plg->LoadNetwork(network);
    }

    GNAExecutableNetwork(const std::string& aotFileName, const std::map<std::string, std::string>& config)
        : GNAExecutableNetwork(aotFileName, std::make_shared<GNAPlugin>(config)) {}

    GNAExecutableNetwork(InferenceEngine::CNNNetwork& network, const std::map<std::string, std::string>& config)
        : GNAExecutableNetwork(network, std::make_shared<GNAPlugin>(config)) {}

    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequestImpl(
        InferenceEngine::InputsDataMap networkInputs,
        InferenceEngine::OutputsDataMap networkOutputs) override {
        return std::make_shared<GNAInferRequest>(plg, networkInputs, networkOutputs);
    }

    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequestImpl(
        const std::vector<std::shared_ptr<const ov::Node>>& inputs,
        const std::vector<std::shared_ptr<const ov::Node>>& outputs) override {
        if (!this->_plugin || !_plugin->IsNewAPI())
            return nullptr;
        return std::make_shared<GNAInferRequest>(plg, inputs, outputs);
    }

    void Export(const std::string& modelFileName) override {
        plg->UpdateInputs(getInputs());
        plg->UpdateOutputs(getOutputs());
        plg->Export(modelFileName);
    }

    void Export(std::ostream& modelStream) override {
        plg->UpdateInputs(getInputs());
        plg->UpdateOutputs(getOutputs());
        plg->Export(modelStream);
    }

    void SetConfig(const std::map<std::string, InferenceEngine::Parameter>& config) override {
        using namespace InferenceEngine::GNAConfigParams;
        if (config.empty()) {
            IE_THROW() << "The list of configuration values is empty";
        }

        auto supported_properties = Config::GetSupportedProperties(true).as<std::vector<ov::PropertyName>>();
        for (auto&& item : config) {
            auto it = std::find(supported_properties.begin(), supported_properties.end(), item.first);
            if (it != supported_properties.end()) {
                if (!it->is_mutable()) {
                    IE_THROW() << "The following config value cannot be changed dynamically "
                               << "for compiled model in the GNA plugin: " << item.first;
                }
            } else if (item.first != KEY_GNA_DEVICE_MODE) {
                IE_THROW() << "The following config value cannot be changed dynamically for ExecutableNetwork in the "
                              "GNA plugin: "
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

    InferenceEngine::Parameter GetConfig(const std::string& name) const override {
        return plg->GetConfig(name, {});
    }

    InferenceEngine::Parameter GetMetric(const std::string& name) const override {
        if (ov::supported_properties == name) {
            return Config::GetSupportedProperties(true);
        } else {
            return plg->GetMetric(name, {});
        }
    }
};

}  // namespace intel_gna
}  // namespace ov
